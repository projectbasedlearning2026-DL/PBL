"""pipeline.py - end-to-end orchestration: data -> models -> evaluation -> inventory -> analytics."""
from __future__ import annotations

import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import analytics
from config import Config
from data import (N_SEQ_FEATURES, WindowSampler, block_origins, build_bundle, build_tabular, future_origins,
                  load_dataset, tab_feature_names, train_origins)
from evaluate import (Z90, aggregate_quantiles, build_summing_matrix, interval_coverage, mae, pinball_np, rmse,
                      wape, wrmsse, mase, mint_reconcile)
from conformal import apply_offset, cqr_offset
from inventory import InventoryParams, compute_schedule, interval_from_point, simulate_policy
from models import ContractSparseTransformer, LSTMBaseline, XGBBaseline, naive_forecast
from training import predict_neural, train_neural

MODELS = ["Naive", "XGBoost", "LSTM", "TFT-Sparse"]        # TFT-Sparse = Contract-Sparse Transformer
EXTRA_MODELS = ["XGBoost-L1"]       # extra baseline (median objective), reported in the tables but not shown in the dashboard
ALL_MODELS = MODELS + EXTRA_MODELS



def run_pipeline(cfg: Config, log=print) -> dict:
    print(f"\n{'='*50}\n🚀 RUNNING ON DATASET: {cfg.source.upper()} ({cfg.max_series} series)\n{'='*50}\n")
    t0 = time.time()
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"device={device}")

    # ------------------------------------------------------------------ data
    df = load_dataset(cfg)
    b = build_bundle(df, cfg)
    S, T, H = b.S, b.T, b.H
    log(f"series={S} steps={T} horizon={H} | train<{b.n_train} val<{b.n_val_end} test<{T}")

    tr_s = WindowSampler(b, *train_origins(b, cfg.train_stride), cfg.batch_size, True, device)
    val_o = np.arange(b.n_train, b.n_val_end - H + 1)
    va_s = WindowSampler(b, np.repeat(np.arange(S), len(val_o)), np.tile(val_o, S), 512, False, device)
    val_blk = WindowSampler(b, *block_origins(b, b.n_train), 512, False, device)
    test_blk = WindowSampler(b, *block_origins(b, b.n_val_end), 512, False, device)
    fut_s = WindowSampler(b, *future_origins(b), 512, False, device)

    # ----------------------------------------------------------------- models
    tft = ContractSparseTransformer(N_SEQ_FEATURES, b.cat_cards, 2, H, cfg.d_model, cfg.n_heads, cfg.n_layers, cfg.d_ff,
                                    cfg.dropout, cfg.n_types, cfg.topk_ratio, H if cfg.force_select_horizon else 0, head=cfg.head)
    tft, tft_hist = train_neural(tft, tr_s, va_s, cfg, "tft", device, log)
    lstm = LSTMBaseline(N_SEQ_FEATURES, b.cat_cards, 2, H, cfg.lstm_hidden)
    lstm, lstm_hist = train_neural(lstm, tr_s, va_s, cfg, "lstm", device, log)

    tab = build_tabular(b)
    names = tab_feature_names(b)
    xgbm = XGBBaseline(names)
    xgbm.fit(tab[(tab.t < b.n_train) & tab.y.notna()], tab[(tab.t >= b.n_train) & (tab.t < b.n_val_end)])
    # same features, median (L1) objective: a fair point-forecast baseline for the median-trained transformer
    xgb_l1 = XGBBaseline(names, objective="reg:absoluteerror")
    xgb_l1.fit(tab[(tab.t < b.n_train) & tab.y.notna()], tab[(tab.t >= b.n_train) & (tab.t < b.n_val_end)])

    # ------------------------------------------------------------ predictions
    PT = 3 if cfg.head == "mean" else 1                                # channel used as the TFT point forecast

    def nn_blocks(model, kind, sampler):
        p = predict_neural(model, sampler, kind)                       # [S*2,H,Q] scaled
        p = p.reshape(S, 2, H, -1).reshape(S, 2 * H, -1) * b.scale[:, None, None]
        return np.clip(p, 0, None)

    def xgb_rows(t0_, n, model=None):
        rows = tab[(tab.t >= t0_) & (tab.t < t0_ + n)].sort_values(["series_idx", "t"])
        return (model or xgbm).predict(rows).reshape(S, n)

    def naive_blocks(start):
        return np.concatenate([naive_forecast(b.sales, start + j * H, H, b.season) for j in range(2)], 1)

    y_val, y_test = b.sales[:, b.n_train:b.n_val_end], b.sales[:, b.n_val_end:T]
    val_pt = {"Naive": naive_blocks(b.n_train), "XGBoost": xgb_rows(b.n_train, 2 * H),
              "LSTM": nn_blocks(lstm, "lstm", val_blk)[..., 0], "TFT-Sparse": nn_blocks(tft, "tft", val_blk)[..., PT],
              "XGBoost-L1": xgb_rows(b.n_train, 2 * H, xgb_l1)}
    sigma = {m: np.sqrt(((y_val - p) ** 2).mean(1)) + 1e-6 for m, p in val_pt.items()}
    tft_te = nn_blocks(tft, "tft", test_blk)                           # [S,2H,3]
    tft_va = nn_blocks(tft, "tft", val_blk)                            # [S,2H,3] validation block
    xgb_pred_test = xgb_rows(b.n_val_end, 2 * H)
    
    # TFT-Sparse = the transformer's own predictions. No test labels, no blending with other models.
    test_pt = {"Naive": naive_blocks(b.n_val_end), "XGBoost": xgb_pred_test,
               "LSTM": nn_blocks(lstm, "lstm", test_blk)[..., 0], "TFT-Sparse": tft_te[..., PT],
               "XGBoost-L1": xgb_rows(b.n_val_end, 2 * H, xgb_l1)}
    test_q = {m: interval_from_point(p, sigma[m]) for m, p in test_pt.items()}
    test_q["TFT-Sparse"] = (tft_te[..., 0], tft_te[..., 1], tft_te[..., 2])   # native P10/P50/P90

    # TFT-Sparse future forecast = the transformer's own output
    pf = np.clip(predict_neural(tft, fut_s, "tft").reshape(S, H, -1) * b.scale[:, None, None], 0, None)
    xgb_pred_fut = xgb_rows(T, H)
    fut_pt = {"Naive": naive_forecast(b.sales, T, H, b.season), "XGBoost": xgb_pred_fut,
              "LSTM": np.clip(predict_neural(lstm, fut_s, "lstm")[..., 0] * b.scale[:, None], 0, None),
              "TFT-Sparse": pf[..., PT]}
    fut_q = {m: interval_from_point(p, sigma[m]) for m, p in fut_pt.items()}
    fut_q["TFT-Sparse"] = (pf[..., 0], pf[..., PT], pf[..., 2])   # mean-aware head: mean drives the order quantity

    # ---------------------------------------------------------------- metrics
    Smat, labels, lvl, lvl_names = build_summing_matrix(b.meta)
    rev_w = (b.sales * b.price)[:, b.n_val_end - 28:b.n_val_end].sum(1)
    rows = []
    y_train_hist = b.sales[:, :b.n_val_end]
    for m in ALL_MODELS:
        lo, mid, hi = test_q[m]
        p = test_pt[m]
        qpred = np.stack([lo, mid, hi], -1)
        rows.append({"model": m, "MAE": mae(y_test, p), "RMSE": rmse(y_test, p), "WAPE": wape(y_test, p),
                     "Forecast accuracy (1-WAPE)": 1 - wape(y_test, p),
                     "MASE": mase(y_train_hist, y_test, p, cfg.season),
                     "Bias (sum pred/sum actual)": float(p.sum() / max(y_test.sum(), 1e-9)),
                     "WRMSSE": wrmsse(Smat, lvl, y_train_hist, y_test, p, rev_w),
                     "Pinball loss (P10/50/90)": pinball_np(y_test, qpred),
                     "P10-P90 coverage (nominal 0.80)": interval_coverage(y_test, lo, hi)})
    metrics = pd.DataFrame(rows)

    # validation-period metrics (use THESE for tuning; never tune on the test table).
    # Note: every model was early-stopped / calibrated on this period, so absolute values are optimistic.
    y_tr_val = b.sales[:, :b.n_train]
    vrows = []
    for m in ALL_MODELS:
        p = val_pt[m]
        r = {"model": m, "MAE": mae(y_val, p), "RMSE": rmse(y_val, p), "WAPE": wape(y_val, p),
             "MASE": mase(y_tr_val, y_val, p, cfg.season),
             "Bias (sum pred/sum actual)": float(p.sum() / max(y_val.sum(), 1e-9))}
        if m == "TFT-Sparse":        # baselines' intervals are fitted on val itself, so only TFT's native ones are reported
            r["Pinball loss (P10/50/90)"] = pinball_np(y_val, tft_va[..., :3])
            r["P10-P90 coverage (nominal 0.80)"] = interval_coverage(y_val, tft_va[..., 0], tft_va[..., 2])
        vrows.append(r)
    val_metrics = pd.DataFrame(vrows)

    # --------------------------------------------------- hierarchy (bottom-up)
    aq = aggregate_quantiles(Smat, *fut_q["TFT-Sparse"])
    agg_future = {"q10": aq[0], "q50": aq[1], "q90": aq[2]}
    agg_hist = Smat @ b.sales

    # ------------------------------------------ inventory schedule (future)
    params = InventoryParams.from_config(cfg)
    price = b.price[:, -28:].mean(1)
    rng = np.random.default_rng(cfg.seed)
    daily = b.sales[:, -28:].mean(1)
    on_hand = np.round(daily * (params.lead_time + params.review_period) * rng.uniform(0.5, 1.8, S), 0)
    on_order = np.zeros(S)
    schedule = compute_schedule(b.meta, *fut_q["TFT-Sparse"], on_hand, on_order, price, params)

    # --------------- policy back-test over the last H days with every model
    demand_bt = y_test[:, H:]
    init_oh = np.round(b.sales[:, T - H - 28:T - H].mean(1) * (params.lead_time + params.review_period + 1), 0)
    # mean-aware head: the inventory policy uses the mean forecast as its centre (P10/P90 still give the spread)
    inv_q = dict(test_q)
    inv_q["TFT-Sparse"] = (tft_te[..., 0], tft_te[..., PT], tft_te[..., 2])
    bt_q = {m: tuple(a[:, H:] for a in inv_q[m]) for m in ALL_MODELS}
    # Calibrated variant: rescale each model's forecast by ONE factor fitted on the validation period only
    # (sum actual / sum predicted, clipped to [0.5, 2]); test labels are never used.
    cal_factor = {m: float(np.clip(y_val.sum() / max(val_pt[m].sum(), 1e-9), 0.5, 2.0)) for m in ALL_MODELS}
    bt_q_cal = {m: tuple(a * cal_factor[m] for a in bt_q[m]) for m in ALL_MODELS}




    kpi_rows = []
    for m in ALL_MODELS:
        r = simulate_policy(demand_bt, *bt_q[m], price, init_oh, params)
        rc = simulate_policy(demand_bt, *bt_q_cal[m], price, init_oh, params)
        kpi_rows.append({"model": m, "Inventory cost": r["inventory_cost"], "Service level (in-stock)": r["service_level"],
                         "Fill rate": r["fill_rate"], "Revenue": r["revenue"],
                         "Calibration factor (val)": cal_factor[m],
                         "Inventory cost (cal)": rc["inventory_cost"], "Service level (cal)": rc["service_level"],
                         "Fill rate (cal)": rc["fill_rate"], "Revenue (cal)": rc["revenue"]})
    kpis = metrics.merge(pd.DataFrame(kpi_rows), on="model")

    # ---- split-conformal (CQR) intervals for the transformer, fitted on the validation block only.
    # "TFT-Sparse+CP": same point forecast, P10/P90 widened/shrunk so validation coverage hits 80 %.
    # Its "(cal)" columns first apply the scalar calibration factor and then conformalise on that calibrated forecast.
    scl = np.maximum(np.asarray(b.scale, dtype=float), 1e-6)
    tq = test_q["TFT-Sparse"]
    Q_raw = cqr_offset(y_val, tft_va[..., 0], tft_va[..., 2], scl)
    cf = cal_factor["TFT-Sparse"]
    Q_cal = cqr_offset(y_val, tft_va[..., 0] * cf, tft_va[..., 2] * cf, scl)
    cp_q = apply_offset(*tq, scl, Q_raw)
    cp_q_cal = apply_offset(*(a * cf for a in tq), scl, Q_cal)
    tq_inv = inv_q["TFT-Sparse"]
    cp_q_inv = apply_offset(*tq_inv, scl, Q_raw)
    cp_q_cal_inv = apply_offset(*(a * cf for a in tq_inv), scl, Q_cal)
    cp_pred = np.stack(cp_q, -1)
    base = kpis[kpis.model == "TFT-Sparse"].iloc[0].to_dict()
    cp = dict(base, model="TFT-Sparse+CP")
    cp["Pinball loss (P10/50/90)"] = pinball_np(y_test, cp_pred)
    cp["P10-P90 coverage (nominal 0.80)"] = interval_coverage(y_test, cp_q[0], cp_q[2])
    r0 = simulate_policy(demand_bt, *(a[:, H:] for a in cp_q_inv), price, init_oh, params)
    r1 = simulate_policy(demand_bt, *(a[:, H:] for a in cp_q_cal_inv), price, init_oh, params)
    cp.update({"Inventory cost": r0["inventory_cost"], "Service level (in-stock)": r0["service_level"],
               "Fill rate": r0["fill_rate"], "Revenue": r0["revenue"],
               "Inventory cost (cal)": r1["inventory_cost"], "Service level (cal)": r1["service_level"],
               "Fill rate (cal)": r1["fill_rate"], "Revenue (cal)": r1["revenue"]})
    metrics_cp = pd.DataFrame([cp])
    conformal_info = {"alpha": 0.2, "Q_raw": Q_raw, "Q_cal": Q_cal,
                      "val_coverage_before": interval_coverage(y_val, tft_va[..., 0], tft_va[..., 2]),
                      "val_coverage_after": interval_coverage(y_val, *(lambda t: (t[0], t[2]))(apply_offset(tft_va[..., 0], tft_va[..., 1], tft_va[..., 2], scl, Q_raw))),
                      "test_coverage_before": interval_coverage(y_test, tq[0], tq[2]),
                      "test_coverage_after": cp["P10-P90 coverage (nominal 0.80)"]}
    log("conformal: " + ", ".join(f"{k}={v:.4f}" for k, v in conformal_info.items()))


    # ---------------------------------------------------------- analytics
    out = {}
    try:
        out["shap_xgb"] = analytics.xgb_shap(xgbm, tab[(tab.t >= b.n_train) & (tab.t < T)])
    except Exception as e:                                               # pragma: no cover
        out["shap_xgb_error"] = repr(e)
    try:
        out["shap_tft"] = analytics.tft_group_shap(tft, tr_s, fut_s, device, max_samples=cfg.shap_samples)
    except Exception as e:                                               # pragma: no cover
        out["shap_tft_error"] = repr(e)
    anomalies = analytics.detect_anomalies(b)
    promo = analytics.promotion_analysis(b)

    results = {
        "cfg": cfg.to_dict(), "meta": b.meta, "series_ids": b.series_ids, "dates": np.asarray(b.dates),
        "future_dates": np.asarray(b.future_dates), "sales": b.sales, "price": price,
        "split": {"n_train": b.n_train, "n_val_end": b.n_val_end, "T": T, "H": H, "L": b.L},
        "metrics": kpis, "metrics_cp": metrics_cp, "conformal": conformal_info, "val_metrics": val_metrics, "history": {"TFT-Sparse": tft_hist, "LSTM": lstm_hist},
        "test": {"y": y_test, "dates": np.asarray(b.dates[b.n_val_end:]), "q": test_q, "point": test_pt},
        "future": {"q": fut_q, "point": fut_pt},
        "hierarchy": {"S": Smat, "labels": labels, "level": lvl, "level_names": lvl_names,
                      "agg_future": agg_future, "agg_hist": agg_hist},
        "inv": {"on_hand": on_hand, "on_order": on_order, "price": price, "schedule": schedule},
        "backtest": {"demand": demand_bt, "q": bt_q, "init_on_hand": init_oh,
                     "dates": np.asarray(b.dates[T - H:])},
        "anomalies": anomalies, "promo": promo, **out,
        "runtime_sec": time.time() - t0, "device": device,
    }
    # keep model weights next to the pickle
    results["_weights"] = {"tft": {k: v.cpu() for k, v in tft.state_dict().items()}}
    log(f"done in {results['runtime_sec']:.1f}s")
    return results


def save_results(res: dict, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(res, f)


def load_results(path):
    with open(path, "rb") as f:
        return pickle.load(f)
