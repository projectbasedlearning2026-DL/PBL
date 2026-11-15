"""train.py - CLI: run the full pipeline and save artifacts for the dashboard.

    python train.py                      # dummy data, quick demo
    python train.py --source m5 --data_dir ./m5 --max_series 300 --epochs 15
"""
import argparse
from pathlib import Path

import torch

from config import Config
from pipeline import run_pipeline, save_results


def main():
    ap = argparse.ArgumentParser()
    d = Config()
    ap.add_argument("--source", default=d.source, choices=["dummy", "m5", "walmart", "favorita"])
    ap.add_argument("--data_dir", default=d.data_dir)
    ap.add_argument("--max_series", type=int, default=d.max_series)
    ap.add_argument("--epochs", type=int, default=d.epochs)
    ap.add_argument("--horizon", type=int, default=d.horizon)
    ap.add_argument("--encoder_len", type=int, default=d.encoder_len)
    ap.add_argument("--freq", default=d.freq)
    ap.add_argument("--season", type=int, default=d.season)
    ap.add_argument("--lambda_mech", type=float, default=d.lambda_mech)
    ap.add_argument("--topk_ratio", type=float, default=d.topk_ratio)
    ap.add_argument("--out", default=d.artifacts_dir)
    ap.add_argument("--seed", type=int, default=d.seed)
    ap.add_argument("--lr", type=float, default=d.lr)
    ap.add_argument("--train_stride", type=int, default=d.train_stride, help="spacing between training windows; raise (e.g. 4) to speed up large runs")
    ap.add_argument("--loss_mode", default=d.loss_mode, choices=["hybrid", "pinball"])
    ap.add_argument("--head", default=d.head, choices=["quantile", "mean"])
    ap.add_argument("--mean_weight", type=float, default=d.mean_weight)
    ap.add_argument("--force_select_horizon", type=int, default=int(d.force_select_horizon), choices=[0, 1])
    a = ap.parse_args()
    cfg = Config(source=a.source, data_dir=a.data_dir, max_series=a.max_series, epochs=a.epochs, horizon=a.horizon,
                 encoder_len=a.encoder_len, freq=a.freq, season=a.season, lambda_mech=a.lambda_mech,
                 topk_ratio=a.topk_ratio, artifacts_dir=a.out, seed=a.seed, lr=a.lr, train_stride=a.train_stride, loss_mode=a.loss_mode, head=a.head, mean_weight=a.mean_weight,
                 force_select_horizon=bool(a.force_select_horizon))
    res = run_pipeline(cfg)
    out = Path(a.out)
    save_results(res, out / "results.pkl")
    torch.save(res["_weights"]["tft"], out / "tft_sparse.pt")
    res["metrics"].to_csv(out / "metrics.csv", index=False)
    res["val_metrics"].to_csv(out / "val_metrics.csv", index=False)
    res["metrics_cp"].to_csv(out / "metrics_cp.csv", index=False)
    import json, pandas as pd
    (out / "config.json").write_text(json.dumps({**cfg.to_dict(), "device": res["device"],
                                                 "runtime_sec": res["runtime_sec"], "torch": torch.__version__}, indent=2))
    for k, h in res["history"].items():
        pd.DataFrame(h).to_csv(out / f"history_{k}.csv", index=False)
    res["inv"]["schedule"].to_csv(out / "optimization_schedule.csv", index=False)
    print("\nVALIDATION metrics (use for tuning):\n", res["val_metrics"].round(4).to_string(index=False))
    print("\nTEST metrics:\n", res["metrics"].round(4).to_string(index=False))
    print("\nCONFORMAL (TFT-Sparse+CP, test):\n", res["metrics_cp"][["model", "P10-P90 coverage (nominal 0.80)", "Pinball loss (P10/50/90)", "Inventory cost", "Inventory cost (cal)", "Service level (cal)"]].round(4).to_string(index=False))
    print("conformal info:", {k: round(v, 4) for k, v in res["conformal"].items()})
    print(f"\nArtifacts saved in {out.resolve()}")


if __name__ == "__main__":
    main()
