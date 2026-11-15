"""aggregate.py - build the paper's tables and figures from saved run folders (no retraining).

    python aggregate.py --roots runs                    # after run_all.ps1
    python aggregate.py --roots runs artifacts          # also include older single runs that have config.json
    python aggregate.py --roots runs --out report

A run folder must contain config.json and metrics.csv (written by the updated train.py); val_metrics.csv is used if present.
Runs are grouped by (dataset, number of series, variant); a variant is derived from the config, so runs of
different seeds with the same settings are averaged together. Everything is TEST-period unless stated.

Writes into --out:
  results_long.csv      every run x model x metric
  summary_test.csv      mean / std / n_seeds per dataset, variant, model, metric
  paired_vs_baselines.csv  TFT-Sparse minus each baseline, per matching seed (mean diff, std, wins)
  ablation.csv          TFT-Sparse per variant, with paired difference to the default variant
  tables.md             readable tables
  fig_main_<dataset>.png, fig_ablation_<dataset>.png
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEFAULTS = dict(topk_ratio=0.5, lambda_mech=0.02, loss_mode="hybrid", force_select_horizon=True, lr=1e-3, train_stride=2, head="quantile")
LOWER_BETTER = {"MAE", "RMSE", "WAPE", "MASE", "WRMSSE", "Pinball loss (P10/50/90)", "Inventory cost",
                "Inventory cost (cal)"}
KEY_METRICS = ["MAE", "RMSE", "WAPE", "MASE", "Bias (sum pred/sum actual)", "WRMSSE", "Pinball loss (P10/50/90)",
               "P10-P90 coverage (nominal 0.80)", "Inventory cost", "Service level (in-stock)", "Fill rate",
               "Inventory cost (cal)", "Service level (cal)", "Fill rate (cal)"]
MODEL_ORDER = ["Naive", "XGBoost", "XGBoost-L1", "LSTM", "TFT-Sparse", "TFT-Sparse+CP"]
TFT = "TFT-Sparse"


DEFAULT_EPOCHS = {"m5": 8, "favorita": 8, "walmart": 30}      # the settings used for the default runs


def variant_name(cfg):
    diffs = []
    ep = cfg.get("epochs")
    if ep is not None and cfg.get("source") in DEFAULT_EPOCHS and ep != DEFAULT_EPOCHS[cfg["source"]]:
        diffs.append(f"epochs={ep}")
    for k, dv in DEFAULTS.items():
        v = cfg.get(k, dv)
        if (isinstance(dv, float) and abs(float(v) - dv) > 1e-12) or (not isinstance(dv, float) and v != dv):
            short = {"topk_ratio": "topk", "lambda_mech": "lmech", "loss_mode": "loss",
                     "force_select_horizon": "force", "lr": "lr", "train_stride": "stride", "head": "head"}[k]
            diffs.append(f"{short}={int(v) if isinstance(v, bool) else v}")
    return "default" if not diffs else ",".join(diffs)


def load_runs(roots):
    rows, skipped = [], []
    for root in roots:
        for cfg_path in sorted(Path(root).glob("*/config.json")):
            d = cfg_path.parent
            if not (d / "metrics.csv").exists():
                skipped.append(str(d)); continue
            cfg = json.loads(cfg_path.read_text())
            meta = dict(run=d.name, source=cfg["source"], n_series=cfg["max_series"], seed=cfg["seed"],
                        variant=variant_name(cfg), device=cfg.get("device", ""), runtime_sec=cfg.get("runtime_sec"))
            for split, fn in (("test", "metrics.csv"), ("test", "metrics_cp.csv"), ("val", "val_metrics.csv")):
                f = d / fn
                if not f.exists():
                    continue
                df = pd.read_csv(f)
                long = df.melt(id_vars="model", var_name="metric", value_name="value").dropna(subset=["value"])
                for k, v in meta.items():
                    long[k] = v
                long["split"] = split
                rows.append(long)
    if not rows:
        raise SystemExit("No run folders with config.json + metrics.csv found under: " + ", ".join(map(str, roots)))
    return pd.concat(rows, ignore_index=True), skipped


def mean_std(x):
    x = pd.Series(x).dropna()
    return pd.Series({"mean": x.mean(), "std": x.std(ddof=1) if len(x) > 1 else np.nan, "n": len(x)})


def fmt(m, s, n, digits=3):
    if pd.isna(m):
        return ""
    return f"{m:.{digits}f}" if (n < 2 or pd.isna(s)) else f"{m:.{digits}f} ± {s:.{digits}f}"


def md_table(df):
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", default=["runs"])
    ap.add_argument("--out", default="report")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    long, skipped = load_runs(a.roots)
    long.to_csv(out / "results_long.csv", index=False)
    test = long[long.split == "test"]
    group = ["source", "n_series", "variant"]

    # ---- summary (mean +- std across seeds)
    summ = (test.groupby(group + ["model", "metric"])["value"].apply(mean_std).unstack().reset_index())
    summ.to_csv(out / "summary_test.csv", index=False)

    # ---- paired comparison TFT vs every other model, same run (= same seed and same series sample)
    wide = test.pivot_table(index=group + ["seed"], columns=["metric", "model"], values="value")
    pairs = []
    for metric in [m for m in KEY_METRICS if m in wide.columns.get_level_values(0)]:
        sub = wide[metric]
        for me in [TFT, TFT + "+CP"]:
            if me not in sub.columns:
                continue
            if me.endswith("+CP") and metric in ("MAE", "RMSE", "WAPE", "MASE", "WRMSSE", "Bias (sum pred/sum actual)"):
                continue          # same point forecast as TFT-Sparse, so identical by construction
            # TFT-Sparse is compared with the baselines; TFT-Sparse+CP with the baselines and with plain TFT-Sparse
            others = [c for c in sub.columns if c != me and (c != TFT + "+CP")]
            for other in others:
                diff = (sub[me] - sub[other]).dropna()
                if diff.empty:
                    continue
                for keys, dd in diff.groupby(level=group):
                    better = (dd < 0) if metric in LOWER_BETTER else (dd > 0)
                    pairs.append(dict(zip(group, keys), model=me, metric=metric, vs=other, n_seeds=len(dd),
                                      mean_diff=dd.mean(), std_diff=dd.std(ddof=1) if len(dd) > 1 else np.nan,
                                      tft_better_in=f"{int(better.sum())}/{len(dd)}" if metric in LOWER_BETTER else "",
                                      lower_is_better=metric in LOWER_BETTER))
    paired = pd.DataFrame(pairs)
    paired.to_csv(out / "paired_vs_baselines.csv", index=False)

    # ---- ablation: TFT variants, paired against the default variant of the same seed
    abl_rows = []
    for (src, n), g in test[test.model == TFT].groupby(["source", "n_series"]):
        w = g.pivot_table(index="seed", columns=["variant", "metric"], values="value")
        if "default" not in w.columns.get_level_values(0):
            continue
        for var in w.columns.get_level_values(0).unique():
            for metric in [m for m in ["MAE", "RMSE", "WAPE", "Pinball loss (P10/50/90)", "Inventory cost",
                                       "Inventory cost (cal)", "Bias (sum pred/sum actual)"]
                           if (var, m) in w.columns]:
                x = w[(var, metric)].dropna()
                d = (w[(var, metric)] - w[("default", metric)]).dropna() if var != "default" else pd.Series(dtype=float)
                abl_rows.append(dict(source=src, n_series=n, variant=var, metric=metric, n_seeds=len(x),
                                     mean=x.mean(), std=x.std(ddof=1) if len(x) > 1 else np.nan,
                                     diff_vs_default=d.mean() if len(d) else np.nan,
                                     diff_std=d.std(ddof=1) if len(d) > 1 else np.nan, n_paired=len(d)))
    abl = pd.DataFrame(abl_rows)
    abl.to_csv(out / "ablation.csv", index=False)

    # ---- markdown
    md = ["# Results report (generated by aggregate.py)\n",
          f"Runs read: {long.run.nunique()} | datasets: {sorted(long.source.unique())} | skipped folders without metrics.csv: {len(skipped)}\n",
          "All numbers are TEST-period, mean ± std over seeds (each seed = different random series sample AND model start). "
          "With a single seed only the value is shown.\n"]
    for (src, n, var), g in summ.groupby(group):
        piv = g.pivot(index="model", columns="metric", values=["mean", "std", "n"])
        cols = [m for m in KEY_METRICS if m in piv["mean"].columns]
        tbl = pd.DataFrame(index=[m for m in MODEL_ORDER if m in piv.index], columns=cols, dtype=object)
        for mod in tbl.index:
            for c in cols:
                tbl.loc[mod, c] = fmt(piv["mean"].loc[mod, c], piv["std"].loc[mod, c], piv["n"].loc[mod, c])
        tbl.insert(0, "model", tbl.index)
        nseeds = int(g["n"].max())
        md += [f"\n## {src}, {n} series, variant `{var}` ({nseeds} seed{'s' if nseeds != 1 else ''})\n", md_table(tbl.reset_index(drop=True))]
    if not paired.empty:
        md += ["\n## Paired differences: TFT-Sparse minus baseline (same seed/series sample)\n",
               "Negative is better for error metrics; `tft_better_in` counts seeds in which TFT-Sparse is better.\n"]
        p = paired[paired.metric.isin(["MAE", "WAPE", "RMSE", "WRMSSE", "Pinball loss (P10/50/90)", "Inventory cost",
                                       "Inventory cost (cal)"])].copy()
        p["mean_diff"] = p["mean_diff"].map(lambda v: f"{v:.4f}")
        p["std_diff"] = p["std_diff"].map(lambda v: "" if pd.isna(v) else f"{v:.4f}")
        md.append(md_table(p[group + ["model", "metric", "vs", "n_seeds", "mean_diff", "std_diff", "tft_better_in"]]))
    if not abl.empty:
        md += ["\n## Ablation (TFT-Sparse variants, paired with the default variant)\n"]
        t = abl.copy()
        t["mean ± std"] = [fmt(m, s, n) for m, s, n in zip(t["mean"], t["std"], t["n_seeds"])]
        t["diff vs default"] = [fmt(m, s, n, 4) if n > 0 else "" for m, s, n in zip(t["diff_vs_default"], t["diff_std"], t["n_paired"])]
        md.append(md_table(t[["source", "n_series", "variant", "metric", "n_seeds", "mean ± std", "diff vs default"]]))
    (out / "tables.md").write_text("\n".join(md), encoding="utf-8")

    # ---- figures
    colors = {"Naive": "#9aa0a6", "XGBoost": "#f29900", "XGBoost-L1": "#d93025", "LSTM": "#a142f4", TFT: "#1a73e8", TFT + "+CP": "#0b8043"}
    for (src, n), g in summ[summ.variant == "default"].groupby(["source", "n_series"]):
        metrics = [m for m in ["MAE", "WAPE", "RMSE", "Inventory cost"] if m in set(g.metric)]
        fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 3.6), squeeze=False)
        for ax, metric in zip(axes[0], metrics):
            gm = g[g.metric == metric].set_index("model").reindex([m for m in MODEL_ORDER if m in set(g.model)])
            ax.bar(gm.index, gm["mean"], yerr=gm["std"].fillna(0), color=[colors[m] for m in gm.index], capsize=3)
            ax.set_title(f"{metric} (lower is better)", fontsize=10)
            ax.tick_params(axis="x", rotation=40)
        fig.suptitle(f"{src}, {n} series, default settings (mean ± std over seeds)", fontsize=11)
        fig.tight_layout(); fig.savefig(out / f"fig_main_{src}_n{n}.png", dpi=160); plt.close(fig)
    if not abl.empty:
        for (src, n), g in abl[abl.metric == "MAE"].groupby(["source", "n_series"]):
            if g.variant.nunique() < 2:
                continue
            fig, ax = plt.subplots(figsize=(1.2 * len(g) + 3, 3.6))
            ax.bar(g.variant, g["mean"], yerr=g["std"].fillna(0), color="#1a73e8", capsize=3)
            ax.set_title(f"TFT-Sparse test MAE by variant, {src} {n} series (± std over seeds)", fontsize=10)
            ax.tick_params(axis="x", rotation=40)
            fig.tight_layout(); fig.savefig(out / f"fig_ablation_{src}_n{n}.png", dpi=160); plt.close(fig)

    # ---- scaling figure: how each metric changes with the number of series (one figure per dataset)
    SERIES_STYLE = [("TFT-Sparse", "#2a78d6", "o"), ("XGBoost", "#eb6834", "s"), ("XGBoost-L1", "#1baf7a", "^"),
                    ("LSTM", "#eda100", "D"), ("Naive", "#e87ba4", "v")]       # fixed colour per model, marker as 2nd cue
    PANELS = [("MAE", "MAE", 1), ("WRMSSE", "WRMSSE (subset)", 1), ("Pinball loss (P10/50/90)", "Pinball loss", 1),
              ("Inventory cost (cal)", "Calibrated inventory cost per series", None)]
    for src, gs in summ.groupby("source"):
        pick = gs.groupby("variant").n_series.nunique()
        if pick.max() < 2:
            continue                                     # only one size for this dataset: nothing to scale
        var = "default" if pick.get("default", 0) == pick.max() else pick.idxmax()
        gv = gs[gs.variant == var]
        fig, axes = plt.subplots(1, len(PANELS), figsize=(4.2 * len(PANELS), 3.8))
        for ax, (metric, title, _) in zip(axes, PANELS):
            for model, col, mk in SERIES_STYLE:
                d = gv[(gv.model == model) & (gv.metric == metric)].sort_values("n_series")
                if d.empty:
                    continue
                div = d.n_series.to_numpy() if metric == "Inventory cost (cal)" else 1.0
                y, s = d["mean"].to_numpy() / div, d["std"].fillna(0).to_numpy() / div
                lw, z = (2.4, 3) if model == "TFT-Sparse" else (1.6, 2)
                ax.plot(d.n_series, y, color=col, marker=mk, ms=6, lw=lw, label=model, zorder=z)
                if model == "TFT-Sparse":
                    ax.fill_between(d.n_series, y - s, y + s, color=col, alpha=0.15, lw=0)
            ax.set_xscale("log")
            xt = sorted(gv.n_series.unique())
            ax.set_xticks(xt); ax.set_xticklabels([str(x) for x in xt])
            ax.minorticks_off()
            ax.set_xlabel("number of series"); ax.set_title(title, fontsize=10)
            ax.grid(alpha=0.25, lw=0.6)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
        h, l = axes[0].get_legend_handles_labels()
        fig.legend(h, l, loc="lower center", ncol=len(l), frameon=False, fontsize=9)
        fig.suptitle(f"{src}: test metrics vs number of series (variant: {var}; lower is better; band = TFT ± std over seeds)",
                     fontsize=10)
        fig.tight_layout(rect=(0, 0.07, 1, 1)); fig.savefig(out / f"fig_scaling_{src}.png", dpi=160); plt.close(fig)

    print(f"Report written to {out.resolve()}")
    print(f"  runs={long.run.nunique()}  summary rows={len(summ)}  paired rows={len(paired)}  ablation rows={len(abl)}")
    if skipped:
        print("  skipped (no metrics.csv):", ", ".join(skipped))


if __name__ == "__main__":
    main()
