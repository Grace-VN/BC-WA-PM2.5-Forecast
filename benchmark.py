"""Benchmark several models on the BC + Washington PM2.5 dataset.

Drives train.py unchanged, one model at a time, via a generated config
(CONFIG_FP - config.yaml itself is never rewritten), then reloads each
repeat's saved test predictions and scores ONLY the forecast hours.
train.py's own saved predict/label arrays span hist_len + pred_len, with the
history hours copied verbatim into predict, so its built-in RMSE/CSI are
flattered by those hours - use this script's numbers.

Scores per model: RMSE / MAE / MAPE overall (MAPE in %, skipping true values
below MAPE_MIN ug/m3 exactly like train.py's get_mape - near-zero PM2.5 makes
percentages meaningless), RMSE on hours with PM2.5 > 35.5 ug/m3,
and CSI / POD / FAR at the US AQI 35.5 ("unhealthy for sensitive groups")
and 55.5 ("unhealthy") ug/m3 thresholds, plus a persistence baseline (last
observed hour repeated across the horizon) on the same test windows.
Only REAL measurements are scored: the dataset's <data>_observed.npy marks
which (hour, station) values were measured rather than gap-filled, and each
saved prediction is matched to its hour through train.py's time.npy.

Each model is trained --repeats times (default 5); repeat k is its own
train.py process with random seed k, and its scores go to their own file,
<metrics>/scores/<model>__rep<k>.csv, as soon as it finishes. Re-running
skips every (model, repeat) that already has a file, so an interrupted run
resumes at the repeat it stopped in; --rerun deletes a model's earlier
files first. One model failing doesn't stop the rest.

Because every run has its own file, the benchmark can be split into groups
(--group 1..4, or any --models list) run in separate sessions - one after
another or at the same time (e.g. several Colab runtimes writing to the same
Drive folder) - as long as two sessions don't train the same model at once.
After each run, all score files are merged into <metrics>/benchmark.csv and
summarised - mean and sample std (n-1) over the repeats, printed as
"mean ± std" and saved as <metrics>/benchmark_summary.csv. --summary only
merges and summarises, without training.

Two output locations:
  METRICS_DIR (default ./results/metrics) - small, keep these: benchmark.csv,
      benchmark_summary.csv, train.py's own metric .txt per run
      (runs/<model>/), the generated config and the full train.py log per
      model and repeat. On Colab, point this at
      Google Drive.
  RESULTS_DIR (default ./results) - train.py's raw outputs: per-repeat
      predict/label/time .npy (~190 MB per repeat for this dataset's test
      set) and model.pth. Only needed to compute the scores above; on Colab
      leave it on the runtime's local disk.

Usage: python benchmark.py [--group 1|2|3|4 | --models AirLapseV2 GRU ...]
                           [--epochs 50] [--repeats 5] [--early_stop 10]
                           [--rerun] [--keep_raw] [--summary]
"""
import argparse
import glob
import os
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import yaml

REPO = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.environ.get("RESULTS_DIR", os.path.join(REPO, "results"))
METRICS = os.environ.get("METRICS_DIR", os.path.join(RESULTS, "metrics"))
THRESHOLDS = (35.5, 55.5)
MAPE_MIN = 1.0      # ug/m3, same mask as train.py's get_mape
DEFAULT_MODELS = [
    "AirLapseV2",                                                    # proposed
    "MLP", "LSTM", "GRU",                                            # neural networks
    "Transformer", "Informer", "Autoformer", "Crossformer",          # transformers
    "PM25_GNN", "AirFormer", "AirPhyNet", "AirDualODE", "AirDDE",   # air-quality graph / physics
    "TCN_DIR", "STMamba",                                            # 2025-26 PM2.5 models (re-implemented)
]
# Four groups of roughly equal training time (from measured per-batch cost),
# for running the benchmark in separate sessions.
GROUPS = {
    1: ["MLP", "GRU", "LSTM", "PM25_GNN", "AirLapseV2", "Crossformer",
        "AirPhyNet", "Transformer", "AirDualODE"],
    2: ["STMamba", "Informer", "AirFormer"],
    3: ["Autoformer", "TCN_DIR"],
    4: ["AirDDE"],
}
assert sorted(sum(GROUPS.values(), [])) == sorted(DEFAULT_MODELS)


def observed_mask(cfg):
    """bool [hours, stations] of real measurements, and the dataset's first hour (epoch s)."""
    ds = cfg["dataset"][cfg["experiments"]["dataset_num"]]
    fp = os.path.join(REPO, ds["data_fp"].replace(".npy", "_observed.npy"))
    if not os.path.exists(fp):
        print(f"WARNING: {fp} not found - scoring gap-filled values too")
        return None, None
    y, mo, d, h, mi = ds["data_start"][0]
    return np.load(fp), pd.Timestamp(year=y, month=mo, day=d, hour=h, minute=mi, tz="UTC").timestamp()


def window_mask(obs, t0, times, H):
    """Observed mask for the forecast part of each saved test window: [samples, pred_len, stations]."""
    if obs is None:
        return None
    idx = np.rint((times[:, H:] - t0) / 3600).astype(int)
    return obs[idx]


def scores(pred, label, mask=None):
    """pred/label: [samples, pred_len, stations, 1] in ug/m3; mask: which to score."""
    p, y = pred[..., 0], label[..., 0]
    if mask is not None:
        p, y = p[mask], y[mask]
    out = dict(RMSE=float(np.sqrt(np.mean((p - y) ** 2))), MAE=float(np.mean(np.abs(p - y))))
    m = np.abs(y) >= MAPE_MIN
    out["MAPE"] = float(np.mean(np.abs((p[m] - y[m]) / y[m])) * 100) if m.any() else np.nan
    high = y > THRESHOLDS[0]
    out["RMSE_gt35.5"] = float(np.sqrt(np.mean((p[high] - y[high]) ** 2))) if high.any() else np.nan
    for thr in THRESHOLDS:
        ph, yh = p >= thr, y >= thr
        hit, miss, fa = (ph & yh).sum(), (~ph & yh).sum(), (ph & ~yh).sum()
        out[f"CSI@{thr}"] = hit / max(hit + miss + fa, 1)
        out[f"POD@{thr}"] = hit / max(hit + miss, 1)
        out[f"FAR@{thr}"] = fa / max(hit + fa, 1)
    return out


def train(model, repeat, args, base_cfg):
    """One repeat (seed = repeat) of one model; returns the run's repeat directory."""
    cfg = yaml.safe_load(yaml.safe_dump(base_cfg))
    cfg["experiments"].update(model=model, save_npy=True)
    cfg["train"].update(epochs=args.epochs, exp_repeat=1, early_stop=args.early_stop)
    cfg_fp = os.path.join(METRICS, "configs", f"config_{model}.yaml")
    os.makedirs(os.path.dirname(cfg_fp), exist_ok=True)
    yaml.safe_dump(cfg, open(cfg_fp, "w"), sort_keys=False)
    env = dict(os.environ, CONFIG_FP=cfg_fp, RESULTS_DIR=RESULTS, SEED=str(repeat))
    r = subprocess.run([sys.executable, "train.py"], cwd=REPO, env=env, capture_output=True, text=True)
    log_fp = os.path.join(METRICS, "logs", f"{model}_rep{repeat}.log")
    os.makedirs(os.path.dirname(log_fp), exist_ok=True)
    open(log_fp, "w", encoding="utf-8").write(r.stdout + "\n" + r.stderr)
    if r.returncode != 0:
        print(r.stderr[-3000:])
        raise RuntimeError(f"{model} failed (full log: {log_fp})")
    metric_fp = r.stdout.strip().splitlines()[-1]   # train.py prints its metric file last
    run_dir = os.path.dirname(metric_fp)
    keep = os.path.join(METRICS, "runs", model, os.path.basename(run_dir))
    os.makedirs(keep, exist_ok=True)
    shutil.copy2(metric_fp, keep)
    return cfg, os.path.join(run_dir, "00")


SUMMARY_COLS = ["RMSE", "MAE", "MAPE", "RMSE_gt35.5", "CSI@35.5", "POD@35.5", "FAR@35.5",
                "CSI@55.5", "POD@55.5", "FAR@55.5"]


def summarise(rows, out_fp):
    """Mean and sample std (n-1) over repeats per model; saves the numbers,
    prints 'mean ± std'."""
    df = pd.DataFrame(rows)
    if df.empty or "status" not in df:
        print("no results yet")
        return
    ok = df[df.status == "ok"]
    if ok.empty:
        print("no successful runs yet; failed:", sorted(set(df.model)))
        return
    g = ok.groupby("model")[SUMMARY_COLS]
    summary = pd.concat([g.size().rename("n_repeats"), g.mean().add_suffix("_mean"),
                         g.std(ddof=1).add_suffix("_std")], axis=1).sort_values("RMSE_mean")
    summary.to_csv(out_fp)
    shown = pd.DataFrame({"n": summary.n_repeats}, index=summary.index)
    pm = "±" if (sys.stdout.encoding or "").lower().replace("-", "") == "utf8" else "+/-"
    for c in SUMMARY_COLS:
        d = 1 if c == "MAPE" else 3
        shown[c] = [f"{m:.{d}f} {pm} {sd:.{d}f}" if pd.notna(sd) else f"{m:.{d}f}"
                    for m, sd in zip(summary[c + "_mean"], summary[c + "_std"])]
    pd.set_option("display.width", 250)
    print(shown.to_string())
    failed = sorted(set(df[df.status == "failed"].model))
    if failed:
        print("failed:", failed)


def score_fp(model, k):
    return os.path.join(METRICS, "scores", f"{model}__rep{k}.csv")


def merge_scores():
    """All per-run score files -> one table, also written to benchmark.csv."""
    files = sorted(glob.glob(os.path.join(METRICS, "scores", "*.csv")))
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True) if files else pd.DataFrame()
    if len(df):
        tmp = os.path.join(METRICS, f".benchmark.{os.getpid()}.csv")
        df.sort_values(["model", "repeat"]).to_csv(tmp, index=False)
        os.replace(tmp, os.path.join(METRICS, "benchmark.csv"))
    return df


def migrate_old_csv():
    """Results from before per-run files: split benchmark.csv into score files once."""
    old = os.path.join(METRICS, "benchmark.csv")
    if os.path.exists(old) and not os.path.isdir(os.path.join(METRICS, "scores")):
        os.makedirs(os.path.join(METRICS, "scores"))
        for r in pd.read_csv(old).to_dict("records"):
            if r.get("status") == "ok":
                pd.DataFrame([r]).to_csv(score_fp(r["model"], int(r["repeat"])), index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--group", type=int, choices=sorted(GROUPS), help="run one predefined model group")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--early_stop", type=int, default=10)
    ap.add_argument("--rerun", action="store_true", help="delete these models' earlier results first")
    ap.add_argument("--keep_raw", action="store_true",
                    help="keep each repeat's predict/label arrays and model.pth in RESULTS_DIR")
    ap.add_argument("--summary", action="store_true", help="only merge and summarise existing results")
    args = ap.parse_args()
    models = args.models or (GROUPS[args.group] if args.group else DEFAULT_MODELS)

    os.makedirs(METRICS, exist_ok=True)
    migrate_old_csv()
    os.makedirs(os.path.join(METRICS, "scores"), exist_ok=True)
    print(f"metrics -> {METRICS}")
    if args.summary:
        summarise(merge_scores().to_dict("records"), os.path.join(METRICS, "benchmark_summary.csv"))
        return
    print(f"raw outputs -> {RESULTS}")
    print(f"models: {' '.join(models)}")
    base = yaml.safe_load(open(os.path.join(REPO, "config.yaml"), encoding="utf-8"))
    obs, t0 = observed_mask(base)
    if args.rerun:
        for m in models:
            for f in glob.glob(os.path.join(METRICS, "scores", f"{m}__rep*.csv")):
                os.remove(f)
    failed = []

    for m in models:
        for k in range(args.repeats):
            if os.path.exists(score_fp(m, k)):
                continue
            t_start = time.time()
            print(f"== {m} repeat {k + 1}/{args.repeats} (seed {k})", flush=True)
            try:
                cfg, rep = train(m, k, args, base)
            except Exception as e:
                print(f"   FAILED: {e}", flush=True)
                failed.append(m)
                break                                   # later repeats would fail the same way
            H = cfg["train"]["hist_len"]
            pf = np.load(os.path.join(rep, "predict.npy"))
            lf = np.load(os.path.join(rep, "label.npy"))
            mask = window_mask(obs, t0, np.load(os.path.join(rep, "time.npy")), H)
            row = dict(model=m, status="ok", repeat=k, minutes=round((time.time() - t_start) / 60, 1),
                       **scores(pf[:, H:], lf[:, H:], mask))
            pd.DataFrame([row]).to_csv(score_fp(m, k), index=False)
            if not os.path.exists(score_fp("Persistence", 0)):
                persist = np.repeat(lf[:, H - 1:H], lf.shape[1] - H, axis=1)
                pd.DataFrame([dict(model="Persistence", status="ok", repeat=0,
                                   **scores(persist, lf[:, H:], mask))]).to_csv(
                    score_fp("Persistence", 0), index=False)
            if not args.keep_raw:                       # ~190 MB of arrays per repeat, scored already
                shutil.rmtree(os.path.dirname(rep), ignore_errors=True)
            merge_scores()
            print(f"   done in {(time.time() - t_start) / 60:.1f} min", flush=True)

    rows = merge_scores().to_dict("records") + [dict(model=m, status="failed") for m in failed]
    summarise(rows, os.path.join(METRICS, "benchmark_summary.csv"))


if __name__ == "__main__":
    main()
