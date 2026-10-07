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

Each model runs in isolation - one failure doesn't stop the rest - and
results append to <metrics>/benchmark.csv (re-running a model replaces its
rows), so an interrupted run resumes with just the missing models.

Two output locations:
  METRICS_DIR (default ./results/metrics) - small, keep these: benchmark.csv,
      train.py's own metric .txt per run (runs/<model>/), the generated
      config and the full train.py log per model. On Colab, point this at
      Google Drive.
  RESULTS_DIR (default ./results) - train.py's raw outputs: per-repeat
      predict/label/time .npy (~190 MB per repeat for this dataset's test
      set) and model.pth. Only needed to compute the scores above; on Colab
      leave it on the runtime's local disk.

Usage: python benchmark.py [--models AirLapseV2 GRU ...] [--epochs 50]
                           [--repeats 3] [--early_stop 10]
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
    "AirLapseV2",                                         # proposed
    "LSTM", "Transformer", "Informer",                    # recurrent / attention
    "PM25_GNN", "AirFormer", "AirPhyNet", "AirDualODE", "AirDDE",   # air-quality graph / physics
    "TCN_DIR", "STMamba",                                 # 2025-26 PM2.5 models (re-implemented)
]


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


def train(model, args, base_cfg):
    cfg = yaml.safe_load(yaml.safe_dump(base_cfg))
    cfg["experiments"].update(model=model, save_npy=True)
    cfg["train"].update(epochs=args.epochs, exp_repeat=args.repeats, early_stop=args.early_stop)
    cfg_fp = os.path.join(METRICS, "configs", f"config_{model}.yaml")
    os.makedirs(os.path.dirname(cfg_fp), exist_ok=True)
    yaml.safe_dump(cfg, open(cfg_fp, "w"), sort_keys=False)
    env = dict(os.environ, CONFIG_FP=cfg_fp, RESULTS_DIR=RESULTS)
    r = subprocess.run([sys.executable, "train.py"], cwd=REPO, env=env, capture_output=True, text=True)
    log_fp = os.path.join(METRICS, "logs", f"{model}.log")
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
    return cfg, run_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--early_stop", type=int, default=10)
    args = ap.parse_args()

    os.makedirs(METRICS, exist_ok=True)
    print(f"metrics -> {METRICS}")
    print(f"raw outputs -> {RESULTS}")
    base = yaml.safe_load(open(os.path.join(REPO, "config.yaml"), encoding="utf-8"))
    obs, t0 = observed_mask(base)
    out = os.path.join(METRICS, "benchmark.csv")
    rows = pd.read_csv(out).to_dict("records") if os.path.exists(out) else []
    rows = [r for r in rows if r["model"] not in args.models]

    for m in args.models:
        t_start = time.time()
        print(f"== {m}", flush=True)
        try:
            cfg, run_dir = train(m, args, base)
        except Exception as e:
            print(f"   FAILED: {e}", flush=True)
            rows.append(dict(model=m, status="failed"))
            pd.DataFrame(rows).to_csv(out, index=False)
            continue
        H = cfg["train"]["hist_len"]
        for rep in sorted(glob.glob(os.path.join(run_dir, "[0-9][0-9]"))):
            pf = np.load(os.path.join(rep, "predict.npy"))
            lf = np.load(os.path.join(rep, "label.npy"))
            mask = window_mask(obs, t0, np.load(os.path.join(rep, "time.npy")), H)
            rows.append(dict(model=m, status="ok", repeat=int(os.path.basename(rep)),
                             minutes=round((time.time() - t_start) / 60, 1),
                             **scores(pf[:, H:], lf[:, H:], mask)))
            if not any(r["model"] == "Persistence" for r in rows):
                persist = np.repeat(lf[:, H - 1:H], lf.shape[1] - H, axis=1)
                rows.append(dict(model="Persistence", status="ok", repeat=0,
                                 **scores(persist, lf[:, H:], mask)))
        print(f"   done in {(time.time() - t_start) / 60:.1f} min", flush=True)
        pd.DataFrame(rows).to_csv(out, index=False)

    df = pd.DataFrame(rows)
    ok = df[df.status == "ok"]
    cols = ["RMSE", "MAE", "MAPE", "RMSE_gt35.5", "CSI@35.5", "POD@35.5", "FAR@35.5", "CSI@55.5"]
    pd.set_option("display.width", 200)
    print(ok.groupby("model")[cols].agg(["mean", "std"]).round(3)
            .sort_values(("RMSE", "mean")).to_string())
    failed = df[df.status == "failed"].model.tolist()
    if failed:
        print("failed:", failed)


if __name__ == "__main__":
    main()
