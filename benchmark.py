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
(--group 1..6, or any --models list) run in separate sessions - one after
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

Usage: python benchmark.py [--group 1..6 | --models AirLapseV2 GRU ...]
                           [--epochs 50] [--repeats 5] [--early_stop 10]
                           [--rerun] [--keep_raw] [--summary]
                           [--max_hours H] [--deadline_hours H] [--gpu I] [--repeat_ids K ...]
                           [--overrides tuned.yaml --label NAME]
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
# Model groups for running the benchmark in separate sessions. AirPhyNet
# (~3.3 h per run on a Kaggle T4 - its ODE solver gains little from a GPU)
# and AirDDE (also ODE-based, expected to be slower still) each get their own.
GROUPS = {
    1: ["AirLapseV2"],                  # tuned for this dataset: see tune_airlapse_v2.py / --overrides
    6: ["MLP", "GRU", "LSTM", "PM25_GNN", "Crossformer", "Transformer", "AirDualODE"],
    2: ["STMamba", "Informer", "AirFormer"],
    3: ["TCN_DIR", "Autoformer"],
    4: ["AirDDE"],
    5: ["AirPhyNet"],
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


def train(model, repeat, args, base_cfg, timeout=None, name=None):
    """One repeat (seed = repeat) of one model; returns the run's repeat directory.
    `name` (default: the model) labels its outputs - e.g. a tuned variant.
    Raises subprocess.TimeoutExpired (after killing train.py) if it runs past timeout s."""
    name = name or model
    cfg = yaml.safe_load(yaml.safe_dump(base_cfg))
    for section in ("experiments", "train"):                  # --overrides (e.g. tuned hyperparameters)
        cfg[section].update(args.override_cfg.get(section, {}))
    cfg["experiments"].update(model=model, save_npy=True)
    cfg["train"].update(epochs=args.epochs, exp_repeat=1, early_stop=args.early_stop)
    cfg_fp = os.path.join(METRICS, "configs", f"config_{name}_rep{repeat}.yaml")
    os.makedirs(os.path.dirname(cfg_fp), exist_ok=True)
    yaml.safe_dump(cfg, open(cfg_fp, "w"), sort_keys=False)
    # each run gets its own output root: train.py names its folder by the
    # second it starts, so two streams starting together would collide
    run_root = os.path.join(RESULTS, f"{name}_rep{repeat}")
    shutil.rmtree(run_root, ignore_errors=True)
    env = dict(os.environ, CONFIG_FP=cfg_fp, RESULTS_DIR=run_root, SEED=str(repeat))
    if args.gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    r = subprocess.run([sys.executable, "train.py"], cwd=REPO, env=env, capture_output=True,
                       text=True, timeout=timeout)
    log_fp = os.path.join(METRICS, "logs", f"{name}_rep{repeat}.log")
    os.makedirs(os.path.dirname(log_fp), exist_ok=True)
    open(log_fp, "w", encoding="utf-8").write(r.stdout + "\n" + r.stderr)
    if r.returncode != 0:
        print(r.stderr[-3000:])
        raise RuntimeError(f"{model} failed (full log: {log_fp})")
    metric_fp = r.stdout.strip().splitlines()[-1]   # train.py prints its metric file last
    run_dir = os.path.dirname(metric_fp)
    keep = os.path.join(METRICS, "runs", name, os.path.basename(run_dir))
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
    ap.add_argument("--max_hours", type=float, default=None,
                    help="don't START a new run after this many hours (e.g. 9 on Kaggle's 12 h limit)")
    ap.add_argument("--deadline_hours", type=float, default=None,
                    help="hard stop: a run still going at this point is killed (and redone next "
                         "time); finished runs are kept. Use it to stay inside a GPU quota.")
    ap.add_argument("--repeat_ids", type=int, nargs="+", default=None,
                    help="run only these repeat indices/seeds (e.g. 0 2 4) - to split one model "
                         "over two GPUs")
    ap.add_argument("--gpu", type=int, default=None, help="GPU index for train.py (CUDA_VISIBLE_DEVICES)")
    ap.add_argument("--overrides", default=None,
                    help="YAML with experiments:/train: keys merged into config.yaml for these runs "
                         "(e.g. tuning/airlapsev2_best.yaml from tune_airlapse_v2.py)")
    ap.add_argument("--label", default=None,
                    help="name for the results instead of the model name (one model only), "
                         "e.g. AirLapseV2_tuned, so default and tuned runs are kept apart")
    args = ap.parse_args()
    models = args.models or (GROUPS[args.group] if args.group else DEFAULT_MODELS)
    if args.label and len(models) != 1:
        sys.exit("--label needs exactly one model")
    args.override_cfg = yaml.safe_load(open(args.overrides, encoding="utf-8")) if args.overrides else {}
    if args.overrides:
        print(f"overrides from {args.overrides}:",
              {k: v for k, v in args.override_cfg.items() if k in ("experiments", "train")})

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
    names = {m: (args.label or m) for m in models}
    if args.rerun:
        for m in models:
            for f in glob.glob(os.path.join(METRICS, "scores", f"{names[m]}__rep*.csv")):
                os.remove(f)
    failed = []
    t_begin, out_of_time = time.time(), False

    repeat_ids = args.repeat_ids if args.repeat_ids is not None else list(range(args.repeats))
    for m in models:
        if out_of_time:
            break
        name = names[m]
        for k in repeat_ids:
            if os.path.exists(score_fp(name, k)):
                continue
            elapsed_h = (time.time() - t_begin) / 3600
            if args.max_hours and elapsed_h > args.max_hours:
                print(f"time budget of {args.max_hours} h used - stopping here; "
                      f"run again to continue from {m} repeat {k + 1}", flush=True)
                out_of_time = True
                break
            timeout = None
            if args.deadline_hours:
                timeout = (args.deadline_hours - elapsed_h) * 3600
                if timeout < 60:
                    print(f"deadline of {args.deadline_hours} h reached - stopping before "
                          f"{m} repeat {k + 1}", flush=True)
                    out_of_time = True
                    break
            t_start = time.time()
            print(f"== {name} repeat {k + 1} (seed {k})" + (f" on GPU {args.gpu}" if args.gpu is not None else ""),
                  flush=True)
            try:
                cfg, rep = train(m, k, args, base, timeout, name)
            except subprocess.TimeoutExpired:
                print(f"   deadline of {args.deadline_hours} h reached - {m} repeat {k + 1} stopped "
                      f"after {(time.time() - t_start) / 60:.0f} min and will be redone next time; "
                      f"all finished runs are saved", flush=True)
                out_of_time = True
                break
            except Exception as e:
                print(f"   FAILED: {e}", flush=True)
                failed.append(name)
                break                                   # later repeats would fail the same way
            H = cfg["train"]["hist_len"]
            pf = np.load(os.path.join(rep, "predict.npy"))
            lf = np.load(os.path.join(rep, "label.npy"))
            mask = window_mask(obs, t0, np.load(os.path.join(rep, "time.npy")), H)
            row = dict(model=name, status="ok", repeat=k, minutes=round((time.time() - t_start) / 60, 1),
                       **scores(pf[:, H:], lf[:, H:], mask))
            pd.DataFrame([row]).to_csv(score_fp(name, k), index=False)
            if not os.path.exists(score_fp("Persistence", 0)):
                persist = np.repeat(lf[:, H - 1:H], lf.shape[1] - H, axis=1)
                pd.DataFrame([dict(model="Persistence", status="ok", repeat=0,
                                   **scores(persist, lf[:, H:], mask))]).to_csv(
                    score_fp("Persistence", 0), index=False)
            if not args.keep_raw:                       # ~190 MB of arrays per repeat, scored already
                shutil.rmtree(os.path.join(RESULTS, f"{name}_rep{k}"), ignore_errors=True)
            merge_scores()
            print(f"   done in {(time.time() - t_start) / 60:.1f} min", flush=True)

    rows = merge_scores().to_dict("records") + [dict(model=m, status="failed") for m in failed]
    summarise(rows, os.path.join(METRICS, "benchmark_summary.csv"))


if __name__ == "__main__":
    main()
