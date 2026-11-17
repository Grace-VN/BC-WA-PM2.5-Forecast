"""Run the AirLapse ablation and sensitivity scenarios in ablations/, one run each.

Every ablations/<NAME>.yaml is one scenario: an optional `model:` (default
AirLapseV3) plus experiments:/train: keys merged into config.yaml, exactly like
benchmark.py --overrides. Each scenario is trained once, with the same seed as
the full model's reference run (AirLapseV3, repeat --seed), via benchmark.py,
so it is scored the same way (forecast hours, measured values only) and its
score file lands in METRICS_DIR/scores/<NAME>__rep<seed>.csv. Scenarios that
already have a score file are skipped, so an interrupted session resumes.

  A*  component ablations - one physical ingredient or module removed
  S*  sensitivity - one hyperparameter moved away from its default

The full model's run with the same seed is the reference: it is trained first
if its score file is missing. With one run per scenario, compare differences
against the full model's spread over seeds (benchmark_summary.csv).

Usage: python run_ablations.py [--only A1_no_advection S_hist_len_48 ...]
                               [--part 0 --of 2 --gpu 0] [--seed 0]
                               [--deadline_hours H] [--summary]
Writes METRICS_DIR/ablation_summary.csv (each scenario and its change vs the full model).
"""
import argparse
import glob
import os
import subprocess
import sys
import time

import pandas as pd
import yaml

REPO = os.path.dirname(os.path.abspath(__file__))
METRICS = os.environ.get("METRICS_DIR", os.path.join(REPO, "results", "metrics"))
FULL = "AirLapseV3"
COLS = ["RMSE", "MAE", "MAPE", "RMSE_gt35.5", "CSI@35.5", "POD@35.5", "FAR@35.5", "CSI@55.5"]


def scenarios():
    out = []
    for fp in sorted(glob.glob(os.path.join(REPO, "ablations", "*.yaml"))):
        cfg = yaml.safe_load(open(fp, encoding="utf-8")) or {}
        out.append((os.path.splitext(os.path.basename(fp))[0], cfg.get("model", FULL), fp))
    return out


def score_fp(name, seed):
    return os.path.join(METRICS, "scores", f"{name}__rep{seed}.csv")


def summary(seed):
    full_fp = score_fp(FULL, seed)
    if not os.path.exists(full_fp):
        print(f"no reference run yet ({full_fp})")
        return
    ref = pd.read_csv(full_fp).iloc[0]
    rows = [dict(scenario=f"{FULL} (full)", **{c: ref[c] for c in COLS})]
    for name, _, _ in scenarios():
        if os.path.exists(score_fp(name, seed)):
            r = pd.read_csv(score_fp(name, seed)).iloc[0]
            rows.append(dict(scenario=name, **{c: r[c] for c in COLS}))
    df = pd.DataFrame(rows).set_index("scenario")
    for c in ("RMSE", "MAE", "RMSE_gt35.5", "CSI@35.5"):
        df[f"d{c}"] = df[c] - ref[c]
    df["dRMSE_%"] = 100 * df["dRMSE"] / ref["RMSE"]
    df.to_csv(os.path.join(METRICS, "ablation_summary.csv"))
    pd.set_option("display.width", 250)
    print(df.round(3).to_string())
    print("\nd* = scenario minus full model (positive dRMSE/dMAE = worse without it). "
          "One run each: compare with the full model's std over seeds.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", default=None, help="scenario names (file names without .yaml)")
    ap.add_argument("--part", type=int, default=0, help="run every --of-th scenario starting here (split over GPUs)")
    ap.add_argument("--of", type=int, default=1)
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0, help="repeat index/seed, the same for every scenario")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--early_stop", type=int, default=10)
    ap.add_argument("--deadline_hours", type=float, default=None,
                    help="hard stop for this whole session; unfinished runs are redone next time")
    ap.add_argument("--summary", action="store_true", help="only print/save the comparison table")
    args = ap.parse_args()
    if args.summary:
        summary(args.seed)
        return

    todo = scenarios()
    if args.only:
        unknown = set(args.only) - {n for n, _, _ in todo}
        if unknown:
            sys.exit(f"unknown scenario(s): {sorted(unknown)}")
        todo = [s for s in todo if s[0] in args.only]
    todo = todo[args.part::args.of]
    if args.part == 0 and not os.path.exists(score_fp(FULL, args.seed)):
        todo = [(FULL, FULL, None)] + todo           # reference run with the same seed
    todo = [s for s in todo if not os.path.exists(score_fp(s[0], args.seed))]
    print(f"{len(todo)} scenario(s) to run: {' '.join(n for n, _, _ in todo) or 'none'}", flush=True)

    t0 = time.time()
    for name, model, fp in todo:
        cmd = [sys.executable, "benchmark.py", "--models", model, "--repeats", str(args.seed + 1),
               "--repeat_ids", str(args.seed), "--epochs", str(args.epochs),
               "--early_stop", str(args.early_stop)]
        if fp:
            cmd += ["--overrides", fp, "--label", name]
        if args.gpu is not None:
            cmd += ["--gpu", str(args.gpu)]
        if args.deadline_hours:
            left = args.deadline_hours - (time.time() - t0) / 3600
            if left * 60 < 5:
                print("deadline reached - run again to continue", flush=True)
                break
            cmd += ["--deadline_hours", f"{left:.3f}"]
        print(f"== {name} ({model})", flush=True)
        r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
        if not os.path.exists(score_fp(name, args.seed)):
            print(r.stdout[-2000:] + r.stderr[-2000:], flush=True)
            print(f"   {name}: no result (failed or stopped at the deadline)", flush=True)
            if "deadline" in r.stdout:
                break
        else:
            print(f"   done after {(time.time() - t0) / 60:.0f} min in total", flush=True)
    summary(args.seed)


if __name__ == "__main__":
    main()
