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

--base NAME rebases the study on another scenario (e.g. the chosen fix,
F3_diff_fix_const_diffusivity): every A*/S* scenario is merged on top of it
(airlapse_ablate lists are combined), labelled <prefix>+<scenario> (e.g.
F3+A1_no_advection), and compared with the base's own run of the same seed,
which is the reference instead of AirLapseV3. F* scenarios and scenarios that
add nothing to the base (e.g. A3 on F3) are skipped. Merged override files are
kept in METRICS_DIR/overrides/.

Usage: python run_ablations.py [--only A1_no_advection S_hist_len_48 ...]
                               [--base F3_diff_fix_const_diffusivity]
                               [--part 0 --of 2 --gpu 0] [--seed 0]
                               [--deadline_hours H] [--summary]
Writes METRICS_DIR/ablation_summary.csv (or ablation_summary_<prefix>.csv with
--base): each scenario and its change vs the reference.
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


def load(fp):
    return yaml.safe_load(open(fp, encoding="utf-8")) or {}


def scenarios():
    """[(name, model, yaml path)] for every ablations/*.yaml."""
    out = []
    for fp in sorted(glob.glob(os.path.join(REPO, "ablations", "*.yaml"))):
        out.append((os.path.splitext(os.path.basename(fp))[0], load(fp).get("model", FULL), fp))
    return out


def merge(base_cfg, cfg):
    """cfg's experiments/train keys on top of base_cfg's; airlapse_ablate lists are combined."""
    out = {}
    for sec in ("experiments", "train"):
        d = dict(base_cfg.get(sec) or {})
        for k, v in (cfg.get(sec) or {}).items():
            if k == "airlapse_ablate":
                v = sorted(set(d.get(k) or []) | set(v or []))
            d[k] = v
        if d:
            out[sec] = d
    out["model"] = cfg.get("model", base_cfg.get("model", FULL))
    return out


def study(base=None):
    """(reference, [(label, model, overrides path)]) for the plain or the rebased study."""
    plain = scenarios()
    if not base:
        return (FULL, FULL, None), plain
    by_name = {n: (m, fp) for n, m, fp in plain}
    if base not in by_name:
        sys.exit(f"unknown base scenario: {base}")
    base_model, base_fp = by_name[base]
    base_cfg = load(base_fp)
    base_key = merge(base_cfg, {})
    prefix = base.split("_")[0]
    os.makedirs(os.path.join(METRICS, "overrides"), exist_ok=True)
    out = []
    for name, _, fp in plain:
        if name.startswith("F"):
            continue                                  # the fix variants themselves
        cfg = merge(base_cfg, load(fp))
        if cfg == base_key:
            continue                                  # already part of the base (e.g. A3 on F3)
        label = f"{prefix}+{name}"
        mfp = os.path.join(METRICS, "overrides", f"{label}.yaml")
        with open(mfp, "w", encoding="utf-8") as f:
            f.write(f"# {name} on top of {base} (written by run_ablations.py --base)\n")
            yaml.safe_dump(cfg, f, sort_keys=False)
        out.append((label, cfg["model"], mfp))
    return (base, base_model, base_fp), out


def score_fp(name, seed):
    return os.path.join(METRICS, "scores", f"{name}__rep{seed}.csv")


def summary(seed, base=None):
    (ref_name, _, _), todo = study(base)
    full_fp = score_fp(ref_name, seed)
    if not os.path.exists(full_fp):
        print(f"no reference run yet ({full_fp})")
        return
    ref = pd.read_csv(full_fp).iloc[0]
    rows = [dict(scenario=f"{ref_name} (reference)", **{c: ref[c] for c in COLS})]
    for name, _, _ in todo:
        if os.path.exists(score_fp(name, seed)):
            r = pd.read_csv(score_fp(name, seed)).iloc[0]
            rows.append(dict(scenario=name, **{c: r[c] for c in COLS}))
    df = pd.DataFrame(rows).set_index("scenario")
    for c in ("RMSE", "MAE", "RMSE_gt35.5", "CSI@35.5"):
        df[f"d{c}"] = df[c] - ref[c]
    df["dRMSE_%"] = 100 * df["dRMSE"] / ref["RMSE"]
    out = "ablation_summary.csv" if not base else f"ablation_summary_{base.split('_')[0]}.csv"
    df.to_csv(os.path.join(METRICS, out))
    pd.set_option("display.width", 250)
    print(df.round(3).to_string())
    print(f"\nd* = scenario minus {ref_name} (positive dRMSE/dMAE = worse without it). "
          "One run each: compare with the reference's std over seeds.")


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
    ap.add_argument("--base", default=None,
                    help="run every A*/S* scenario on top of this scenario (e.g. F3_diff_fix_const_diffusivity)")
    args = ap.parse_args()
    if args.summary:
        summary(args.seed, args.base)
        return

    ref, todo = study(args.base)
    if args.only:
        unknown = set(args.only) - {n for n, _, _ in todo}
        if args.base:                                 # accept plain names too: A1_no_advection
            pre = args.base.split("_")[0] + "+"
            args.only = [o if o.startswith(pre) or o in {n for n, _, _ in todo} else pre + o
                         for o in args.only]
            unknown = set(args.only) - {n for n, _, _ in todo}
        if unknown:
            sys.exit(f"unknown scenario(s): {sorted(unknown)}")
        todo = [s for s in todo if s[0] in args.only]
    todo = todo[args.part::args.of]
    if args.part == 0 and not os.path.exists(score_fp(ref[0], args.seed)):
        todo = [ref] + todo                          # reference run with the same seed
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
    summary(args.seed, args.base)


if __name__ == "__main__":
    main()
