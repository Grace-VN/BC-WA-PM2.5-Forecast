"""Measure every benchmark model's computational cost in ONE session on ONE GPU.

The efficiency block in train.py's metric files cannot be compared across the
benchmark: runs were spread over different sessions and GPUs, and its epoch
time includes data loading (which dominates for the smallest models). This
script measures all models the same way, back to back, without full training:

  params_total     all parameters of the module
  params_used      parameters that receive a gradient in a training step
                   (differs only when a model builds modules it never uses,
                   e.g. AirLapseV3_phys, which skips the encoder)
  train_ms_batch   one optimiser step (forward + loss + backward + step) on a
                   batch of 32, the model's own training loss as in train.py,
                   inputs already on the GPU - model compute only
  epoch_s          train_ms_batch x training batches per epoch (an estimate of
                   one epoch of pure compute)
  infer_ms_sample  forward pass in eval/no_grad mode over the whole test set,
                   per test sample (= one 24 h forecast for all 115 stations)
  peak_train_MB / peak_infer_MB   torch.cuda.max_memory_allocated during each

Every model runs in its own process (train.py builds the model and loads the
data at import), with the same seed, after --warmup untimed batches; GPU work
is synchronised before each clock reading. Results are appended to
METRICS_DIR/efficiency.csv as each model finishes, so an interrupted session
resumes with the models still missing. ODE models (AirPhyNet, AirDualODE,
AirDDE) use adaptive solvers whose cost changes during training; here they are
timed at initialisation.

A configuration variant of one model (e.g. the chosen fix F3) is measured with
--overrides FILE --label NAME, the same flags as benchmark.py; its row is named NAME.

Usage: python efficiency.py [--models AirLapseV3 GRU ...] [--batches 50]
                            [--warmup 5] [--infer_batches N] [--timeout_min 40] [--rerun]
       python efficiency.py --models AirLapseV3 --overrides ablations/F3_diff_fix_const_diffusivity.yaml
                            --label F3_diff_fix_const_diffusivity
       python efficiency.py --plot ACCURACY.csv [MORE.csv ...] [--drop AirLapseV3]
                            [--names F3_diff_fix_const_diffusivity=AirLapse]
         (ACCURACY.csv: benchmark_summary.csv, or any CSV/XLSX with columns
          model and RMSE_mean; several files are combined)
"""
import argparse
import json
import os
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.abspath(__file__))
METRICS = os.environ.get("METRICS_DIR", os.path.join(REPO, "results", "metrics"))
OUT = os.path.join(METRICS, "efficiency.csv")


def worker(model_name, n_batches, warmup, infer_batches=None):
    """Runs inside a child process whose CONFIG_FP selects model_name."""
    import numpy as np
    import torch
    sys.argv = [sys.argv[0]]
    import train as T                                    # loads config + data, defines get_model()

    torch.manual_seed(0)
    np.random.seed(0)
    dev = T.device
    sync = torch.cuda.synchronize if T.use_cuda else (lambda: None)
    loader = torch.utils.data.DataLoader(T.train_data, batch_size=T.batch_size, shuffle=True,
                                         drop_last=True)
    n_epoch_batches = len(loader)
    batches = []
    for pm25, feature, t in loader:                      # pre-load: time model compute, not I/O
        batches.append((pm25.to(dev), feature.to(dev), t))
        if len(batches) == warmup + n_batches:
            break

    model = T.get_model().to(dev)
    params_total = sum(p.numel() for p in model.parameters())
    opt = torch.optim.RMSprop(model.parameters(), lr=T.lr, weight_decay=T.weight_decay)

    T.train(batches[:1], model, opt)                     # one step: which parameters get a gradient
    params_used = sum(p.numel() for p in model.parameters() if p.grad is not None)
    T.train(batches[1:warmup], model, opt) if warmup > 1 else None

    if T.use_cuda:
        torch.cuda.reset_peak_memory_stats(dev)
    sync()
    t0 = time.perf_counter()
    T.train(batches[warmup:], model, opt)                # T.train is train.py's own training step
    sync()
    train_ms = (time.perf_counter() - t0) / n_batches * 1000
    peak_train = torch.cuda.max_memory_allocated(dev) / 1e6 if T.use_cuda else float("nan")
    del batches, opt

    test_loader = torch.utils.data.DataLoader(T.test_data, batch_size=T.batch_size, shuffle=False,
                                              drop_last=True)
    model.eval()
    if T.use_cuda:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(dev)
    infer_s, n = 0.0, 0
    with torch.no_grad():
        for i, (pm25, feature, _) in enumerate(test_loader):
            if infer_batches and i >= infer_batches + 2:
                break
            pm25, feature = pm25.to(dev), feature.to(dev)
            sync()
            t0 = time.perf_counter()
            model(pm25[:, :T.hist_len], feature)
            sync()
            if i >= 2:                                   # first batches warm up kernels
                infer_s += time.perf_counter() - t0
                n += pm25.shape[0]
    peak_infer = torch.cuda.max_memory_allocated(dev) / 1e6 if T.use_cuda else float("nan")

    print("EFFICIENCY_JSON " + json.dumps(dict(
        model=model_name, params_total=params_total, params_used=params_used,
        train_ms_batch=round(train_ms, 2), epoch_s=round(train_ms * n_epoch_batches / 1000, 1),
        infer_ms_sample=round(infer_s / max(n, 1) * 1000, 4), peak_train_MB=round(peak_train, 1),
        peak_infer_MB=round(peak_infer, 1), batches_timed=n_batches, epoch_batches=n_epoch_batches,
        gpu=torch.cuda.get_device_name(0) if T.use_cuda else "CPU", torch=torch.__version__)))


def run_all(args):
    import pandas as pd
    import yaml
    sys.path.insert(0, REPO)
    from benchmark import DEFAULT_MODELS
    models = args.models or DEFAULT_MODELS
    if args.label and len(models) != 1:
        sys.exit("--label needs exactly one model")
    overrides = yaml.safe_load(open(args.overrides, encoding="utf-8")) if args.overrides else {}
    names = {m: (args.label or m) for m in models}
    os.makedirs(METRICS, exist_ok=True)
    done = pd.read_csv(OUT) if os.path.exists(OUT) else pd.DataFrame(columns=["model"])
    if args.rerun:
        done = done[~done.model.isin(names.values())]
        done.to_csv(OUT, index=False)
    todo = [m for m in models if names[m] not in set(done.model)]
    print(f"{len(todo)} model(s) to measure: {' '.join(names[m] for m in todo) or 'none'} -> {OUT}", flush=True)

    base = yaml.safe_load(open(os.path.join(REPO, "config.yaml"), encoding="utf-8"))
    for m in todo:
        cfg = yaml.safe_load(yaml.safe_dump(base))
        for section in ("experiments", "train"):          # --overrides, as in benchmark.py
            cfg[section].update(overrides.get(section) or {})
        cfg["experiments"].update(model=m, save_npy=False)
        cfg["train"].update(exp_repeat=1)
        cfg_fp = os.path.join(METRICS, "configs", f"config_efficiency_{names[m]}.yaml")
        os.makedirs(os.path.dirname(cfg_fp), exist_ok=True)
        yaml.safe_dump(cfg, open(cfg_fp, "w"), sort_keys=False)
        env = dict(os.environ, CONFIG_FP=cfg_fp, SEED="0",
                   RESULTS_DIR=os.environ.get("RESULTS_DIR", os.path.join(REPO, "results")))
        t0 = time.time()
        print(f"== {names[m]}", flush=True)
        try:
            r = subprocess.run([sys.executable, __file__, "--worker", names[m], "--batches", str(args.batches),
                                "--warmup", str(args.warmup)] + (["--infer_batches", str(args.infer_batches)] if args.infer_batches else []), cwd=REPO, env=env, capture_output=True,
                               text=True, timeout=args.timeout_min * 60)
        except subprocess.TimeoutExpired:
            print(f"   timed out after {args.timeout_min} min - skipped", flush=True)
            continue
        line = [l for l in r.stdout.splitlines() if l.startswith("EFFICIENCY_JSON ")]
        if r.returncode or not line:
            print((r.stdout + r.stderr)[-2500:], flush=True)
            print(f"   {names[m]} failed", flush=True)
            continue
        row = json.loads(line[-1].split(" ", 1)[1])
        pd.DataFrame([row]).to_csv(OUT, mode="a", header=not os.path.exists(OUT), index=False)
        print(f"   {row['train_ms_batch']} ms/batch, ~{row['epoch_s']} s/epoch, "
              f"{row['infer_ms_sample']} ms/sample, {row['peak_train_MB']} MB "
              f"({time.time() - t0:.0f} s)", flush=True)

    df = pd.read_csv(OUT)
    if df.gpu.nunique() > 1:
        print(f"WARNING: efficiency.csv mixes GPUs {sorted(df.gpu.unique())} - rerun on one GPU")
    pd.set_option("display.width", 250)
    print(df.drop(columns=["torch", "batches_timed"]).sort_values("epoch_s").to_string(index=False))


def plot(acc_fps, drop=(), names=()):
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    eff = pd.read_csv(OUT)
    accs = []
    for fp in acc_fps:
        a = pd.read_excel(fp) if fp.endswith(".xlsx") else pd.read_csv(fp)
        a["model"] = a.model.astype(str).str.replace(r"\s*\(\d{4}\)", "", regex=True)
        a["model"] = a.model.replace({"AirLapse": "AirLapseV3", "Airformer": "AirFormer"})   # result.xlsx names
        accs.append(a[["model", "RMSE_mean"]])
    acc = pd.concat(accs).drop_duplicates("model", keep="last")
    df = eff.merge(acc, on="model").dropna(subset=["RMSE_mean"])
    df = df[~df.model.isin(drop)].copy()
    rename = dict(n.split("=", 1) for n in names)
    df["model"] = df.model.replace(rename)
    print(df[["model", "params_used", "epoch_s", "infer_ms_sample", "peak_train_MB", "RMSE_mean"]]
          .sort_values("RMSE_mean").to_string(index=False))
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                         "mathtext.fontset": "stix", "font.size": 8, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.dpi": 300, "savefig.bbox": "tight",
                         "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    size = 12 + 120 * np.sqrt(df.params_used / df.params_used.max())
    ours = df.model.str.startswith("AirLapse") | df.model.isin(rename.values())
    ax.scatter(df.epoch_s[~ours], df.RMSE_mean[~ours], s=size[~ours], c="0.6", edgecolor="k", lw=0.4, zorder=2)
    ax.scatter(df.epoch_s[ours], df.RMSE_mean[ours], s=size[ours], c="#D55E00", edgecolor="k", lw=0.4, zorder=3)
    for _, r in df.iterrows():
        ax.annotate(r.model, (r.epoch_s, r.RMSE_mean), xytext=(4, 2), textcoords="offset points", fontsize=6)
    ax.set(xscale="log", xlabel=f"Training time per epoch (s, {df.gpu.iloc[0]})",
           ylabel=r"Test RMSE ($\mu$g m$^{-3}$)")
    out = os.path.join(METRICS, "efficiency_vs_accuracy")
    fig.savefig(out + ".png")
    fig.savefig(out + ".pdf")
    print(f"saved {out}.png/.pdf (marker area ~ parameters used)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--batches", type=int, default=50, help="timed training batches per model")
    ap.add_argument("--warmup", type=int, default=5, help="untimed training batches first")
    ap.add_argument("--infer_batches", type=int, default=None, help="limit timed test batches (default: whole test set)")
    ap.add_argument("--timeout_min", type=float, default=40)
    ap.add_argument("--rerun", action="store_true", help="measure these models again")
    ap.add_argument("--overrides", default=None, help="YAML merged into config.yaml (as benchmark.py)")
    ap.add_argument("--label", default=None, help="row name for the overridden model")
    ap.add_argument("--plot", metavar="ACCURACY_CSV", nargs="+", default=None)
    ap.add_argument("--drop", nargs="+", default=[], help="--plot: models to leave out")
    ap.add_argument("--names", nargs="+", default=[], help="--plot: OLD=NEW display names")
    ap.add_argument("--worker", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.worker:
        worker(args.worker, args.batches, args.warmup, args.infer_batches)
    elif args.plot:
        plot(args.plot, args.drop, args.names)
    else:
        run_all(args)


if __name__ == "__main__":
    main()
