"""Optuna hyperparameter search for AirLapseV2 on the BC-WA dataset.

AirLapseV2's defaults were set on KnowAir (184 Chinese cities, 3-hourly).
The search ranges below come from eda_airlapse_priors.py on this dataset's
TRAINING period:
  - stations are close (median nearest neighbour 27 km) and PM2.5
    correlation drops from 0.63 (< 25 km) to ~0.3 beyond 100 km, so the
    default 375 km neighbourhood (~74 neighbours per station) is far too
    wide -> dist_threshold_km 50-300, sigma_d 25-300;
  - an upwind neighbour's past PM2.5 leads at 1-4 h (peak 2 h), not beyond
    6 h -> max_lag 2-8 (hours: dt is 1 h here), sigma_tau_init_h 0.5-4;
  - elevation difference matters little -> sigma_h 300-3000 m.
hist_len / pred_len / batch size / epochs / early stopping stay as in
config.yaml, identical to every other benchmark model.

Model selection uses ONLY the validation split (val loss, normalised MSE):
the test split is never touched here. Each trial trains for at most
--search_epochs with patience --search_early_stop and Optuna's median
pruner. Trials are seeded (seed 0), so trials differ only by their
hyperparameters.

The study lives in an SQLite file, so the search resumes where it stopped,
and several workers can share it - e.g. one per GPU:
  CUDA_VISIBLE_DEVICES=0 python tune_airlapse_v2.py --timeout_hours 4 &
  CUDA_VISIBLE_DEVICES=1 python tune_airlapse_v2.py --timeout_hours 4
Afterwards (or any time): python tune_airlapse_v2.py --export
writes the best configuration to <metrics>/tuning/airlapsev2_best.yaml, the
file benchmark.py takes as --overrides:
  python benchmark.py --models AirLapseV2 --label AirLapseV2_tuned \\
      --overrides <metrics>/tuning/airlapsev2_best.yaml

Paths: METRICS_DIR (default ./results/metrics), as in benchmark.py.
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
import yaml

REPO = os.path.dirname(os.path.abspath(__file__))
METRICS = os.environ.get("METRICS_DIR", os.path.join(os.environ.get("RESULTS_DIR", os.path.join(REPO, "results")), "metrics"))
TUNE_DIR = os.path.join(METRICS, "tuning")

SEARCH = {
    'airlapsev2_hidden_dim': [32, 64, 96, 128],
    'airlapsev2_latent_dim': [8, 16, 32],
    'airlapsev2_attn_dim': [16, 32, 48],
    'airlapsev2_dropout': (0.0, 0.4),
    'airlapsev2_max_lag': (2, 8),                        # hours (dt = 1 h)
    'airlapsev2_dist_threshold_km': (50.0, 300.0),
    'airlapsev2_sigma_d': (25.0, 300.0),
    'airlapsev2_sigma_h': (300.0, 3000.0),
    'airlapsev2_sigma_tau_init_h': (0.5, 4.0),
    'airlapsev2_diff_hidden_dim': [8, 16, 32],
    'airlapsev2_diffusivity_along_init': (5.0, 150.0),   # km^2/h
    'airlapsev2_diffusivity_cross_init': (2.0, 100.0),
    'lr': (1e-4, 3e-3),                                  # log scale
    'weight_decay': (1e-5, 1e-3),                        # log scale
}


def suggest(trial):
    p = {}
    for k in ('airlapsev2_hidden_dim', 'airlapsev2_latent_dim', 'airlapsev2_attn_dim', 'airlapsev2_diff_hidden_dim'):
        p[k] = trial.suggest_categorical(k, SEARCH[k])
    for k in ('airlapsev2_dropout', 'airlapsev2_dist_threshold_km', 'airlapsev2_sigma_d', 'airlapsev2_sigma_h',
              'airlapsev2_sigma_tau_init_h', 'airlapsev2_diffusivity_along_init',
              'airlapsev2_diffusivity_cross_init'):
        p[k] = trial.suggest_float(k, *SEARCH[k])
    p['airlapsev2_max_lag'] = trial.suggest_int('airlapsev2_max_lag', *SEARCH['airlapsev2_max_lag'])
    mode = trial.suggest_categorical('airlapsev2_spatial_mix_mode', ['bottleneck', 'per_step'])
    p['airlapsev2_spatial_mix_mode'] = mode
    p['airlapsev2_num_layers'] = trial.suggest_int('airlapsev2_num_layers', 1, 2) if mode == 'bottleneck' else 1
    p['lr'] = trial.suggest_float('lr', *SEARCH['lr'], log=True)
    p['weight_decay'] = trial.suggest_float('weight_decay', *SEARCH['weight_decay'], log=True)
    return p


def best_to_yaml(study, fp):
    """Best trial -> overrides file for benchmark.py ({experiments: ..., train: ...})."""
    params = dict(study.best_trial.params)
    if params.get('airlapsev2_spatial_mix_mode') == 'per_step':
        params['airlapsev2_num_layers'] = 1
    train_part = {k: params.pop(k) for k in ('lr', 'weight_decay')}
    doc = {'experiments': params, 'train': train_part,
           '_search': {'best_val_loss': float(study.best_value), 'best_trial': study.best_trial.number,
                       'n_trials_complete': sum(t.value is not None for t in study.trials),
                       'n_trials_total': len(study.trials)}}
    os.makedirs(os.path.dirname(fp), exist_ok=True)
    yaml.safe_dump(doc, open(fp, 'w'), sort_keys=False)
    return doc


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--n_trials', type=int, default=None, help='trials for THIS worker (default: until timeout)')
    ap.add_argument('--timeout_hours', type=float, default=None, help='stop starting trials after this')
    ap.add_argument('--search_epochs', type=int, default=15)
    ap.add_argument('--search_early_stop', type=int, default=3)
    ap.add_argument('--study', default='airlapsev2_bcwa')
    ap.add_argument('--storage', default=os.path.join(TUNE_DIR, 'airlapsev2_bcwa.db'))
    ap.add_argument('--export', action='store_true', help='only write the best config so far and print the top trials')
    ap.add_argument('--smoke', action='store_true', help='tiny data subset + 1 epoch, to test the pipeline')
    args = ap.parse_args()
    try:
        import optuna
    except ImportError:
        sys.exit("optuna is needed: pip install optuna")
    os.makedirs(os.path.dirname(args.storage), exist_ok=True)
    # Workers started together race to create the database schema; the loser
    # gets "table ... already exists" - wait and retry instead of dying.
    for attempt in range(12):
        try:
            storage = optuna.storages.RDBStorage(f'sqlite:///{args.storage}',
                                                 engine_kwargs={'connect_args': {'timeout': 60}})
            study = optuna.create_study(study_name=args.study, storage=storage, load_if_exists=True,
                                        direction='minimize', sampler=optuna.samplers.TPESampler(seed=None),
                                        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3))
            break
        except Exception as e:                      # sqlalchemy OperationalError / IntegrityError
            if attempt == 11:
                raise
            print(f'study database busy ({type(e).__name__}), retrying in 5 s', flush=True)
            time.sleep(5)
    out_yaml = os.path.join(os.path.dirname(args.storage), 'airlapsev2_best.yaml')

    if not args.export:
        sys.argv = [sys.argv[0]]
        import train as T                       # loads config, data and graph once
        T.exp_model = 'AirLapseV2'
        base_exp = dict(T.config['experiments'])
        train_data = T.train_data
        if args.smoke:
            train_data = torch.utils.data.Subset(T.train_data, range(0, len(T.train_data), 40))
            args.search_epochs = 1

        def objective(trial):
            p = suggest(trial)
            lr, wd = p.pop('lr'), p.pop('weight_decay')
            T.config['experiments'] = {**base_exp, **p}
            torch.manual_seed(0); np.random.seed(0)
            model = T.get_model().to(T.device)
            opt = torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=wd)   # same optimiser as train.py
            tl = torch.utils.data.DataLoader(train_data, batch_size=T.batch_size, shuffle=True, drop_last=True)
            vl = torch.utils.data.DataLoader(T.val_data, batch_size=T.batch_size, shuffle=False, drop_last=True)
            best, best_ep = float('inf'), 0
            for ep in range(args.search_epochs):
                T.train(tl, model, opt)
                v = T.val(vl, model)
                if not np.isfinite(v):
                    raise optuna.TrialPruned()
                if v < best:
                    best, best_ep = v, ep
                trial.report(v, ep)
                if trial.should_prune():
                    raise optuna.TrialPruned()
                if ep - best_ep >= args.search_early_stop:
                    break
            return best

        t0 = time.time()
        study.optimize(objective, n_trials=args.n_trials,
                       timeout=args.timeout_hours * 3600 if args.timeout_hours else None,
                       catch=(RuntimeError,))
        print(f'worker done after {(time.time() - t0) / 3600:.2f} h')

    done = [t for t in study.trials if t.value is not None]
    if not done:
        sys.exit('no completed trial yet')
    doc = best_to_yaml(study, out_yaml)
    df = study.trials_dataframe().sort_values('value')
    df.to_csv(os.path.join(os.path.dirname(args.storage), 'airlapsev2_trials.csv'), index=False)
    print(f'{len(done)} completed / {len(study.trials)} trials; best val loss {study.best_value:.4f} '
          f'(trial {study.best_trial.number})')
    print(yaml.safe_dump({k: v for k, v in doc.items() if k != '_search'}, sort_keys=False))
    print(f'-> {out_yaml}')


if __name__ == '__main__':
    main()
