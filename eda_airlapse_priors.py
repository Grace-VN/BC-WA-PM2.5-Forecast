"""EDA of the BC-WA dataset for AirLapseV2's physics hyperparameters.

AirLapseV2's defaults were set on KnowAir (184 Chinese cities, 3-hourly).
This script measures the quantities those hyperparameters encode, on this
dataset, using measured (not gap-filled) PM2.5 only:
  - station spacing / neighbours within a radius   -> dist_threshold_km
  - PM2.5 correlation decay with distance          -> sigma_d
  - correlation decay with elevation difference    -> sigma_h
  - wind-implied travel time between neighbours    -> max_lag, sigma_tau_init_h
  - persistence (autocorrelation)                  -> context for hist_len
  - does the upwind neighbour lead at its travel time? -> is transport worth modelling
Uses the TRAINING period only, so nothing about val/test feeds the choices.

Usage: python eda_airlapse_priors.py   (prints a report; writes eda/airlapse_priors.json)
"""
import json
import os

import numpy as np
import pandas as pd
import yaml

REPO = os.path.dirname(os.path.abspath(__file__))
cfg = yaml.safe_load(open(os.path.join(REPO, 'config.yaml'), encoding='utf-8'))
ds = cfg['dataset'][1]
arr = np.load(os.path.join(REPO, ds['data_fp']))
obs = np.load(os.path.join(REPO, ds['data_fp'].replace('.npy', '_observed.npy')))
sites = pd.read_csv(os.path.join(REPO, 'data', 'BCWA_AirNow_2y_sites.csv'))
hours = pd.date_range('2024-10-01', periods=len(arr), freq='h')
train = hours <= pd.Timestamp(*ds['train_end'][0])
pm = np.where(obs, arr[..., -1], np.nan)[train]                     # [T, N] measured only
ws_kmh = arr[train, :, 3] * 3.6                                     # 10 m wind speed
wd = arr[train, :, 4]                                               # direction wind comes FROM (deg)
N = pm.shape[1]
out = {}


def hav(la1, lo1, la2, lo2):
    la1, lo1, la2, lo2 = map(np.radians, (la1, lo1, la2, lo2))
    a = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 2 * 6371 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


lat, lon, elev = sites.lat.values, sites.lon.values, sites.elevation.values
D = hav(lat[:, None], lon[:, None], lat[None], lon[None])
dH = np.abs(elev[:, None] - elev[None])
iu = np.triu_indices(N, 1)

# ---- spacing / neighbours
nn = np.sort(D + np.eye(N) * 1e9, axis=1)
print('== Station spacing')
print(f'nearest neighbour km: median {np.median(nn[:, 0]):.0f}, p90 {np.percentile(nn[:, 0], 90):.0f}, max {nn[:, 0].max():.0f}')
print(f'5th neighbour km:     median {np.median(nn[:, 4]):.0f}, p90 {np.percentile(nn[:, 4], 90):.0f}')
for r in (50, 100, 150, 200, 300, 375):
    k = (D <= r).sum(1) - 1
    print(f'  neighbours within {r:3d} km: median {np.median(k):.0f}, stations with none: {(k == 0).sum()}')
out['nn_km_median'] = float(np.median(nn[:, 0]))

# ---- correlation vs distance (daily-mean anomalies would remove the diurnal cycle; use hourly as the model sees it)
z = (pm - np.nanmean(pm, 0)) / np.nanstd(pm, 0)
C = pd.DataFrame(z).corr(min_periods=500).values
pairs = pd.DataFrame({'d': D[iu], 'dh': dH[iu], 'r': C[iu]}).dropna()
bins = [0, 25, 50, 100, 150, 200, 300, 400, 600, 1200]
print('\n== Hourly PM2.5 correlation vs distance (training period)')
g = pairs.groupby(pd.cut(pairs.d, bins)).r.agg(['median', 'count'])
print(g.round(2).to_string())
# e-folding scale of r(d) ~ exp(-d / L), fitted on pairs within 600 km with r > 0
p = pairs[(pairs.d < 600) & (pairs.r > 0.05)]
L = -1 / np.polyfit(p.d, np.log(p.r), 1)[0]
print(f'e-folding distance of correlation: {L:.0f} km')
out['corr_efold_km'] = float(L)

# ---- elevation, at comparable distance (< 150 km)
near = pairs[pairs.d < 150]
print('\n== Correlation vs elevation difference (pairs < 150 km apart)')
print(near.groupby(pd.cut(near.dh, [-1, 100, 300, 600, 1000, 2000])).r.agg(['median', 'count']).round(2).to_string())
pn = near[near.r > 0.05]
coef = np.polyfit(pn.dh, np.log(pn.r), 1)[0]
print(f'e-folding elevation difference: {(-1 / coef if coef < 0 else float("inf")):.0f} m')
out['corr_efold_m'] = float(-1 / coef) if coef < 0 else None

# ---- persistence
s = pd.Series(np.nanmean(pm, 1))
acf = {h: s.autocorr(h) for h in (1, 3, 6, 12, 24, 48)}
print('\n== Autocorrelation of station-mean PM2.5:', {k: round(v, 2) for k, v in acf.items()})

# ---- wind-implied travel time between neighbours (< 150 km)
print('\n== Wind speed (10 m) km/h: median %.1f, p25 %.1f, p75 %.1f' %
      tuple(np.nanpercentile(ws_kmh, q) for q in (50, 25, 75)))
ii, jj = np.where((D > 0) & (D < 150))
tau = D[ii, jj][None] / np.maximum(ws_kmh[:, ii], 1.0)            # [T, pairs] hours
print('travel time (pairs < 150 km), hours: median %.1f, p25 %.1f, p75 %.1f, share <= 6 h %.2f, <= 12 h %.2f' % (
    np.median(tau), np.percentile(tau, 25), np.percentile(tau, 75), (tau <= 6).mean(), (tau <= 12).mean()))
out['tau_median_h'] = float(np.median(tau))

# ---- does the upwind neighbour lead at its travel time?
# For pairs < 150 km: is j's PM2.5 at t-k more correlated with i's at t when the wind at j blows towards i
# and k matches the travel time, than unconditionally?
brg = np.degrees(np.arctan2(np.sin(np.radians(lon[None] - lon[:, None])) * np.cos(np.radians(lat[None])),
                            np.cos(np.radians(lat[:, None])) * np.sin(np.radians(lat[None])) -
                            np.sin(np.radians(lat[:, None])) * np.cos(np.radians(lat[None])) *
                            np.cos(np.radians(lon[None] - lon[:, None])))) % 360      # bearing from row -> col
print('\n== Lead of the upwind neighbour (pairs < 150 km): mean z_j(t-k) * z_i(t)')
zz = np.nan_to_num(z)
rows = []
for k in (0, 1, 2, 3, 4, 6, 8, 12):
    src, dst = jj, ii                                    # j upwind source, i receptor
    to_dst = (wd[:, src] + 180) % 360                    # direction the wind at j blows TO
    ang = np.abs((to_dst - brg[src, dst] + 180) % 360 - 180)
    towards = ang < 45
    t_match = np.abs(D[src, dst][None] / np.maximum(ws_kmh[:, src], 1.0) - k) <= max(1, 0.5 * k)
    lagged = np.zeros_like(zz[:, src]); lagged[k:] = zz[:len(zz) - k, src] if k else zz[:, src]
    prod = lagged * zz[:, dst]
    valid = np.arange(len(zz))[:, None] >= k
    m = towards & t_match & valid
    rows.append(dict(lag_h=k, matched=prod[m].mean() if m.any() else np.nan, n=int(m.sum()),
                     unconditional=prod[valid.repeat(len(src), 1)].mean()))
lead = pd.DataFrame(rows)
lead['lift'] = lead.matched - lead.unconditional
print(lead.round(3).to_string(index=False))
out['lead_lift'] = lead[['lag_h', 'lift']].round(4).values.tolist()

os.makedirs(os.path.join(REPO, 'eda'), exist_ok=True)
json.dump(out, open(os.path.join(REPO, 'eda', 'airlapse_priors.json'), 'w'), indent=1)
