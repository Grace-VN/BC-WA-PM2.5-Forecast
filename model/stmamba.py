"""STMamba baseline: correlated-station fusion + Mamba, re-implemented from
Zhang et al. (2025), "WOA-STMamba: A spatiotemporal Mamba model enhanced by
Whale Optimization for short-term PM2.5 forecasting" (reference code:
github.com/Bubbles929/WOA-STMamba, class oMamba).

Kept from the reference:
  - for each target station, the K training-period most-correlated stations
    (Pearson on PM2.5, the station itself first) are stacked as channels of a
    [K, time, features] tensor and fused by a 1x1 Conv2d stack
    (K -> hidden -> 1, BatchNorm + ReLU after each), giving one fused
    [time, features] sequence;
  - a Mamba selective state-space layer (Gu & Dao, 2023) over the history
    (d_state 16, d_conv 4, expand 2), read out at the last step;
  - history only: like the reference it does not use future weather.
Changed for this benchmark:
  - direct 24-hour output (Linear to pred_len) instead of one step ahead;
  - one model shared by all stations (the reference trains one per target
    station), with the correlation-ranked neighbour order shared by all;
  - features are projected to d_model before the Mamba layer (the reference
    runs Mamba directly on its 12 raw features);
  - fixed hyperparameters - no Whale Optimization search;
  - Mamba is a pure-PyTorch implementation of the "Mamba simple" block, so it
    runs on CPU and any GPU without the CUDA-only mamba-ssm package (same
    maths, sequential scan - fine for 24-step sequences).
"""
import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        d_inner = expand * d_model
        self.dt_rank = math.ceil(d_model / 16)
        self.d_state = d_state
        self.in_proj = nn.Linear(d_model, 2 * d_inner)
        self.conv = nn.Conv1d(d_inner, d_inner, d_conv, groups=d_inner, padding=d_conv - 1)
        self.x_proj = nn.Linear(d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, d_inner)
        dt = torch.exp(torch.rand(d_inner) * (math.log(0.1) - math.log(1e-3)) + math.log(1e-3))
        with torch.no_grad():
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))      # inverse softplus
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1).float()).repeat(d_inner, 1))
        self.D = nn.Parameter(torch.ones(d_inner))
        self.out_proj = nn.Linear(d_inner, d_model)

    def forward(self, x):                                          # [B, L, d_model]
        L = x.shape[1]
        x, z = self.in_proj(x).chunk(2, dim=-1)
        x = F.silu(self.conv(x.transpose(1, 2))[..., :L].transpose(1, 2))
        dt, Bm, Cm = self.x_proj(x).split([self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt))                          # [B, L, d_inner]
        A = -torch.exp(self.A_log)                                 # [d_inner, d_state]
        h = x.new_zeros(x.shape[0], x.shape[2], self.d_state)
        ys = []
        for t in range(L):
            h = torch.exp(dt[:, t, :, None] * A) * h + dt[:, t, :, None] * Bm[:, t, None, :] * x[:, t, :, None]
            ys.append((h * Cm[:, t, None, :]).sum(-1))
        y = torch.stack(ys, dim=1) + x * self.D
        return self.out_proj(y * F.silu(z))


class STMambaPM25(nn.Module):
    def __init__(self, hist_len, pred_len, in_dim, city_num, batch_size, device,
                 train_pm25=None, k_stations=8, fusion_hidden=64, d_model=32,
                 d_state=16, d_conv=4, expand=2, n_layers=1):
        super().__init__()
        self.hist_len, self.pred_len = hist_len, pred_len
        k = min(k_stations, city_num)
        if train_pm25 is not None:
            series = np.asarray(train_pm25)[:, 0, :, 0]                         # [hours, N]
            corr = np.nan_to_num(np.corrcoef(series.T), nan=-1.0)
            np.fill_diagonal(corr, np.inf)                                      # self first
            nbr = np.argsort(-corr, axis=1)[:, :k]
        else:
            nbr = np.tile(np.arange(city_num)[:, None], (1, k))
        self.register_buffer("nbr", torch.tensor(nbr, dtype=torch.long))       # [N, K]
        self.fuse = nn.Sequential(
            nn.Conv2d(k, fusion_hidden, 1), nn.BatchNorm2d(fusion_hidden), nn.ReLU(),
            nn.Conv2d(fusion_hidden, 1, 1), nn.BatchNorm2d(1), nn.ReLU(),
        )
        self.proj = nn.Linear(in_dim, d_model)
        self.layers = nn.ModuleList([MambaBlock(d_model, d_state, d_conv, expand) for _ in range(n_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.head = nn.Linear(d_model, pred_len)

    def forward(self, pm25_hist, feature):
        B, H, N, _ = pm25_hist.shape
        x = torch.cat([pm25_hist, feature[:, :H]], dim=-1)                  # [B, H, N, F]
        g = x[:, :, self.nbr]                                                # [B, H, N, K, F]
        g = g.permute(0, 2, 3, 1, 4).reshape(B * N, self.nbr.shape[1], H, -1)
        h = self.proj(self.fuse(g)[:, 0])                                   # [B*N, H, d_model]
        for layer, norm in zip(self.layers, self.norms):
            h = h + layer(norm(h))
        return self.head(h[:, -1]).reshape(B, N, self.pred_len, 1).permute(0, 2, 1, 3)
