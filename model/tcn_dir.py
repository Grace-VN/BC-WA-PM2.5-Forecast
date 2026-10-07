"""TCN-DIR baseline: multi-scale temporal CNN trained with deep imbalanced
regression, re-implemented from Seo et al. (2026), "Adaptive expert-guided
deep imbalanced regression for global PM2.5 forecasting with temporal
convolutional networks" (reference code: github.com/junhseo/global-pm25-dir-resmoe,
NonCasualImprovedFDSEnhancedTemporalCNN + weighted_mse with LDS).

Kept from the reference:
  - three non-causal multi-scale dilated Conv1d blocks (32/64/128 channels per
    branch, dilations [1,2,4] / [2,4,8] / [4,8,16], BatchNorm, LeakyReLU,
    dropout 0.2, 1x1 residual), temporal self-attention over the 384 channels,
    global average pooling, then a 384 -> 64 -> pred_len head;
  - Label Distribution Smoothing (Yang et al., ICML 2021) loss weights: a
    histogram of training PM2.5 in 1 ug/m3 bins (0..num_bins) smoothed with a
    Gaussian (sigma), each sample weighted by 1/sqrt(smoothed density at its
    mean target), scaled so the largest training weight is 1. Applied to the
    training loss only, through train.py's `weighted_loss` hook.
Changed for this benchmark:
  - no GEOS-FP forecast inputs and no residual mixture-of-experts stage (that
    stage corrects a chemical-transport forecast this dataset doesn't have);
    the input sequence is the hist_len history (PM2.5 + weather) followed by
    the pred_len forecast hours (weather, PM2.5 slot zeroed, plus a 0/1
    "future" flag), one sequence per station;
  - the reference's FDS layers are left out: as released their running
    statistics are never updated, so they pass features through unchanged;
  - trained with this repo's shared optimiser/schedule (not AdamW + StepLR).
"""
import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d
from torch import nn


class MultiScaleConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dilations, dropout=0.2):
        super().__init__()
        self.convs = nn.ModuleList([nn.Conv1d(in_ch, out_ch, 3, padding="same", dilation=d)
                                    for d in dilations])
        total = out_ch * len(dilations)
        self.bn = nn.BatchNorm1d(total)
        self.act = nn.LeakyReLU()
        self.drop = nn.Dropout(dropout)
        self.res = nn.Conv1d(in_ch, total, 1) if in_ch != total else nn.Identity()

    def forward(self, x):
        y = torch.cat([c(x) for c in self.convs], dim=1)
        return self.drop(self.act(self.bn(y))) + self.res(x)


class TemporalSelfAttention(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.q, self.k, self.v = nn.Linear(ch, ch), nn.Linear(ch, ch), nn.Linear(ch, ch)

    def forward(self, x):                                   # [B, C, T]
        x = x.permute(0, 2, 1)
        a = torch.softmax(self.q(x) @ self.k(x).transpose(1, 2) / x.shape[-1] ** 0.5, dim=-1)
        return (a @ self.v(x)).permute(0, 2, 1)


class TCNDIR(nn.Module):
    def __init__(self, hist_len, pred_len, in_dim, city_num, batch_size, device,
                 train_pm25=None, pm25_mean=0.0, pm25_std=1.0, lds=True,
                 num_bins=501, sigma=1.0, dropout=0.2):
        super().__init__()
        self.hist_len, self.pred_len = hist_len, pred_len
        self.pm25_mean, self.pm25_std = float(pm25_mean), float(pm25_std)
        ch = in_dim + 1                                     # + future flag
        self.convs = nn.Sequential(
            MultiScaleConvBlock(ch, 32, [1, 2, 4], dropout),
            MultiScaleConvBlock(96, 64, [2, 4, 8], dropout),
            MultiScaleConvBlock(192, 128, [4, 8, 16], dropout),
        )
        self.attn = TemporalSelfAttention(384)
        self.head = nn.Sequential(nn.Linear(384, 64), nn.LeakyReLU(), nn.Dropout(0.3),
                                  nn.Linear(64, pred_len))
        self.lds = lds and train_pm25 is not None
        if self.lds:
            # train_pm25: normalised training windows [S, hist+pred, N, 1]
            raw = np.asarray(train_pm25)[..., 0] * self.pm25_std + self.pm25_mean
            values = np.concatenate([raw[:, 0].ravel(), raw[-1, 1:].ravel()])   # each hour once
            hist, _ = np.histogram(values, bins=np.arange(0, num_bins + 1))
            density = np.maximum(1e-6, gaussian_filter1d(hist.astype(float), sigma=sigma))
            sample_mean = raw[:, hist_len:].mean(axis=1).ravel()
            idx = np.clip(sample_mean.astype(int), 0, num_bins - 1)
            w_max = (1.0 / np.sqrt(density[idx])).max()
            self.register_buffer("lds_weight", torch.tensor(1.0 / np.sqrt(density) / w_max,
                                                            dtype=torch.float32))

    def forward(self, pm25_hist, feature):
        B, H, N, _ = pm25_hist.shape
        P = self.pred_len
        pm = torch.cat([pm25_hist, pm25_hist.new_zeros(B, P, N, 1)], dim=1)
        flag = torch.cat([pm25_hist.new_zeros(B, H, N, 1), pm25_hist.new_ones(B, P, N, 1)], dim=1)
        x = torch.cat([pm, feature[:, :H + P], flag], dim=-1)            # [B, H+P, N, C]
        x = x.permute(0, 2, 3, 1).reshape(B * N, -1, H + P)               # [B*N, C, T]
        x = self.attn(self.convs(x)).mean(dim=-1)                          # [B*N, 384]
        return self.head(x).reshape(B, N, P, 1).permute(0, 2, 1, 3)

    def weighted_loss(self, pred, label):
        """LDS-weighted MSE (training only); pred/label normalised [B, P, N, 1]."""
        if not self.lds:
            return torch.mean((pred - label) ** 2)
        target = (label[..., 0] * self.pm25_std + self.pm25_mean).mean(dim=1)   # [B, N]
        idx = target.long().clamp(0, self.lds_weight.numel() - 1)
        w = self.lds_weight[idx]                                                 # [B, N]
        return torch.mean((pred - label) ** 2 * w[:, None, :, None])
