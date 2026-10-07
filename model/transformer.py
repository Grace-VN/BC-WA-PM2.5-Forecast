"""Vanilla encoder-decoder Transformer baseline (Vaswani et al., NeurIPS 2017).

Each station is an independent sequence (stations folded into the batch), as
in the other per-station baselines here. The encoder reads the history -
PM2.5 + weather for hist_len hours; the decoder gets one query per forecast
hour built from that hour's (known) weather and attends to the encoder. All
pred_len hours are decoded in one pass with no causal mask (a direct,
non-autoregressive multi-step decoder, as in Informer's generative-style
decoder), using torch.nn.Transformer with sinusoidal positions.
"""
import math

import torch
from torch import nn


class TransformerPM25(nn.Module):
    def __init__(self, hist_len, pred_len, in_dim, city_num, batch_size, device,
                 d_model=64, n_heads=4, e_layers=2, d_layers=1, d_ff=256, dropout=0.1):
        super().__init__()
        self.hist_len, self.pred_len = hist_len, pred_len
        self.enc_in = nn.Linear(in_dim, d_model)          # PM2.5 + weather
        self.dec_in = nn.Linear(in_dim - 1, d_model)      # weather only (PM2.5 unknown)
        L = hist_len + pred_len
        pos = torch.arange(L).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(L, d_model)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)
        self.transformer = nn.Transformer(d_model, n_heads, e_layers, d_layers, d_ff, dropout,
                                          batch_first=True)
        self.out = nn.Linear(d_model, 1)

    def forward(self, pm25_hist, feature):
        B, H, N, _ = pm25_hist.shape
        P = self.pred_len
        x = torch.cat([pm25_hist, feature[:, :H]], dim=-1).permute(0, 2, 1, 3).reshape(B * N, H, -1)
        f = feature[:, H:H + P].permute(0, 2, 1, 3).reshape(B * N, P, -1)
        y = self.transformer(self.enc_in(x) + self.pe[:H], self.dec_in(f) + self.pe[H:H + P])
        return self.out(y).reshape(B, N, P, 1).permute(0, 2, 1, 3)
