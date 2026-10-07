"""Crossformer baseline (Zhang & Yan, ICLR 2023, "Crossformer: Transformer
Utilizing Cross-Dimension Dependency for Multivariate Time Series
Forecasting"), ported from the official code
(github.com/Thinklab-SJTU/Crossformer, Apache-2.0) without the einops
dependency.

Kept from the reference: Dimension-Segment-Wise (DSW) embedding with learned
position embeddings; the hierarchical encoder - one scale block per level,
SegMerging (win_size adjacent segments -> one) before every level but the
first, each followed by a Two-Stage Attention (TSA) layer (cross-time
attention, then cross-dimension attention through a small learned router);
and the decoder with e_layers + 1 layers, each a TSA layer on learned future
segment queries plus cross-attention to the encoder output of its scale,
whose per-scale segment predictions are summed.

Adapted to this benchmark: the Crossformer "dimensions" are the stations, so
its cross-dimension stage models cross-station dependency. A DSW segment
embeds all of a station's input variables over seg_len hours (PM2.5 +
weather, Linear(seg_len * in_dim -> d_model)) instead of a single variable;
only PM2.5 is predicted. Like the reference, the decoder sees no future
covariates (future weather unused). Defaults are scaled down for ~115
stations and 24 h in / 24 h out: seg_len 6 (4 input and 4 output segments),
win_size 2, e_layers 3 (4 -> 2 -> 1 segments), router factor 5, d_model 64.
"""
import math

import torch
from torch import nn


class AttentionLayer(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.h = n_heads
        self.q, self.k, self.v = (nn.Linear(d_model, d_model) for _ in range(3))
        self.o = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, q, k, v):                                  # [B, L, d], [B, S, d]
        B, L, d = q.shape
        S = k.shape[1]
        q = self.q(q).view(B, L, self.h, -1).transpose(1, 2)
        k = self.k(k).view(B, S, self.h, -1).transpose(1, 2)
        v = self.v(v).view(B, S, self.h, -1).transpose(1, 2)
        a = self.drop(torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1]), dim=-1))
        return self.o((a @ v).transpose(1, 2).reshape(B, L, d))


class TwoStageAttention(nn.Module):
    """[B, D, L, d] -> [B, D, L, d]: cross-time, then cross-dimension via a router."""

    def __init__(self, seg_num, factor, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.time_attn = AttentionLayer(d_model, n_heads, dropout)
        self.dim_sender = AttentionLayer(d_model, n_heads, dropout)
        self.dim_receiver = AttentionLayer(d_model, n_heads, dropout)
        self.router = nn.Parameter(torch.randn(seg_num, factor, d_model))
        self.drop = nn.Dropout(dropout)
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(4)])
        self.mlp1 = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))
        self.mlp2 = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))

    def forward(self, x):
        B, D, L, d = x.shape
        t = x.reshape(B * D, L, d)
        t = self.norms[0](t + self.drop(self.time_attn(t, t, t)))
        t = self.norms[1](t + self.drop(self.mlp1(t)))
        s = t.reshape(B, D, L, d).transpose(1, 2).reshape(B * L, D, d)          # (b seg) dim d
        router = self.router.repeat(B, 1, 1)                                      # (b seg) factor d
        buf = self.dim_sender(router, s, s)
        s = self.norms[2](s + self.drop(self.dim_receiver(s, buf, buf)))
        s = self.norms[3](s + self.drop(self.mlp2(s)))
        return s.reshape(B, L, D, d).transpose(1, 2)


class SegMerging(nn.Module):
    def __init__(self, d_model, win):
        super().__init__()
        self.win = win
        self.norm = nn.LayerNorm(win * d_model)
        self.lin = nn.Linear(win * d_model, d_model)

    def forward(self, x):                                        # [B, D, L, d]
        pad = (-x.shape[2]) % self.win
        if pad:
            x = torch.cat([x, x[:, :, -pad:]], dim=2)
        x = torch.cat([x[:, :, i::self.win] for i in range(self.win)], dim=-1)
        return self.lin(self.norm(x))


class DecoderLayer(nn.Module):
    def __init__(self, seg_len, d_model, n_heads, d_ff, dropout, out_seg_num, factor):
        super().__init__()
        self.self_attn = TwoStageAttention(out_seg_num, factor, d_model, n_heads, d_ff, dropout)
        self.cross_attn = AttentionLayer(d_model, n_heads, dropout)
        self.norm1, self.norm2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        self.mlp = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.pred = nn.Linear(d_model, seg_len)

    def forward(self, x, cross):
        B, D, L, d = x.shape
        x = self.self_attn(x).reshape(B * D, L, d)
        c = cross.reshape(B * D, cross.shape[2], d)
        x = self.norm1(x + self.drop(self.cross_attn(x, c, c)))
        x = self.norm2(x + self.mlp(x)).reshape(B, D, L, d)
        return x, self.pred(x)                                   # [B, D, L, seg_len]


class CrossformerPM25(nn.Module):
    def __init__(self, hist_len, pred_len, in_dim, city_num, batch_size, device,
                 seg_len=6, win_size=2, factor=5, d_model=64, d_ff=128, n_heads=4,
                 e_layers=3, dropout=0.2):
        super().__init__()
        self.hist_len, self.pred_len, self.seg_len = hist_len, pred_len, seg_len
        self.in_seg = math.ceil(hist_len / seg_len)
        self.out_seg = math.ceil(pred_len / seg_len)
        self.pad_in = self.in_seg * seg_len - hist_len
        self.embed = nn.Linear(seg_len * in_dim, d_model)
        self.enc_pos = nn.Parameter(torch.randn(1, city_num, self.in_seg, d_model))
        self.pre_norm = nn.LayerNorm(d_model)
        self.merges = nn.ModuleList()
        self.enc_tsa = nn.ModuleList()
        for i in range(e_layers):
            self.merges.append(SegMerging(d_model, win_size) if i > 0 else nn.Identity())
            self.enc_tsa.append(TwoStageAttention(math.ceil(self.in_seg / win_size ** i), factor,
                                                  d_model, n_heads, d_ff, dropout))
        self.dec_pos = nn.Parameter(torch.randn(1, city_num, self.out_seg, d_model))
        self.dec_layers = nn.ModuleList([
            DecoderLayer(seg_len, d_model, n_heads, d_ff, dropout, self.out_seg, factor)
            for _ in range(e_layers + 1)])

    def forward(self, pm25_hist, feature):
        B, H, N, _ = pm25_hist.shape
        x = torch.cat([pm25_hist, feature[:, :H]], dim=-1)                 # [B, H, N, C]
        if self.pad_in:                                                     # reference: repeat first step
            x = torch.cat([x[:, :1].expand(-1, self.pad_in, -1, -1), x], dim=1)
        x = x.permute(0, 2, 1, 3).reshape(B, N, self.in_seg, -1)           # [B, N, seg, seg_len*C]
        x = self.pre_norm(self.embed(x) + self.enc_pos)
        enc = [x]
        for merge, tsa in zip(self.merges, self.enc_tsa):
            x = tsa(merge(x))
            enc.append(x)
        dec = self.dec_pos.expand(B, -1, -1, -1)
        pred = 0
        for layer, cross in zip(self.dec_layers, enc):
            dec, p = layer(dec, cross)
            pred = pred + p                                                 # [B, N, out_seg, seg_len]
        pred = pred.reshape(B, N, -1)[:, :, :self.pred_len]
        return pred.permute(0, 2, 1).unsqueeze(-1)                          # [B, P, N, 1]
