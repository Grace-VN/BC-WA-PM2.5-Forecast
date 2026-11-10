"""
AirLapse V3 - the Green's function moves from the encoder into the forecast.

V2 (model/airlapse_v2.py) evaluates its 2-D anisotropic Green's function once,
from the history window, and hands the result (`transported`) to mix_gate as one
more feature. The 24-step decoder never sees physics again.

V3 keeps V2's encoder unchanged (per-station GRU, temporal graph attention,
physics-guided spatial mixing, mix gate, latent head) and replaces the decoder
with a PROPAGATOR ROLLOUT in physical concentration units (ug/m3):

    a_ij(t)   = A * G_ij(dt; future wind and diffusivity at hour t)
    c_phys    = (1 - sum_j a_ij) c_i + sum_j a_ij c_j          transport (convex mix)
    c_phys    = b_i + (c_phys - b_i) * exp(-k_i(t) dt)         relaxation to background
    h_{t+1}   = GRUCell([future weather_t, z, c_phys], h_t)
    c_{t+1}   = c_phys + r(h_{t+1})                             neural correction
    and c_{t+1} is carried into the next step.

Why a propagator and not an ODE right-hand side: G is the SOLUTION of the
advection-diffusion equation over an elapsed time (G(dt) = exp(L dt)), not its
rate of change. Putting G inside dz/dt (V2's optional ContinuousTransportODEFunc)
integrates time twice and only works with an artificial t_eps; with t_eps =
0.25 h and stations ~26 km apart its conductance is ~1e-7 per hour, so that
branch barely couples stations. Applying G(dt = 1 h) once per forecast hour is
the exact solution for wind held constant over the hour, uses the actual hourly
future wind (non-autonomous), and needs no ODE solver.

Terms:
  A        learned effective area per station (km2): converts G (a density per
           km2) into the fraction of a source's anomaly that reaches a receiver.
           Initialised to the squared median nearest-neighbour distance.
  a_ij     rescaled per receiver so sum_j a_ij <= max_inflow (< 1): the update is
           then a convex combination of station values - stable and
           non-negative for any learned D, A or wind.
  b_i, k   per-station background concentration (init: training mean) and a
           weather-dependent relaxation rate k = softplus(MLP(future weather)),
           initialised to 1/tau with tau = 11.6 h (EDA e-folding time of PM2.5
           anomalies). Covers deposition, washout, mixing-height dilution and
           unresolved sources that pure station-to-station transport cannot.
  r        residual head, initialised near zero, so training starts from the
           physics forecast and learns corrections to it.

physics_only=True drops the encoder/GRU and the residual: an interpretable
physics baseline (transport + relaxation, a few hundred parameters) for the
ablation table.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.airlapse import haversine_km, bearing_deg, _inv_softplus
from model.airlapse_v2 import AirLapseV2


class GreenPropagator(nn.Module):
    """One forecast step of station-to-station advection-diffusion transport
    plus relaxation to a background, in ug/m3."""

    def __init__(self, station_coords, station_elevation, feature_dim, pm25_mean,
                 dist_threshold_km=375.0, dt_hours=1.0, diff_hidden_dim=16,
                 diffusivity_along_init=50.0, diffusivity_cross_init=50.0,
                 area_init_km2=None, relax_tau_init_h=11.6, max_inflow=0.9):
        super().__init__()
        self.dt = float(dt_hours)
        self.max_inflow = max_inflow

        dist = haversine_km(station_coords)
        N = dist.shape[0]
        not_self = ~torch.eye(N, dtype=torch.bool)
        self.register_buffer('mask', ((dist <= dist_threshold_km) & not_self).float())
        self.register_buffer('dist_km', dist)
        # [receiver, source]: bearing from the source to the receiver (as V2)
        self.register_buffer('bearing_j_i', bearing_deg(station_coords).t())
        self.register_buffer('elev_km', station_elevation.float() / 1000.0)

        if area_init_km2 is None:
            nn_km = dist.masked_fill(~not_self, float('inf')).min(dim=1).values.median().item()
            area_init_km2 = nn_km ** 2
        self.log_area = nn.Parameter(torch.tensor(math.log(area_init_km2)))

        # diffusivity from the SOURCE station's context, same inputs as V2
        self.diff_mlp = nn.Sequential(nn.Linear(5, diff_hidden_dim), nn.Tanh(),
                                      nn.Linear(diff_hidden_dim, 2))
        with torch.no_grad():
            self.diff_mlp[-1].weight.mul_(0.01)
            self.diff_mlp[-1].bias[0] = _inv_softplus(diffusivity_along_init)
            self.diff_mlp[-1].bias[1] = _inv_softplus(diffusivity_cross_init)

        # relaxation rate from the future weather (all feature channels)
        self.relax_mlp = nn.Sequential(nn.Linear(feature_dim, diff_hidden_dim), nn.Tanh(),
                                       nn.Linear(diff_hidden_dim, 1))
        with torch.no_grad():
            self.relax_mlp[-1].weight.mul_(0.01)
            self.relax_mlp[-1].bias.fill_(_inv_softplus(1.0 / relax_tau_init_h))
        self.raw_background = nn.Parameter(
            torch.full((N,), _inv_softplus(max(float(pm25_mean), 0.1))))

    def weights(self, bearing, speed, hour, c_norm):
        """a_ij for one step. bearing (rad, travel-toward), speed (km/h), hour, c_norm:
        [B, N] per source station. Returns [B, N_recv, N_src]."""
        B, N = bearing.shape
        ang = bearing.unsqueeze(1) - self.bearing_j_i.unsqueeze(0)       # [B, recv, src]
        d = self.dist_km.unsqueeze(0)
        x_down, y_cross = d * torch.cos(ang), d * torch.sin(ang)

        h = hour * (2.0 * math.pi / 24.0)
        ctx = torch.stack([speed / 10.0, torch.sin(h), torch.cos(h),
                           self.elev_km.expand(B, N), c_norm], dim=-1)  # [B, src, 5]
        D = F.softplus(self.diff_mlp(ctx)).clamp(2.0, 2000.0)           # km2/h
        D_al, D_cr = D[..., 0].unsqueeze(1), D[..., 1].unsqueeze(1)      # [B, 1, src]

        t = self.dt
        v = speed.unsqueeze(1)
        expo = (x_down - v * t) ** 2 / (4 * D_al * t) + y_cross ** 2 / (4 * D_cr * t)
        G = torch.exp(-expo) / (4 * math.pi * t * torch.sqrt(D_al * D_cr))   # per km2
        a = self.log_area.exp() * G * self.mask
        inflow = a.sum(-1, keepdim=True)
        return a * torch.clamp(self.max_inflow / (inflow + 1e-8), max=1.0)

    def forward(self, c, bearing, speed, hour, feat, c_norm):
        """c: [B, N] ug/m3 -> c after one step of transport + relaxation."""
        a = self.weights(bearing, speed, hour, c_norm)
        c = (1.0 - a.sum(-1)) * c + torch.bmm(a, c.unsqueeze(-1)).squeeze(-1)
        k = F.softplus(self.relax_mlp(feat)).squeeze(-1)                 # [B, N], 1/h
        b = F.softplus(self.raw_background)
        return b + (c - b) * torch.exp(-k * self.dt)


class AirLapseV3(AirLapseV2):
    """V2 encoder + Green's-function propagator decoder (see module docstring).
    Extra arguments: pm25_mean/pm25_std (the dataset's PM2.5 normalisation),
    physics_only, area_init_km2, relax_tau_init_h, max_inflow,
    prop_diffusivity_along_init/cross_init."""

    def __init__(self, *args, pm25_mean, pm25_std, physics_only=False,
                 area_init_km2=None, relax_tau_init_h=11.6, max_inflow=0.9,
                 prop_diffusivity_along_init=50.0, prop_diffusivity_cross_init=50.0, **kw):
        kw['ode_transport'] = False
        super().__init__(*args, **kw)
        self.physics_only = physics_only
        self.pm25_mean = float(pm25_mean)
        self.pm25_std = float(pm25_std)
        station_coords = kw['station_coords']
        self.propagator = GreenPropagator(
            station_coords=torch.as_tensor(station_coords, dtype=torch.float32),
            station_elevation=torch.as_tensor(kw['station_elevation'], dtype=torch.float32),
            feature_dim=self.feature_dim, pm25_mean=self.pm25_mean,
            dist_threshold_km=kw.get('dist_threshold_km', 375.0), dt_hours=self.dt_hours,
            diff_hidden_dim=kw.get('diff_hidden_dim', 16),
            diffusivity_along_init=prop_diffusivity_along_init,
            diffusivity_cross_init=prop_diffusivity_cross_init,
            area_init_km2=area_init_km2, relax_tau_init_h=relax_tau_init_h,
            max_inflow=max_inflow)
        # the decoder also sees the physics forecast for its station
        self.decoder_cell = nn.GRUCell(self.feature_dim + self.latent_dim + 1, self.hidden_dim)
        with torch.no_grad():                    # start from the physics forecast
            self.output_head.weight.mul_(0.01)
            self.output_head.bias.zero_()

    def forward(self, pm25_hist, feature):
        H, P = self.hist_len, self.pred_len
        feature_hist = feature[:, :H]
        feature_future = feature[:, H:H + P]
        B, _, N, _ = pm25_hist.shape
        if N != self.city_num:
            raise ValueError(f"AirLapseV3 was built with city_num={self.city_num}, got N={N}")
        m, s = self.pm25_mean, self.pm25_std

        bearing, speed, hour = self._wind_at_idxs(feature_future, list(range(P)), B, N)  # [B, P, N]
        c = (pm25_hist[:, -1, :, 0] * s + m).clamp(min=0.0)                  # [B, N] ug/m3

        if not self.physics_only:
            inputs = torch.cat([pm25_hist, feature_hist], dim=-1)
            x = inputs.permute(0, 2, 1, 3).reshape(B * N, H, -1)
            if self.spatial_mix_mode == 'bottleneck':
                h_mixed = self._encode_bottleneck(x, feature_hist, B, N)
            else:
                h_mixed = self._encode_per_step(x, feature_hist, B, N)
            z = self.latent_head(h_mixed)
            h = self.decoder_init(torch.cat([h_mixed, z], dim=-1))
            feat_fut = feature_future.permute(0, 2, 1, 3).reshape(B * N, P, self.feature_dim)

        preds = []
        for t in range(P):
            c_phys = self.propagator(c, bearing[:, t], speed[:, t], hour[:, t],
                                     feature_future[:, t], (c - m) / s)
            if self.physics_only:
                c_next = c_phys
            else:
                phys_in = ((c_phys - m) / s).reshape(B * N, 1)
                h = self.decoder_cell(torch.cat([feat_fut[:, t], z, phys_in], dim=-1), h)
                c_next = c_phys + s * self.output_head(h).reshape(B, N)
            preds.append((c_next - m) / s)
            c = c_next.clamp(min=0.0)            # carry the corrected field forward

        return torch.stack(preds, dim=1).unsqueeze(-1)                     # [B, P, N, 1]
