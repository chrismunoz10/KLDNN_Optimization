"""
Pressure and saturation surrogate reconstruction for the CO2 optimization.

Both surrogates take the same 21-d input:
    s = [ xi_xy(6), xi_z(6), xi_por(6), rate_norm(1), well_i(1), well_j(1) ]
      = [ 18 geological latents , normalized injection rate , well (i,j) ]

Saturation:  MLP predicts the plume DIRECTLY on the active subdomain
             (93 x 94 x 74 x 52), zero-padded into the full (126,125,110,52) grid.
Pressure:    MLP predicts a 20-d latent (gamma); a nested Karhunen-Loeve
             reconstruction maps gamma -> eta -> log-pressure, exp() back to Pa,
             scattered into the active (non-NaN) cells of the full grid.

This mirrors the user's reconstruction script; the vectorized scatter is
equivalent to the original per-index loop but far faster.

Requires: torch, numpy, h5py.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

try:
    import torch
    import torch.nn as nn
except Exception:                     # torch is only needed at predict time
    torch, nn = None, object

# ---- fixed grid / subdomain geometry (from the reconstruction script) ------ #
NX, NY, NZ, NT = 126, 125, 110, 52
# active saturation subdomain (end-exclusive slices): x[16:109] y[12:106] z[28:102]
SAT_X = (16, 109)      # 109-16 = 93
SAT_Y = (12, 106)      # 106-12 = 94
SAT_Z = (28, 102)      # 102-28 = 74
SAT_SHAPE = (SAT_X[1] - SAT_X[0], SAT_Y[1] - SAT_Y[0], SAT_Z[1] - SAT_Z[0], NT)


# =========================================================================== #
# 1. Network definitions  (must match the trained weights exactly)            #
# =========================================================================== #
class NN_etas(nn.Module):
    """Pressure latent net: 21 -> 20-d latent (gamma)."""
    def __init__(self):
        super().__init__()
        self.input = nn.Linear(21, 50)
        self.hidden1 = nn.Linear(50, 80)
        self.hidden2 = nn.Linear(80, 50)
        self.predict = nn.Linear(50, 20)

    def forward(self, x):
        x = torch.tanh(self.input(x))
        x = torch.tanh(self.hidden1(x))
        x = torch.tanh(self.hidden2(x))
        return self.predict(x)


class NN_Sats(nn.Module):
    """Saturation net: 21 -> full plume on the active subdomain (flattened)."""
    def __init__(self):
        super().__init__()
        self.input = nn.Linear(21, 12)
        self.hidden1 = nn.Linear(12, 24)
        self.hidden2 = nn.Linear(24, 12)
        self.hidden3 = nn.Linear(12, 20)
        self.predict = nn.Linear(20, int(np.prod(SAT_SHAPE)))   # 93*94*74*52

    def forward(self, x):
        # NOTE: input layer has NO activation in the original script; the three
        # hidden layers use tanh. Confirm this matches how the weights were
        # trained (activations are not stored in the state_dict).
        x = torch.tanh(self.input(x))
        x = torch.tanh(self.hidden1(x))
        x = torch.tanh(self.hidden2(x))
        x = torch.tanh(self.hidden3(x))
        return self.predict(x)


# =========================================================================== #
# 2. Config + one-time loading                                                #
# =========================================================================== #
@dataclass
class ReconConfig:
    pressure_weights: str = "pres/model_pres_latent.pth"
    saturation_weights: str = "sats/model_s_latent.pth"
    kl_hdf5: str = "pres/p_reconstruction.hdf5"      # A1, b1, A2, b2
    nan_locations: str = "pres/nan_locations.npy"    # (K,4) inactive-cell indices
    rate_file: str = "rate_ft3_day.npy"              # for min-max rate normalization
    device: str = "cpu"

    # filled by load()
    _pnet: object = field(default=None, repr=False)
    _snet: object = field(default=None, repr=False)
    _kl: tuple = field(default=None, repr=False)
    _active: np.ndarray = field(default=None, repr=False)     # (nx,ny,nz,nt) bool
    _active_sp: np.ndarray = field(default=None, repr=False)  # (nx,ny,nz) bool (spatial)
    q_min: float = None
    q_max: float = None


def load(cfg: ReconConfig) -> ReconConfig:
    """Load both networks, the KL matrices, the active-cell mask, and rate stats."""
    dev = torch.device(cfg.device)

    cfg._pnet = NN_etas().to(dev)
    cfg._pnet.load_state_dict(torch.load(cfg.pressure_weights, map_location=dev,
                                         weights_only=True))
    cfg._pnet.eval()

    cfg._snet = NN_Sats().to(dev)
    cfg._snet.load_state_dict(torch.load(cfg.saturation_weights, map_location=dev,
                                         weights_only=True))
    cfg._snet.eval()

    import h5py
    with h5py.File(cfg.kl_hdf5, "r") as f:
        A1, b1 = f["A1"][:], f["b1"][:]
        A2, b2 = f["A2"][:], f["b2"][:]
    cfg._kl = (A1, b1, A2, b2)

    # active-cell mask: NaN at inactive cells, active elsewhere
    nanloc = np.load(cfg.nan_locations)
    grid = np.zeros((NX, NY, NZ, NT))
    grid[tuple(np.asarray(nanloc).T)] = np.nan
    cfg._active = ~np.isnan(grid)
    cfg._active_sp = cfg._active[..., 0]     # spatial active mask (same for all t)

    Q = np.load(cfg.rate_file)
    cfg.q_min, cfg.q_max = float(Q.min()), float(Q.max())
    return cfg


# =========================================================================== #
# 3. Input assembly                                                           #
# =========================================================================== #
def assemble_input(xi18, q_phys, well_i, well_j, cfg: ReconConfig) -> np.ndarray:
    """Build the normalized 21-vector from a physical rate + raw well (i,j).
    xi18 may be (18,) or (B,18); q/i/j scalars or (B,). Returns (B,21)."""
    xi18 = np.atleast_2d(xi18).astype(np.float64)
    q = np.atleast_1d(q_phys).astype(np.float64)
    i = np.atleast_1d(well_i).astype(np.float64)
    j = np.atleast_1d(well_j).astype(np.float64)
    q_norm = (q - cfg.q_min) / (cfg.q_max - cfg.q_min)   # min-max, as in training
    B = max(xi18.shape[0], q.shape[0])
    xi18 = np.broadcast_to(xi18, (B, 18))
    cols = [xi18, q_norm.reshape(B, 1), i.reshape(B, 1), j.reshape(B, 1)]
    return np.concatenate(cols, axis=1)                  # (B,21)


# =========================================================================== #
# 4. Reconstruction kernels                                                   #
# =========================================================================== #
def _reconstruct_pressure_one(gamma, A1, b1, A2, b2):
    """Nested-KL: gamma(20) -> eta(nt,Neta) -> log P(nt,Nxyz) -> P (exp). """
    Neta = b1.shape[1]
    Ngamma = A1.shape[2]
    nt, Nxyz = b2.shape
    g = gamma.reshape(Ngamma, Neta).T                    # (Neta, Ngamma)
    eta = np.empty((nt, Neta))
    for jj in range(Neta):
        eta[:, jj] = b1[:, jj] + A1[jj] @ g[jj]          # (nt,Ngamma)@(Ngamma,)->(nt,)
    p = np.empty((nt, Nxyz))
    for ii in range(nt):
        p[ii] = b2[ii] + A2[ii] @ eta[ii]                # (Nxyz,Neta)@(Neta,)->(Nxyz,)
    return np.exp(p)                                     # log-pressure -> Pa, (nt,Nxyz)


def _scatter_pressure(p_nt_nxyz, active):
    """Place active-cell pressures into the full (nx,ny,nz,nt) grid; NaN elsewhere.
    Vectorized equivalent of the original per-index loop (C-order alignment)."""
    out = np.full((NX, NY, NZ, NT), np.nan)
    # active cells enumerated C-order over (nx,ny,nz,nt); values are spatial-major,
    # time-minor -> transpose (nt,Nxyz)->(Nxyz,nt) then ravel to match.
    out[active] = np.transpose(p_nt_nxyz, (1, 0)).ravel()
    return out                                           # (nx,ny,nz,nt), Pa


def predict_pressure(X21: np.ndarray, cfg: ReconConfig) -> np.ndarray:
    """Batched pressure field. X21:(B,21) -> P:(B,nx,ny,nz,nt) Pa, NaN inactive."""
    A1, b1, A2, b2 = cfg._kl
    with torch.no_grad():
        xt = torch.as_tensor(np.atleast_2d(X21), dtype=torch.float32,
                             device=cfg.device)
        gamma = cfg._pnet(xt).cpu().numpy()              # (B,20)
    out = np.empty((gamma.shape[0], NX, NY, NZ, NT))
    for k, g in enumerate(gamma):
        out[k] = _scatter_pressure(_reconstruct_pressure_one(g, A1, b1, A2, b2),
                                   cfg._active)
    return out


def predict_pressure_timesteps(X21, cfg, t_indices):
    """Reconstruct pressure ONLY at the requested timesteps (fast path for the AoR,
    which needs just t0 and t_end). X21:(B,21), t_indices e.g. (1, 37)
    -> P:(B, nx, ny, nz, len(t_indices)) Pa, NaN inactive.

    ~26x less work and memory than the full 52-timestep field: the nested-KL
    reconstruction is evaluated only at the selected timesteps.
    """
    A1, b1, A2, b2 = cfg._kl
    Neta = b1.shape[1]
    Ngamma = A1.shape[2]
    nt = b2.shape[0]
    tt = [t % nt for t in t_indices]                     # normalize negatives
    with torch.no_grad():
        xt = torch.as_tensor(np.atleast_2d(X21), dtype=torch.float32,
                             device=cfg.device)
        gamma = cfg._pnet(xt).cpu().numpy()              # (B,20)
    B = gamma.shape[0]
    out = np.full((B, NX, NY, NZ, len(tt)), np.nan)
    for k, g in enumerate(gamma):
        gg = g.reshape(Ngamma, Neta).T                   # (Neta, Ngamma)
        eta = np.empty((len(tt), Neta))
        for j in range(Neta):                            # eta only at selected t
            eta[:, j] = b1[tt, j] + A1[j][tt] @ gg[j]    # (len,Ngamma)@(Ngamma,)->(len,)
        for m, ti in enumerate(tt):
            p_t = np.exp(b2[ti] + A2[ti] @ eta[m])       # (Nxyz,) Pa at timestep ti
            full = np.full((NX, NY, NZ), np.nan)
            full[cfg._active_sp] = p_t
            out[k, ..., m] = full
    return out


def predict_saturation(X21: np.ndarray, cfg: ReconConfig) -> np.ndarray:
    """Batched saturation field. X21:(B,21) -> Sg:(B,nx,ny,nz,nt), zero outside subdomain."""
    with torch.no_grad():
        xt = torch.as_tensor(np.atleast_2d(X21), dtype=torch.float32,
                             device=cfg.device)
        flat = cfg._snet(xt).cpu().numpy()               # (B, 93*94*74*52)
    B = flat.shape[0]
    out = np.zeros((B, NX, NY, NZ, NT))
    sub = flat.reshape((B,) + SAT_SHAPE)
    out[:, SAT_X[0]:SAT_X[1], SAT_Y[0]:SAT_Y[1], SAT_Z[0]:SAT_Z[1], :] = sub
    return out