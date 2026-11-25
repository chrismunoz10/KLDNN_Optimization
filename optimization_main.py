"""
Surrogate-based multiobjective optimization of CO2 injection (IBDP, real surrogates).

Objectives
----------
    f1 = -M_CO2        (maximize injected CO2 mass; analytic in the rate)
    f2 =  A_oR         (minimize pressure-isobar Area of Review; dp_crit = 25 psi)

Decision variables (3)
----------------------
    u = [q, i, j]      injection rate (physical), well column indices (i, j)

Fixed context (18)
------------------
    xi in R^18         geological latents [xi_xy(6), xi_z(6), xi_por(6)] for the site
                       (single realization, or an ensemble in robust mode).

Pipeline per candidate:
    u -> assemble 21-input (normalized rate + raw i,j) -> pressure surrogate
      -> nested-KL reconstruction -> P(nx,ny,nz,nt) Pa -> AoR.
Both surrogates live in surrogate_reconstruction.py (real nets + reconstruction).

Algorithm: NSGA-II (pymoo).  ->  pip install pymoo numpy torch h5py
"""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

import surrogate_reconstruction as recon   # real pressure/saturation reconstruction

from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.lhs import LHS
from pymoo.termination.robust import RobustTermination
from pymoo.termination.ftol import MultiObjectiveSpaceTermination
from pymoo.core.callback import Callback
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.optimize import minimize


class _CollectF(Callback):
    """Record only the objective values each generation (tiny), for the solution-space
    plot. Avoids save_history=True, which deep-copies the whole algorithm -- including
    the multi-GB KL matrices and torch models -- every generation and OOMs."""
    def __init__(self):
        super().__init__()
        self.gens = []
 
    def notify(self, algorithm):
        self.gens.append(algorithm.pop.get("F").copy())
        

# --------------------------------------------------------------------------- #
# 1. Footprint geometry (lateral dx*dy; robust to vertical irregularity)       #
# --------------------------------------------------------------------------- #
def lateral_footprint(xyz_grid, z_axis=2):
    """(nx,ny) map area per column from top-layer x,y centers. Never uses thickness."""
    top = np.take(xyz_grid, 0, axis=z_axis)
    dx = np.abs(np.gradient(top[..., 0], axis=0))
    dy = np.abs(np.gradient(top[..., 1], axis=1))
    return dx * dy


# --------------------------------------------------------------------------- #
# 2. Configuration                                                            #
# --------------------------------------------------------------------------- #
@dataclass
class Config:
    recon: recon.ReconConfig = field(default_factory=recon.ReconConfig)

    # --- geology (fixed context) ---
    xi:          np.ndarray = None      # (18,) single-site latent (deterministic)
    xi_ensemble: np.ndarray = None      # (M,18) ensemble latents (robust mode)

    # --- Area of Review ---
    xyz_grid_path: str = "xyz_grid.npy"  # (nx,ny,nz,3) -> lateral footprint
    z_axis: int = 2
    pa_to_psi: float = 0.000145037738
    dP_crit_psi: float = 75.0           # from the end-of-injection batch sweep (band 50-100)
    # CO2 injection window: starts at index 1 (timestep 2), ends at index 37 (36 months).
    # AoR buildup = P[end-of-injection] - P[injection-start]; end-of-injection (37) is
    # the pressure PEAK -- do NOT use the last timestep (dissipated).
    t0_idx: int = 1                     # injection start (baseline ~ initial pressure)
    t_end_idx: int = 37                # end of injection (pressure peak)

    # --- injected mass (analytic): M = rho_ref * q * T ---
    T_days: float = 36 * 30.4375        # 36-month injection window, in days
    rho_ref: float = 1.872 * (0.3048)**3 * 1e-9               # rate(ft^3/day)->mass factor; set to your density -> MMTon / ft^3

    # --- decision bounds: [q, i, j] ---
    q_bounds: tuple = None              # default set from training rate range in build()
    i_bounds: tuple = (41.0, 81.0)      # keep well inside the active subdomain
    j_bounds: tuple = (37.0, 76.0)

    # --- robust reduction over the geological ensemble ---
    robust: bool = False
    reduce_mode: str = "expected"       # "expected" | "cvar" | "worst" | "chance"
    alpha: float = 0.10
    aor_cap: float = None               # for "chance": P(AoR>cap) <= alpha

    # --- optional containment constraint (uses saturation surrogate) ---
    use_containment: bool = True
    sat_boundary_tol: float = 1e-2

    footprint2d: np.ndarray = field(default=None, repr=False)

    def build(self):
        """Load surrogates + geometry once."""
        recon.load(self.recon)
        self.footprint2d = lateral_footprint(np.load(self.xyz_grid_path), self.z_axis)
        if self.q_bounds is None:                      # default to training rate range
            self.q_bounds = (self.recon.q_min, self.recon.q_max)
        return self


# --------------------------------------------------------------------------- #
# 3. Objective / constraint kernels                                           #
# --------------------------------------------------------------------------- #
def injected_mass(q, cfg):
    return cfg.rho_ref * q * cfg.T_days


def pressure_fields(U, xi, cfg):
    """U:(B,3)=[q,i,j], xi:(18,) or (B,18) -> P:(B,nx,ny,nz,nt) Pa (NaN inactive).
    NOTE: one field is ~0.7 GB; do NOT call this on a whole population at once.
    Use aor_of_candidates() in the optimization loop, which is memory-bounded."""
    X = recon.assemble_input(xi, U[:, 0], U[:, 1], U[:, 2], cfg.recon)
    return recon.predict_pressure(X, cfg.recon)


def area_of_review_2ts(P2, cfg):
    """AoR from a two-timestep field P2:(B,nx,ny,nz,2) = [t0, t_end]. -> (B,) m^2."""
    dP = P2[..., 1] - P2[..., 0]                          # (B,nx,ny,nz) buildup, Pa
    with np.errstate(invalid="ignore"):
        dP_col = np.nanmax(dP, axis=cfg.z_axis + 1)       # collapse depth -> (B,nx,ny)
    exceed = np.greater(dP_col * cfg.pa_to_psi, cfg.dP_crit_psi)
    return (exceed * cfg.footprint2d[None]).sum(axis=(1, 2))


def aor_of_candidates(U, xi, cfg):
    """Fast, memory-safe AoR over a population: reconstruct only the two timesteps
    the AoR needs (t0, t_end) per candidate. U:(B,3) -> (B,) AoR [m^2]."""
    X = recon.assemble_input(xi, U[:, 0], U[:, 1], U[:, 2], cfg.recon)
    out = np.empty(X.shape[0])
    for k in range(X.shape[0]):
        P2 = recon.predict_pressure_timesteps(
            X[k:k + 1], cfg.recon, (cfg.t0_idx, cfg.t_end_idx))   # (1,nx,ny,nz,2)
        out[k] = area_of_review_2ts(P2, cfg)[0]
        del P2
    return out


def containment_of_candidates(U, xi, cfg):
    """Memory-safe containment over a population, one saturation field at a time."""
    X = recon.assemble_input(xi, U[:, 0], U[:, 1], U[:, 2], cfg.recon)
    out = np.empty(X.shape[0])
    for k in range(X.shape[0]):
        Sg = recon.predict_saturation(X[k:k + 1], cfg.recon)
        m = np.nanmax(Sg[0], axis=(2, 3))                     # over z,t -> (nx,ny)
        out[k] = _subdomain_boundary_max(m) - cfg.sat_boundary_tol
        del Sg
    return out


def area_of_review(P, cfg):
    """Area-weighted AoR: final-minus-initial buildup, vertical worst case, psi
    threshold, nan-safe. P:(B,nx,ny,nz,nt) -> (B,) area [m^2]."""
    dP = P[..., cfg.t_end_idx] - P[..., cfg.t0_idx]       # (B,nx,ny,nz) Pa
    with np.errstate(invalid="ignore"):
        dP_col = np.nanmax(dP, axis=cfg.z_axis + 1)       # collapse depth -> (B,nx,ny)
    exceed = np.greater(dP_col * cfg.pa_to_psi, cfg.dP_crit_psi)   # NaN -> False
    return (exceed * cfg.footprint2d[None]).sum(axis=(1, 2))


def _subdomain_boundary_max(m2d):
    """Max of a per-column field over the ring at the SATURATION SUBDOMAIN edge
    (i=16/108, j=12/105) -- the boundary of the region the net actually models.
    m2d: (...,nx,ny) -> (...): last axis reduced over the ring."""
    xs0, xs1 = recon.SAT_X[0], recon.SAT_X[1] - 1        # 16, 108
    ys0, ys1 = recon.SAT_Y[0], recon.SAT_Y[1] - 1        # 12, 105
    ring = np.concatenate([
        m2d[..., xs0, ys0:ys1 + 1], m2d[..., xs1, ys0:ys1 + 1],   # top/bottom rows
        m2d[..., xs0:xs1 + 1, ys0], m2d[..., xs0:xs1 + 1, ys1],   # left/right cols
    ], axis=-1)
    return ring.max(axis=-1)


def containment_violation(U, xi, cfg):
    """g<=0 feasible: max SUBDOMAIN-boundary saturation - tol. U:(B,3) -> (B,)."""
    X = recon.assemble_input(xi, U[:, 0], U[:, 1], U[:, 2], cfg.recon)
    Sg = recon.predict_saturation(X, cfg.recon)           # (B,nx,ny,nz,nt)
    m = np.nanmax(Sg, axis=(3, 4))                        # over z,t -> (B,nx,ny)
    return _subdomain_boundary_max(m) - cfg.sat_boundary_tol


def reduce_ensemble(vals, cfg):
    if cfg.reduce_mode in ("expected", "chance"):
        return vals.mean(axis=1)
    if cfg.reduce_mode == "worst":
        return vals.max(axis=1)
    if cfg.reduce_mode == "cvar":
        k = max(1, int(np.ceil(cfg.alpha * vals.shape[1])))
        return np.sort(vals, axis=1)[:, ::-1][:, :k].mean(axis=1)
    raise ValueError(cfg.reduce_mode)


# --------------------------------------------------------------------------- #
# 4. pymoo Problem                                                            #
# --------------------------------------------------------------------------- #
class CO2InjectionProblem(Problem):
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.hist_X, self.hist_F = [], []      # every evaluated population (solution space)
        xl = np.array([cfg.q_bounds[0], cfg.i_bounds[0], cfg.j_bounds[0]])
        xu = np.array([cfg.q_bounds[1], cfg.i_bounds[1], cfg.j_bounds[1]])
        n_constr = int(cfg.use_containment) + int(cfg.robust and
                                                  cfg.reduce_mode == "chance")
        super().__init__(n_var=3, n_obj=2, n_ieq_constr=n_constr, xl=xl, xu=xu)

    def _evaluate(self, U, out, *args, **kwargs):
        cfg = self.cfg
        f1 = -injected_mass(U[:, 0], cfg)                 # analytic, geology-free
        G = []

        if not cfg.robust:
            f2 = aor_of_candidates(U, cfg.xi, cfg)
            if cfg.use_containment:
                G.append(containment_of_candidates(U, cfg.xi, cfg))
        else:
            aor = np.column_stack([aor_of_candidates(U, xi, cfg)
                                   for xi in cfg.xi_ensemble])   # (B, M)
            f2 = reduce_ensemble(aor, cfg)
            if cfg.reduce_mode == "chance":
                G.append((aor > cfg.aor_cap).mean(axis=1) - cfg.alpha)
            if cfg.use_containment:
                sv = np.column_stack([containment_of_candidates(U, xi, cfg)
                                      for xi in cfg.xi_ensemble])
                G.append(sv.max(axis=1))

        out["F"] = np.column_stack([f1, f2])
        if G:
            out["G"] = np.column_stack(G)
        # record this population (for the solution-space cloud + global front)
        self.hist_X.append(np.array(U, copy=True))
        self.hist_F.append(out["F"].copy())

# --------------------------------------------------------------------------- #
# 5. Run                                                                       #
# --------------------------------------------------------------------------- #
def run(cfg: Config, pop_size: int = 30, n_gen: int = 50, seed: int = 1, out_prefix: str = "pareto_front"):
    cfg.build()
    problem = CO2InjectionProblem(cfg)
    algorithm = NSGA2(
        pop_size=pop_size, sampling=LHS(),
        crossover=SBX(prob=0.9, eta=15), mutation=PM(eta=20),
        eliminate_duplicates=True,
    )
    if n_gen is None:
        termination = RobustTermination(
            MultiObjectiveSpaceTermination(tol=0.0025), period=10)
        print(f"NSGA-II: pop={pop_size}, adaptive termination (no hard cap)")
    else:
        termination = ("n_gen", n_gen)
        print(f"NSGA-II: pop={pop_size}, n_gen={n_gen}  "
              f"(~{pop_size * n_gen} evaluations"
              f"{f', x{len(cfg.xi_ensemble)} realizations' if cfg.robust else ''})")
    
    # collect per-generation objective values via a lightweight callback
    # (NOT save_history=True, which deep-copies the KL matrices/models each gen -> OOM)
    # copy_algorithm=False -> the SAME `problem` object accumulates hist_X/hist_F,
    # so we capture every evaluated population without relying on pymoo's callback hook
    res = minimize(problem, algorithm, termination, seed=seed,
                   copy_algorithm=False, verbose=True)
 
    if problem.hist_F:
        allF, allX = np.vstack(problem.hist_F), np.vstack(problem.hist_X)
        init_F = problem.hist_F[0]                    # first population = initial pop
        # PARETO FRONT = non-dominated set over ALL evaluations (not just the final
        # generation), so the plotted front is the true envelope of the solution space
        nd = NonDominatedSorting().do(allF, only_non_dominated_front=True)
        frontF, frontX = allF[nd], allX[nd]
    else:
        allF, init_F = None, None
        frontF, frontX = res.F, res.X                # fallback: final-generation front
 
    mass, aor_km2 = -frontF[:, 0], frontF[:, 1] / 1e6
    order = np.argsort(aor_km2)
    tag = f"robust[{cfg.reduce_mode}]" if cfg.robust else "deterministic"
    print(f"\nPareto front ({tag}); AoR in km^2 (dP_crit={cfg.dP_crit_psi} psi):")
    print(f"{'mass':>14} {'AoR_km2':>10} {'q':>12} {'i':>7} {'j':>7}")
    for k in np.argsort(mass):
        print(f"{mass[k]:14.4g} {aor_km2[k]:10.3f} "
              f"{frontX[k,0]:12.4g} {frontX[k,1]:7.1f} {frontX[k,2]:7.1f}")
 
    # --- save the front (data) ---
    front = np.column_stack([mass, aor_km2, frontX])[np.argsort(mass)]  # mass,AoR,q,i,j
    np.savetxt(out_prefix + ".csv", front, delimiter=",",
               header="mass,AoR_km2,q,well_i,well_j", comments="")
 
    # --- figure: solution space (grey) + initial population (blue) + front (red) ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 5.2))
 
        # solution space (all evaluations) + initial population
        if allF is not None:
            ax.scatter(allF[:, 1] / 1e6, -allF[:, 0], s=9, c="0.6",
                       label="Solution space", zorder=1)
            ax.scatter(init_F[:, 1] / 1e6, -init_F[:, 0], s=16, c="#2b6cb0",
                       label="Initial population", zorder=2)
            npts = allF.shape[0]
        else:
            print("  (note: no per-generation history captured; plotting front only)")
            npts = 0
 
        # robust: shade each front design's AoR spread across ALL test cases
        if cfg.robust and cfg.xi_ensemble is not None:
            perF = np.vstack([aor_of_candidates(frontX, xi, cfg)
                              for xi in cfg.xi_ensemble]) / 1e6      # (M, n_front) km^2
            o = np.argsort(mass)
            ax.fill_betweenx(mass[o], perF.min(0)[o], perF.max(0)[o],
                             color="#e53e3e", alpha=0.15, zorder=2,
                             label="AoR range across test cases")
            hdr = ("mass,AoR_mean_km2," +
                   ",".join(f"AoR_case{m}" for m in range(perF.shape[0])))
            np.savetxt(out_prefix + "_percase.csv",
                       np.column_stack([mass[o], perF.mean(0)[o], perF.T[o]]),
                       delimiter=",", header=hdr, comments="")
            print(f"wrote {out_prefix}_percase.csv (AoR per test case per front design)")
 
        ax.plot(aor_km2[order], mass[order], "-", color="#e53e3e", lw=1, zorder=3)
        ax.scatter(aor_km2, mass, s=18, c="#e53e3e", label="Pareto front", zorder=4)
        ax.set_xlabel("Area of Review [km$^2$]  (minimize)")
        ax.set_ylabel("injected CO$_2$ mass [MMT]  (maximize)")
        #ax.set_title(f"Pareto front ({tag}, $\\Delta P_{{crit}}$={cfg.dP_crit_psi:.0f} psi)")
        ax.legend(frameon=True); ax.grid(alpha=.3)
        fig.tight_layout()
        fig.savefig(out_prefix + ".png", dpi=140)
        print(f"\nwrote {out_prefix}.csv and {out_prefix}.png"
              f"{f' ({npts} evaluated points shown)' if npts else ''}")
    except Exception as e:
        print(f"(plot skipped: {e}); wrote {out_prefix}.csv")
    return res

XI_PATH  = "inputs/xi_site_test.npy"   # (test_size, 18) ensemble, or a single (18,) vector
#XI_PATH  = "inputs/xi_inv_historic.npy"   # (test_size, 18) ensemble, or a single (18,) vector

def load_xi(index=0):
    """Return one 18-d geology latent. Extracts row `index` if xi_test.npy is an
    ensemble (test_size, 18); returns it directly if it is already (18,)."""
    arr = np.load(XI_PATH)
    xi = arr[index] if arr.ndim == 2 else arr
    xi = np.asarray(xi, dtype=np.float64).reshape(-1)
    if xi.shape[0] != 18:
        raise SystemExit(
            f"xi must be length 18 (6 xy + 6 z + 6 por); got {xi.shape[0]} from "
            f"{XI_PATH}. If that file holds only one field's latent (e.g. 6), "
            f"concatenate xy+z+por into an (18,) or (test_size,18) array first.")
    if arr.ndim == 2:
        print(f"using test-case xi index {index} of {arr.shape[0]}")
    return xi


# Geological ensemble for robust design (realizations of the same site). Point this
# at the training realizations for train/optimize -> test/validate discipline, or
# reuse XI_PATH to optimize robustly over the test set.
ENSEMBLE_PATH = XI_PATH


def make_config(xi_index=0, robust=False, reduce_mode="expected", aor_cap=None):
    cfg = Config(
        xi=load_xi(xi_index),               # one test case (deterministic mode / smoke test)
        xyz_grid_path="/home/camunoz4/IBDP_2024Dataset/xyz_grid_right_bounds.npy",
        dP_crit_psi=75.0,
        robust=robust,
        reduce_mode=reduce_mode,
        aor_cap=aor_cap,
    )
    if robust:
        ens = np.load(ENSEMBLE_PATH)
        ens = ens.reshape(1, -1) if ens.ndim == 1 else ens
        assert ens.shape[1] == 18, f"ensemble must be (M,18); got {ens.shape}"
        cfg.xi_ensemble = ens
        print(f"robust over {ens.shape[0]} realizations, reduce='{reduce_mode}'")
    cfg.recon.pressure_weights   = "pres/model_pres_latent.pth"
    cfg.recon.saturation_weights = "sats/model_s_latent.pth"
    cfg.recon.kl_hdf5            = "pres/p_reconstruction.hdf5"
    cfg.recon.nan_locations      = "pres/nan_locations.npy"
    cfg.recon.rate_file          = "rate_ft3_day.npy"
    return cfg


def smoke_test(cfg):
    """One-candidate sanity check of the reconstruction + AoR before the full run.
    Verifies shapes, that pressure is positive and physically scaled (Pa), and
    that the AoR is a plausible area. Run with:  python co2_moop_scaffold.py --check
    """
    cfg.build()
    q_mid = 0.5 * (cfg.q_bounds[0] + cfg.q_bounds[1])
    i_mid = 0.5 * (cfg.i_bounds[0] + cfg.i_bounds[1])
    j_mid = 0.5 * (cfg.j_bounds[0] + cfg.j_bounds[1])
    U = np.array([[q_mid, i_mid, j_mid]])
    print(f"smoke test: q={q_mid:.4g} (range {cfg.q_bounds}), well=({i_mid:.0f},{j_mid:.0f})")

    P = pressure_fields(U, cfg.xi, cfg)                 # (1,nx,ny,nz,nt)
    print(f"  pressure field shape : {P.shape}   (expect (1,126,125,110,52))")
    print(f"  active-cell P range  : {np.nanmin(P):.4g} .. {np.nanmax(P):.4g} Pa")
    dP = P[0, ..., cfg.t_end_idx] - P[0, ..., cfg.t0_idx]
    with np.errstate(invalid="ignore"):
        dpsi = np.nanmax(dP) * cfg.pa_to_psi
    print(f"  peak buildup         : {dpsi:.2f} psi   (final - initial)")
    print(f"  domain area          : {cfg.footprint2d.sum()/1e6:.1f} km^2")
    aor = area_of_review(P, cfg)[0]                       # full-field
    aor_fast = aor_of_candidates(U, cfg.xi, cfg)[0]       # 2-timestep path (optimization uses this)
    print(f"  AoR @ {cfg.dP_crit_psi:.0f} psi (full) : {aor/1e6:.3f} km^2")
    print(f"  AoR @ {cfg.dP_crit_psi:.0f} psi (fast) : {aor_fast/1e6:.3f} km^2   (should match full)")

    if cfg.use_containment:
        g = containment_violation(U, cfg.xi, cfg)[0]
        print(f"  containment g (<=0)  : {g:.4g}")

    ok = (P.shape == (1, recon.NX, recon.NY, recon.NZ, recon.NT)
          and np.nanmin(P) > 0 and np.isfinite(aor)
          and np.isclose(aor, aor_fast))
    print("  ->", "PASS: reconstruction looks sane; ready for run(cfg)." if ok
          else "CHECK: something looks off (see the flagged items in the notes).")
    return ok


if __name__ == "__main__":
    import sys
    args = sys.argv[1:]
    # optional integer picks the test-case row (deterministic / smoke test):
    #   python co2_moop_scaffold.py --check 3     one-candidate sanity check, case 3
    #   python co2_moop_scaffold.py 3             deterministic front for case 3
    #   python co2_moop_scaffold.py --robust      one robust front over the ensemble
    nums = [a for a in args if a.lstrip("-").isdigit()]
    xi_index = int(nums[0]) if nums else 0
    robust = "--robust" in args
    cfg = make_config(xi_index=xi_index, robust=robust, reduce_mode="expected")
    if "--check" in args:
        smoke_test(cfg)          # quick one-candidate sanity check
    else:
        run(cfg, out_prefix = 'pareto_front_historic')                 # full NSGA-II optimization (robust if --robust)