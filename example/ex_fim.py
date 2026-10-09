"""Fisher-information experimental design for EUV scatterometry with grcwa.

Follows the Fisher-information workflow of Durant, Wilkins & Cooper,
J. Appl. Cryst. 55, 769 (2022), as used by Klein et al., arXiv:2504.12133,
with grcwa as the forward model:

  1. Candidate measurement conditions c = (HHG harmonic, grazing angle, polarization).
  2. Grating-equation pruning: keep only reflected orders that propagate and land
     on the detector (analytic, no RCWA).
  3. grcwa diffraction efficiencies r_m and a central-difference Jacobian dr_m/dxi.
  4. Noise model: photon shot noise + background + detector read noise.
  5. Normalized Fisher information per condition, cached.
  6. Two criteria (A-optimality, E-optimality) x K in {1, 2} conditions -> 4 plans.
     K = 2 uses greedy selection and is checked against an exhaustive pair search.
  7. Monte Carlo: synthetic noisy data, grcwa least-squares fit, sigma_MC vs CRLB.

The sample is a toy 1D line grating (Si lines on SiO2 on Si). The optical
constants, HHG flux, and detector numbers below are ILLUSTRATIVE ONLY; replace
them with tabulated values (e.g. CXRO) and your own instrument parameters.

Run:  python example/ex_fim.py            (about 1–2 minutes)
      python example/ex_fim.py --quick    (smaller candidate set, fewer MC runs)

Units: lengths in nm, so freq = 1/lambda[nm] (grcwa uses c = 1).
"""

import argparse
import itertools
import time
from dataclasses import dataclass

import numpy as np

import grcwa

# ---------------------------------------------------------------- ① sample
PERIOD = 100.0  # nm, grating period along x
LY = 5.0  # nm, short y period keeps every y order evanescent (needs LY < lambda/2)
NX, NY = 400, 1  # pixels of the patterned layer (line grating: uniform in y)
NG = 41  # truncation order; with LY << PERIOD these are x orders m = -20..20

PARAM_NAMES = ("CD", "h", "t_ox")
XI0 = np.array([30.0, 40.0, 10.0])  # nominal CD, line height, oxide thickness (nm)
PRIOR_RANGE = np.array([4.0, 4.0, 3.0])  # full width of each prior range (nm)
FD_STEP = np.array([0.05, 0.05, 0.05])  # central-difference steps (nm)


# ---------------------------------------------------- ② optical constants
def refractive_index(material, lam):
    """Toy EUV index n = 1 - delta + i*beta. ILLUSTRATIVE ONLY: use CXRO tables."""
    s = lam / 13.5
    delta, beta = {
        "Si": (1.0e-3 * s**2, 1.8e-3 * s**2),
        "SiO2": (2.2e-2 * s**2, 1.1e-2 * s**2),
    }[material]
    return 1.0 - delta + 1j * beta


def permittivity(material, lam):
    return refractive_index(material, lam) ** 2


# ------------------------------------------------ Ⓐ design space
@dataclass(frozen=True)
class Condition:
    q: int  # harmonic order of an 800 nm driver
    graze: float  # incidence angle from the sample surface, degrees
    pol: str  # "s" or "p"
    phi: float = 0.0  # azimuth, radians

    @property
    def lam(self):
        return 800.0 / self.q

    @property
    def theta(self):
        """Polar angle from the surface normal, as grcwa expects."""
        return np.deg2rad(90.0 - self.graze)

    def label(self):
        return f"{self.lam:5.2f} nm (q{self.q}), {self.graze:4.1f} deg, {self.pol}"


HARMONICS = (59, 53, 47, 41, 35)
GRAZE_ANGLES = np.arange(4.0, 25.0, 2.0)
POLARIZATIONS = ("s", "p")
DETECTOR_GRAZE = (2.0, 60.0)  # exit grazing angles covered by the detector, degrees

# ------------------------------------------------ ⑦ noise model (illustrative)
FLUX = {59: 2e7, 53: 4e7, 47: 6e7, 41: 8e7, 35: 6e7}  # photons/s on sample per harmonic
T_OPTICS = 0.3  # optics throughput
QE = 0.5  # detector quantum efficiency
P_VAC = 1e-2  # chamber pressure, Pa
PATH_LENGTH = 1.0  # beam path in residual gas, m
SIGMA_READ = 5.0  # read noise per order (counts)
BKG_RATE = 2.0  # stray light + dark counts per second per order
T_TOTAL = 10.0  # total measurement time shared by all conditions, s
TAUS = np.linspace(0.2, 0.8, 7)  # time split tested for K = 2


def gas_transmission(lam):
    """Beer-Lambert absorption by residual gas. Cross-section is a rough toy value."""
    n_gas = P_VAC / (1.380649e-23 * 295.0)  # molecules / m^3
    sigma = 1e-21 * (lam / 13.5) ** 2  # m^2
    return np.exp(-n_gas * sigma * PATH_LENGTH)


def photons(c, t):
    """s: expected detected photons per unit diffraction efficiency."""
    return FLUX[c.q] * T_OPTICS * gas_transmission(c.lam) * QE * t


def count_variance(r, c, t):
    s = photons(c, t)
    return s, s * r + BKG_RATE * t + SIGMA_READ**2


# ------------------------------------------------ ③ grating-equation pruning
LK1, LK2 = grcwa.Lattice_Reciprocate([PERIOD, 0.0], [0.0, LY])
G_ORDERS, NG_EFF = grcwa.Lattice_getG(NG, LK1, LK2)  # same ordering as RT_Solve(byorder=1)


def usable_orders(c, detector=True):
    """Indices of reflected orders that propagate (and, if detector, reach the detector)."""
    k0 = 2 * np.pi / c.lam
    kx0 = k0 * np.sin(c.theta) * np.cos(c.phi)
    ky0 = k0 * np.sin(c.theta) * np.sin(c.phi)
    kx, ky = grcwa.Lattice_SetKs(G_ORDERS, kx0, ky0, LK1, LK2)
    kpar = np.sqrt(kx**2 + ky**2)
    keep = kpar < k0
    if detector:
        exit_graze = np.degrees(np.arccos(np.clip(kpar / k0, 0.0, 1.0)))
        keep &= (exit_graze >= DETECTOR_GRAZE[0]) & (exit_graze <= DETECTOR_GRAZE[1]) & (kx > 0)
    return np.where(keep)[0]


# ------------------------------------------------ ④–⑥ grcwa forward model
def fill_fraction(cd):
    """Area fraction of each pixel covered by a centred line of width cd.

    Partial edge pixels make r_m a smooth function of CD, so finite differences work.
    """
    edges = np.arange(NX + 1) * PERIOD / NX
    lo, hi = PERIOD / 2 - cd / 2, PERIOD / 2 + cd / 2
    overlap = np.minimum(edges[1:], hi) - np.maximum(edges[:-1], lo)
    return np.clip(overlap, 0.0, None) / (PERIOD / NX)


def efficiencies(xi, c, n_g=NG):
    """Reflected efficiency of every order, ordered like G_ORDERS."""
    lam = c.lam
    obj = grcwa.obj(n_g, [PERIOD, 0.0], [0.0, LY], 1.0 / lam, c.theta, c.phi, verbose=0)
    obj.Add_LayerUniform(1.0, 1.0)  # vacuum
    obj.Add_LayerGrid(xi[1], NX, NY)  # Si lines
    obj.Add_LayerUniform(xi[2], permittivity("SiO2", lam))  # oxide film
    obj.Add_LayerUniform(1.0, permittivity("Si", lam))  # substrate
    obj.Init_Setup()
    p_amp, s_amp = (1.0, 0.0) if c.pol == "p" else (0.0, 1.0)
    obj.MakeExcitationPlanewave(p_amp, 0.0, s_amp, 0.0, order=0)
    eps_line = permittivity("Si", lam)
    grid = 1.0 + fill_fraction(xi[0]) * (eps_line - 1.0)
    obj.GridLayer_geteps(np.repeat(grid, NY))
    ri, _ = obj.RT_Solve(normalize=1, byorder=1)
    return np.asarray(ri)


def forward_and_jacobian(xi, c, m):
    """Ⓒ r_m and J_mi = dr_m/dxi_i by central differences (1 + 2P grcwa solves)."""
    r0 = efficiencies(xi, c)[m]
    jac = np.empty((len(m), len(xi)))
    for k in range(len(xi)):
        step = np.zeros_like(xi)
        step[k] = FD_STEP[k]
        jac[:, k] = (efficiencies(xi + step, c)[m] - efficiencies(xi - step, c)[m]) / (
            2 * FD_STEP[k]
        )
    return r0, jac


# ------------------------------------------------ Ⓓ–Ⓕ normalized FIM
D_NORM = np.diag(PRIOR_RANGE)


def fim_normalized(r, jac, c, t):
    """ĝ = D (Jᵀ M J) D with M = diag(s² / Var N)."""
    s, var = count_variance(r, c, t)
    w = s**2 / var
    g = jac.T @ (w[:, None] * jac)
    return D_NORM @ g @ D_NORM


def psi_a(g):
    """A-optimality: -tr(ĝ⁻¹); -inf when ĝ is singular."""
    ev = np.linalg.eigvalsh(g)
    if ev[0] <= 1e-12 * max(ev[-1], 1e-300):
        return -np.inf
    return -np.sum(1.0 / ev)


def psi_e(g):
    """E-optimality: smallest eigenvalue of ĝ."""
    return np.linalg.eigvalsh(g)[0]


CRITERIA = {"A": psi_a, "E": psi_e}


def improvement(new, old):
    if old == -np.inf:
        return np.inf if new > -np.inf else 0.0
    return (new - old) / abs(old)


# ------------------------------------------------ Ⓖ selection
def plan_fim(plan, fims):
    return sum(fims[(c, t)] for c, t in plan)


def select_plans(conds, fims, psi, threshold):
    """Return K = 1 plan, greedy K = 2 plan, exhaustive K = 2 plan and the adopted plan."""
    score = lambda plan: psi(plan_fim(plan, fims))  # noqa: E731

    k1 = max(([(c, T_TOTAL)] for c in conds), key=score)
    c1 = k1[0][0]
    greedy = max(
        ([(c1, tau * T_TOTAL), (c, (1 - tau) * T_TOTAL)] for c in conds if c != c1 for tau in TAUS),
        key=score,
    )
    pair = max(
        (
            [(a, tau * T_TOTAL), (b, (1 - tau) * T_TOTAL)]
            for a, b in itertools.combinations(conds, 2)
            for tau in TAUS
        ),
        key=score,
    )
    best2 = pair if score(pair) >= score(greedy) else greedy
    adopt2 = improvement(score(best2), score(k1)) > threshold
    return {
        "K1": k1,
        "greedy": greedy,
        "pair": pair,
        "K2": best2 if adopt2 else k1,
        "K2_adopted": adopt2,
        "score": score,
    }


def describe(plan, fims):
    g = plan_fim(plan, fims)
    ev = np.linalg.eigvalsh(g)
    if ev[0] <= 1e-12 * ev[-1]:
        return {"sigma": np.full(len(XI0), np.inf), "lam_min": ev[0], "tr": np.inf, "rho": np.nan}
    cov = np.linalg.inv(g)
    sig = np.sqrt(np.diag(cov))
    rho = cov / np.outer(sig, sig)
    off = np.abs(rho[~np.eye(len(sig), dtype=bool)])
    return {"sigma": sig, "lam_min": ev[0], "tr": np.trace(cov), "rho": off.max()}


def plan_text(plan):
    return " + ".join(f"[{c.label()}, {t:.1f} s]" for c, t in plan)


# ------------------------------------------------ ⑧–Ⓘ Monte Carlo
def clip_params(xi):
    """Keep the fit inside twice the prior window (the fit has no other prior)."""
    return np.clip(xi, XI0 - PRIOR_RANGE, XI0 + PRIOR_RANGE)


def fit(plan, data, orders, n_iter=6):
    """Weighted Gauss-Newton least squares with the same grcwa model."""
    xi = XI0.copy()
    for _ in range(n_iter):
        jtwj = np.zeros((len(xi), len(xi)))
        jtwr = np.zeros(len(xi))
        for (c, t), r_hat in zip(plan, data):
            r, jac = forward_and_jacobian(xi, c, orders[c])
            s, var = count_variance(np.clip(r, 0.0, None), c, t)
            w = s**2 / var
            jtwj += jac.T @ (w[:, None] * jac)
            jtwr += jac.T @ (w * (r_hat - r))
        step = np.linalg.lstsq(jtwj, jtwr, rcond=1e-10)[0]
        step *= min(1.0, 0.5 / np.max(np.abs(step) / PRIOR_RANGE))  # limit to half a prior range
        xi = clip_params(xi + step)
        if np.max(np.abs(step) / PRIOR_RANGE) < 1e-4:
            break
    return xi


def monte_carlo(plan, orders, n_mc, rng):
    errors = []
    for _ in range(n_mc):
        xi_true = XI0 + (rng.random(len(XI0)) - 0.5) * PRIOR_RANGE
        data = []
        for c, t in plan:
            r = efficiencies(xi_true, c)[orders[c]]
            s = photons(c, t)
            counts = rng.poisson(s * r + BKG_RATE * t) + rng.normal(0.0, SIGMA_READ, len(r))
            data.append((counts - BKG_RATE * t) / s)
        errors.append(fit(plan, data, orders) - xi_true)
    errors = np.array(errors)
    return errors.std(axis=0), errors.mean(axis=0)


# ------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--quick", action="store_true", help="smaller candidate set and MC")
    ap.add_argument("--n-mc", type=int, default=40, help="Monte Carlo fits per plan")
    ap.add_argument("--threshold", type=float, default=0.2, help="min gain to adopt K = 2")
    ap.add_argument("--max-ratio", type=float, default=2.0, help="max sigma_MC / sigma_CRLB")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    harmonics, angles, n_mc = HARMONICS, GRAZE_ANGLES, args.n_mc
    if args.quick:
        harmonics, angles, n_mc = HARMONICS[::2], GRAZE_ANGLES[::2], min(n_mc, 15)
    rng = np.random.default_rng(args.seed)
    t_start = time.time()

    # Ⓐ → ③ → Ⓑ
    candidates = [
        Condition(q, float(a), p) for q in harmonics for a in angles for p in POLARIZATIONS
    ]
    orders = {c: usable_orders(c) for c in candidates}
    conds = [c for c in candidates if len(orders[c]) > 0]
    n_ord = [len(orders[c]) for c in conds]
    print(f"Ⓐ candidates: {len(candidates)}   Ⓑ after grating-equation pruning: {len(conds)}")
    n_prop = [len(usable_orders(c, detector=False)) for c in conds]
    print(
        f"   reflected orders per condition: {min(n_prop)}–{max(n_prop)} propagating, "
        f"{min(n_ord)}–{max(n_ord)} on the detector"
    )

    # nG convergence spot check (also the cheapest guard against a bad Jacobian)
    c_chk = conds[len(conds) // 2]
    m_chk = orders[c_chk]
    r_a = efficiencies(XI0, c_chk)[m_chk]
    obj_b = efficiencies(XI0, c_chk, n_g=2 * NG)
    g_b, _ = grcwa.Lattice_getG(2 * NG, LK1, LK2)
    idx_b = [int(np.where((g_b == G_ORDERS[i]).all(axis=1))[0][0]) for i in m_chk]
    r_b = obj_b[idx_b]
    print(
        f"   nG check at {c_chk.label()}: max |Δr|/r = "
        f"{np.max(np.abs(r_a - r_b) / np.maximum(r_b, 1e-30)):.1e} (nG {NG} vs {2 * NG})"
    )

    # ④–Ⓒ forward model + Jacobian, cached per condition
    cache = {}
    for c in conds:
        cache[c] = forward_and_jacobian(XI0, c, orders[c])
    print(f"④–Ⓒ grcwa solves: {len(conds) * (1 + 2 * len(XI0))}  ({time.time() - t_start:.0f} s)")

    # Ⓓ–Ⓕ normalized FIM for every (condition, time) the selection can use
    times = {T_TOTAL, *(tau * T_TOTAL for tau in TAUS), *((1 - tau) * T_TOTAL for tau in TAUS)}
    fims = {(c, t): fim_normalized(*cache[c], c, t) for c in conds for t in times}

    # Ⓖ–Ⓗ selection and comparison
    results = {}
    for name, psi in CRITERIA.items():
        sel = select_plans(conds, fims, psi, args.threshold)
        results[name] = sel
        score = sel["score"]
        miss = score(sel["pair"]) > score(sel["greedy"]) + 1e-9 * abs(score(sel["greedy"]))
        print(f"\n=== {name}-optimality ===")
        print(f"  {name}1 (K=1):      {plan_text(sel['K1'])}")
        print(f"  greedy K=2:      {plan_text(sel['greedy'])}")
        print(
            f"  exhaustive K=2:  {plan_text(sel['pair'])}"
            + ("   <- greedy missed this" if miss else "")
        )
        gain = improvement(max(score(sel["pair"]), score(sel["greedy"])), score(sel["K1"]))
        print(
            f"  K=2 gain over K=1: {gain:+.0%}  -> "
            + ("adopted" if sel["K2_adopted"] else f"not adopted (threshold {args.threshold:.0%})")
        )

    plans = {
        "A1": results["A"]["K1"],
        "A2": results["A"]["K2"],
        "E1": results["E"]["K1"],
        "E2": results["E"]["K2"],
    }

    print("\n=== Ⓗ four plans (CRLB from FIM, σ in nm) ===")
    header = "plan  " + "  ".join(f"σ_{p:>4}" for p in PARAM_NAMES) + "   λ_min    tr(ĝ⁻¹)  max|ρ|"
    print(header)
    info = {}
    for key, plan in plans.items():
        d = describe(plan, fims)
        info[key] = d
        sig_nm = d["sigma"] * PRIOR_RANGE
        print(
            f"{key:4}  "
            + "  ".join(f"{s:6.3f}" for s in sig_nm)
            + f"  {d['lam_min']:8.2f}  {d['tr']:8.4f}  {d['rho']:6.3f}"
        )

    # ⑧–Ⓘ Monte Carlo on each distinct plan
    print(f"\n=== ⑧–Ⓘ Monte Carlo ({n_mc} fits per plan, ξ_true drawn from the prior) ===")
    print("plan  " + "  ".join(f"σMC_{p:>4}" for p in PARAM_NAMES) + "   max σMC/σCRLB")
    done, scores = {}, {}
    for key, plan in plans.items():
        pk = tuple(plan)
        if pk not in done:
            done[pk] = monte_carlo(plan, orders, n_mc, rng)
        sig_mc, _ = done[pk]
        ratio = np.max(sig_mc / (info[key]["sigma"] * PRIOR_RANGE))
        scores[key] = (np.sqrt(np.mean((sig_mc / PRIOR_RANGE) ** 2)), ratio)
        print(f"{key:4}  " + "  ".join(f"{s:8.3f}" for s in sig_mc) + f"   {ratio:8.2f}")

    ok = {k: v for k, v in scores.items() if v[1] <= args.max_ratio}
    print()
    if ok:
        best = min(ok, key=lambda k: ok[k][0])
        print(f"d* = {best}: {plan_text(plans[best])}")
    else:
        print(f"Every plan has σ_MC > {args.max_ratio}×CRLB: enlarge the design space (Ⓐ).")
    print(f"total time {time.time() - t_start:.0f} s")


if __name__ == "__main__":
    main()
