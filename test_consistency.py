"""
test_consistency.py  ---  Numerical equivalence verification before and after refactoring

Runs both the original notebook code (extracted verbatim) and the new modular
code under identical conditions, and confirms that the kernel matrix K and the
estimated mass distribution f_estimated agree to within floating-point precision.

Synthetic data (log-normal distribution + Poisson noise) are used as input,
so no real measurement CSV file is required.

Usage:
    python test_consistency.py
"""
import sys
import types
import numpy as np
from scipy.stats import lognorm

# ==============================================================================
# Shared parameters  (identical values used by both old and new code)
# ==============================================================================
E   = 1.60219e-19
VIS = 1.83e-5
P   = 1.013e5

L     = 100.0e-3
R1    = 24.0e-3
R2    = 25.0e-3
RC    = 0.5 * (R1 + R2)
DELTA = 0.5 * (R2 - R1)

RPM    = 4010.0
DMOB   = 450.0e-9
Q_LPM  = 0.3
Q      = Q_LPM * 1e-3 / 60.0
OMEGA  = RPM / 60.0 * 2.0 * np.pi

CC = 1.0 + 1.0 / (P * 1e-3 * DMOB * 1e6) * (
    15.60 + 7.00 * np.exp(-0.059 * (P * 1e-3 * DMOB * 1e6))
)

M_MIN_FG, M_MAX_FG = 5.0, 80.0
M_MIN = M_MIN_FG * 1e-18
M_MAX = M_MAX_FG * 1e-18
J     = 40
NR0   = 1000
DZ    = 1.0e-4
MAX_ITER      = 2000
CHI_THRESHOLD = 1.0

# Synthetic measurement grid (20 voltage points)
V_ARRAY = np.linspace(200, 1200, 20)
I       = len(V_ARRAY)

Q_CPC_CCPS     = 0.3 * 1000.0 / 60.0   # [cm^3/s]
T_MEAS         = 10.0                   # [s] per bin
V_SAMPLE_ARRAY = np.full(I, Q_CPC_CCPS * T_MEAS)


# ==============================================================================
# Original code (extracted verbatim from the notebook)
# ==============================================================================

def _calc_transfer_efficiency_old(m, V):
    r0      = np.linspace(R1 + 1e-5, R2 - 1e-5, NR0)
    r       = r0.copy()
    weights = 1.5 * (1.0 - ((r0 - RC) / DELTA) ** 2)
    total_weight = np.sum(weights)
    coef    = DZ * 8.0 / (9.0 * VIS) * (CC / DMOB) * (DELTA * RC) / Q
    V_term  = E * V / np.log(R2 / R1)
    active  = np.ones(NR0, dtype=bool)
    num_steps = int(L / DZ)
    for _ in range(num_steps):
        if not np.any(active):
            break
        ra = r[active]
        def f(rad):
            return coef * (m * OMEGA**2 * rad - V_term / rad) / (
                1.0 - ((rad - RC) / DELTA) ** 2
            )
        k1 = f(ra)
        k2 = f(ra + 0.5 * k1)
        k3 = f(ra + 0.5 * k2)
        k4 = f(ra + k3)
        r[active] = ra + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        active[active] = np.abs(r[active] - RC) < (DELTA - 1e-5)
    return np.sum(weights[active]) / total_weight


def _build_K_old():
    m_array = np.linspace(M_MIN, M_MAX, J)
    dm      = m_array[1] - m_array[0]
    K       = np.zeros((I, J))
    for i, V in enumerate(V_ARRAY):
        for j, m in enumerate(m_array):
            K[i, j] = _calc_transfer_efficiency_old(m, V) * dm
    return K, m_array


def _solve_old(K_matrix, n_meas, vol_array):
    f_est  = np.ones(J) * (np.sum(n_meas) / (M_MAX - M_MIN))
    K_norm = K_matrix / np.max(K_matrix)
    for k in range(MAX_ITER):
        calc_n   = K_matrix @ f_est
        calc_n   = np.maximum(calc_n, 1e-10)
        variance = np.maximum(calc_n / vol_array, 1.0 / vol_array ** 2)
        chi_sq   = np.sum(((n_meas - calc_n) ** 2) / variance) / I
        if chi_sq < CHI_THRESHOLD:
            break
        ratio = n_meas / calc_n
        f_new = np.copy(f_est)
        for j in range(J):
            correction = np.sum(ratio * K_norm[:, j]) / max(
                np.sum(K_norm[:, j]), 1e-10
            )
            f_new[j] = f_est[j] * max(correction, 0.01)
        f_smooth       = np.copy(f_new)
        f_smooth[1:-1] = 0.25 * f_new[:-2] + 0.5 * f_new[1:-1] + 0.25 * f_new[2:]
        f_est = np.copy(f_smooth)
    return f_est


# ==============================================================================
# Generate synthetic measurement data  (using the original kernel)
# ==============================================================================
print("=" * 60)
print("Step 1: Building kernel matrix with original code (I=20, J=40)...")
K_old, m_array_old = _build_K_old()

f_true = lognorm.pdf(m_array_old, s=np.log(1.25), scale=2.5e-17)
f_true = f_true / np.sum(f_true * (m_array_old[1] - m_array_old[0])) * 800.0

np.random.seed(42)
n_ideal  = K_old @ f_true
n_counts = np.random.poisson(np.maximum(n_ideal, 0) * V_SAMPLE_ARRAY)
n_meas   = n_counts / V_SAMPLE_ARRAY

print("Step 2: Running inversion with original code...")
f_old = _solve_old(K_old, n_meas, V_SAMPLE_ARRAY)


# ==============================================================================
# New code (using modules)
# ==============================================================================
print("Step 3: Building kernel matrix with new code...")

from data_parser      import MeasurementData
from kernel_simulator import build_kernel_1d
from inversion_solver import solve_chahine_twomey

# Supply identical parameters via SimpleNamespace instead of params.py
p = types.SimpleNamespace(
    r1=R1, r2=R2, L=L, Q_a_lpm=Q_LPM, Q_cpc_lpm=0.3,
    dz=DZ, nr0=NR0,
    m_min_fg=M_MIN_FG, m_max_fg=M_MAX_FG, J=J,
    num_bins=20,
    max_iter=MAX_ITER, chi_threshold=CHI_THRESHOLD,
)

data_new = MeasurementData(
    V_array=V_ARRAY,
    n_meas=n_meas,
    V_sample_array=V_SAMPLE_ARRAY,
    t_meas_array=np.full(I, T_MEAS),
    RPM=RPM,
    Dmob=DMOB,
    I=I,
)

K_new, m_array_new = build_kernel_1d(data_new, p)

print("Step 4: Running inversion with new code...")
f_new = solve_chahine_twomey(K_new, m_array_new, data_new, p)


# ==============================================================================
# Numerical comparison
# ==============================================================================
print("\n" + "=" * 60)
print("Numerical equivalence verification results")
print("=" * 60)

RTOL = 1e-12   # relative tolerance
ATOL = 0.0     # absolute tolerance (0 = strict relative comparison)

results = {}

# m_array
m_match = np.array_equal(m_array_old, m_array_new)
results["m_array (exact)"] = m_match
print(f"  m_array (exact):     {'PASS' if m_match else 'FAIL'}")

# K matrix
k_close   = np.allclose(K_old, K_new, rtol=RTOL, atol=ATOL)
k_max_err = float(np.max(np.abs(K_old - K_new)))
k_rel_err = float(np.max(np.abs((K_old - K_new) / np.maximum(np.abs(K_old), 1e-300))))
results["K matrix"] = k_close
print(f"  K matrix:            {'PASS' if k_close else 'FAIL'}"
      f"  (max abs error = {k_max_err:.2e},  max rel error = {k_rel_err:.2e})")

# f_estimated
f_close   = np.allclose(f_old, f_new, rtol=RTOL, atol=ATOL)
f_max_err = float(np.max(np.abs(f_old - f_new)))
f_rel_err = float(np.max(np.abs((f_old - f_new) / np.maximum(np.abs(f_old), 1e-300))))
results["f_estimated"] = f_close
print(f"  f_estimated:         {'PASS' if f_close else 'FAIL'}"
      f"  (max abs error = {f_max_err:.2e},  max rel error = {f_rel_err:.2e})")

print("=" * 60)
all_pass = all(results.values())
if all_pass:
    print("All items match: numerical equivalence before and after refactoring confirmed.")
    sys.exit(0)
else:
    failed = [k for k, v in results.items() if not v]
    print(f"Mismatch detected: {failed}")
    sys.exit(1)
