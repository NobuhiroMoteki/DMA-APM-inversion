"""
inversion_solver.py  ---  Chahine-Twomey nonlinear iterative inversion

Solves the Fredholm integral equation of the first kind n = K f.
Combines the Chahine-type weighted-average update rule with Markowski
1-2-1 internal smoothing, and uses the reduced chi-squared based on
Poisson counting statistics as the convergence criterion.

References:
    Twomey (1975) J. Comput. Phys. doi:10.1016/0021-9991(75)90028-5
    Markowski (1987) Aerosol Sci. Technol. doi:10.1080/02786828708959153
"""
from __future__ import annotations

import numpy as np
from data_parser import MeasurementData


def solve_chahine_twomey(
    K:       np.ndarray,
    m_array: np.ndarray,
    data:    MeasurementData,
    params,
) -> np.ndarray:
    """Estimate the mass distribution f(m) using the Chahine-Twomey method.

    Update equation:
        f_j^(k+1) = f_j^(k) x sum_i [K_norm(i,j) * (n_meas_i / n_calc_i^(k))]
                                    / sum_i  K_norm(i,j)

    Convergence criterion (reduced chi-squared):
        chi^2 = (1/I) sum_i [(n_meas_i - n_calc_i)^2 / Var(n_i)]  < chi_threshold

    Poisson variance:
        Var(n_i) = max(n_calc_i / V_sample_i,  1 / V_sample_i^2)

    Smoothing (Markowski 1-2-1):
        f_j <- 0.25*f_{j-1} + 0.50*f_j + 0.25*f_{j+1}   (interior points only)

    Args:
        K:       Kernel matrix [I x J]
        m_array: Mass grid [kg], shape (J,)
        data:    Binned measurement data (n_meas, V_sample_array, I)
        params:  User configuration (J, m_min_fg, m_max_fg, max_iter, chi_threshold)

    Returns:
        f_estimated: Estimated mass distribution dN/dm [cm^-3 kg^-1], shape (J,)
    """
    n_meas         = data.n_meas
    V_sample_array = data.V_sample_array
    I              = data.I
    J              = params.J
    m_min          = params.m_min_fg * 1e-18   # [kg]
    m_max          = params.m_max_fg * 1e-18   # [kg]

    # Initial guess: uniform distribution over the mass range
    f_est  = np.ones(J) * (np.sum(n_meas) / (m_max - m_min))
    K_norm = K / np.max(K)

    chi_sq = np.inf
    print("Starting inversion (Chahine-Twomey iteration)...")

    for k in range(params.max_iter):
        calc_n = K @ f_est
        calc_n = np.maximum(calc_n, 1e-10)   # prevent division by zero

        # Reduced chi-squared based on Poisson statistics
        variance = np.maximum(calc_n / V_sample_array, 1.0 / V_sample_array ** 2)
        chi_sq   = np.sum(((n_meas - calc_n) ** 2) / variance) / I

        if chi_sq < params.chi_threshold:
            print(f"  -> Converged after {k} iterations (chi^2 = {chi_sq:.4f})")
            break

        # Chahine-type weighted-average update
        ratio = n_meas / calc_n
        f_new = np.copy(f_est)
        for j in range(J):
            correction = np.sum(ratio * K_norm[:, j]) / max(
                np.sum(K_norm[:, j]), 1e-10
            )
            f_new[j] = f_est[j] * max(correction, 0.01)

        # Markowski 1-2-1 smoothing (endpoints are not updated)
        f_smooth        = np.copy(f_new)
        f_smooth[1:-1]  = (
            0.25 * f_new[:-2] + 0.5 * f_new[1:-1] + 0.25 * f_new[2:]
        )
        f_est = np.copy(f_smooth)

    else:
        print(f"  -> Reached maximum iterations {params.max_iter} (chi^2 = {chi_sq:.4f})")

    return f_est
