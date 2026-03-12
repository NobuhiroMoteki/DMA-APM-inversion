"""
kernel_simulator.py  ---  Numerical computation of the APM transfer function
                          and kernel matrix construction

An original numerical simulation method that directly tracks particle
trajectories inside the APM gap using the 4th-order Runge-Kutta (RK4) method
under an assumed parabolic flow profile, and computes the transmission
efficiency Omega_APM(m, V, Z_p).

The default APM geometry (L=100 mm, r1=24 mm, r2=25 mm) corresponds to the
nominal design values of the Kanomax APM Model-3601.

References:
    Ehara et al. (1996) J. Aerosol Sci. doi:10.1016/0021-8502(96)00014-4
    Hinds, W. C. (1999). Aerosol Technology (2nd ed.). Wiley. Eq. (3.22)
"""
from __future__ import annotations

import numpy as np
from data_parser import MeasurementData

# ------------------------------------------------------------------------------
# Physical constants
# ------------------------------------------------------------------------------
E_CHARGE: float = 1.60219e-19   # Elementary charge [C]
AIR_VISC: float = 1.83e-5       # Dynamic viscosity of air [Pa·s]  (approx. 25°C)
ATM_PRES: float = 1.013e5       # Standard atmospheric pressure [Pa]


# ------------------------------------------------------------------------------
# Internal utilities
# ------------------------------------------------------------------------------

def _cunningham(Dmob: float) -> float:
    """Compute the Cunningham slip correction factor.

    Reference: Hinds, W. C. (1999). Aerosol Technology (2nd ed.). Wiley. Eq. (3.22)

    Args:
        Dmob: Electrical mobility diameter [m]

    Returns:
        Cc: Cunningham slip correction factor (dimensionless)
    """
    x = ATM_PRES * 1e-3 * Dmob * 1e6   # P [kPa] x d [µm]
    return 1.0 + (15.60 + 7.00 * np.exp(-0.059 * x)) / x


def _rk4_transmission(
    m:         float,
    V:         float,
    coef:      float,
    V_term:    float,
    omega:     float,
    rc:        float,
    delta:     float,
    r1:        float,
    r2:        float,
    nr0:       int,
    num_steps: int,
) -> float:
    """Integrate particle trajectories via RK4 and compute APM transmission efficiency.

    Tracks nr0 particles with initial positions spanning the APM gap (r1 <= r <= r2)
    and returns the flux-weighted fraction that traverse the full electrode length L
    without hitting the walls.

    Equation of motion (dr/dz):
        dr/dz = coef x (m*omega^2*r - eV/(r*ln(r2/r1))) / (1 - ((r-rc)/delta)^2)

    where coef = dz x 8/(9*eta) x (Cc/D_mob) x (delta*rc)/Q already includes dz,
    so each RK4 increment k_i directly gives a displacement [m].

    Flux weighting uses the parabolic velocity profile:
        weight(r0) = 1.5 x (1 - ((r0-rc)/delta)^2)

    Args:
        m:         Particle mass [kg]
        V:         Applied voltage [V]
        coef:      Common coefficient of dr/dz (includes dz) [m/step]
        V_term:    Electrostatic force term eV/ln(r2/r1) [N·m]
        omega:     APM angular velocity [rad/s]
        rc:        Gap centre radius [m]
        delta:     Gap half-width [m]
        r1:        Inner cylinder radius [m]
        r2:        Outer cylinder radius [m]
        nr0:       Number of initial particle positions
        num_steps: Number of RK4 integration steps

    Returns:
        Transmission efficiency (0.0 to 1.0)
    """
    r0      = np.linspace(r1 + 1e-5, r2 - 1e-5, nr0)
    r       = r0.copy()
    weights = 1.5 * (1.0 - ((r0 - rc) / delta) ** 2)
    total_weight = np.sum(weights)
    active  = np.ones(nr0, dtype=bool)

    def dr_dz(rad: np.ndarray) -> np.ndarray:
        return coef * (m * omega**2 * rad - V_term / rad) / (
            1.0 - ((rad - rc) / delta) ** 2
        )

    for _ in range(num_steps):
        if not np.any(active):
            break
        ra = r[active]
        k1 = dr_dz(ra)
        k2 = dr_dz(ra + 0.5 * k1)
        k3 = dr_dz(ra + 0.5 * k2)
        k4 = dr_dz(ra + k3)
        r[active] = ra + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        active[active] = np.abs(r[active] - rc) < (delta - 1e-5)

    return np.sum(weights[active]) / total_weight


# ------------------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------------------

def build_kernel_1d(
    data:   MeasurementData,
    params,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the 1D approximation kernel matrix K[I, J].

    Approximates the DMA output as a delta function and evaluates the kernel
    at a single mobility Z_p*:
        K[i, j] = Omega_APM(m_j, V_i, Z_p*) x Delta_m

    Args:
        data:   Binned measurement data (V_array, RPM, Dmob, I)
        params: User configuration (r1, r2, L, Q_a_lpm, dz, nr0,
                m_min_fg, m_max_fg, J)

    Returns:
        K:       Kernel matrix [I x J]
        m_array: Mass grid [kg], shape (J,)
    """
    rc    = 0.5 * (params.r1 + params.r2)
    delta = 0.5 * (params.r2 - params.r1)
    Cc    = _cunningham(data.Dmob)
    omega = data.RPM / 60.0 * 2.0 * np.pi
    Q     = params.Q_a_lpm * 1e-3 / 60.0           # [m^3/s]

    # Common coefficient of dr/dz: dz x 8/(9*eta) x (Cc/D_mob) x (delta*rc)/Q
    coef_base     = params.dz * 8.0 / (9.0 * AIR_VISC) * (Cc / data.Dmob) * (delta * rc) / Q
    V_term_factor = E_CHARGE / np.log(params.r2 / params.r1)
    num_steps     = int(params.L / params.dz)

    m_min   = params.m_min_fg * 1e-18
    m_max   = params.m_max_fg * 1e-18
    m_array = np.linspace(m_min, m_max, params.J)
    dm      = m_array[1] - m_array[0]

    I, J = data.I, params.J
    K    = np.zeros((I, J))

    print(f"Computing 1D kernel matrix K[{I}x{J}]...")
    for i, V in enumerate(data.V_array):
        V_term = V_term_factor * V
        for j, m in enumerate(m_array):
            K[i, j] = _rk4_transmission(
                m, V, coef_base, V_term, omega,
                rc, delta, params.r1, params.r2,
                params.nr0, num_steps,
            ) * dm
    print("Computation complete.")

    return K, m_array


def build_kernel_2d(
    data:   MeasurementData,
    params,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the 2D convolution kernel matrix K_eff[I, J].

    Computes the effective kernel by convolving the APM transfer function with
    the DMA triangular transfer function Omega_DMA(Z_p):
        K_eff[i,j] = [integral Omega_APM(m_j,V_i,Z_p) Omega_DMA(Z_p) dZ_p
                      / integral Omega_DMA dZ_p] x Delta_m

    The Z_p integral is discretised using params.num_Zp_bins quadrature points
    with the rectangle rule.

    Args:
        data:   Binned measurement data
        params: User configuration (beta, num_Zp_bins, plus same as 1D)

    Returns:
        K:       Kernel matrix [I x J]
        m_array: Mass grid [kg], shape (J,)
    """
    rc    = 0.5 * (params.r1 + params.r2)
    delta = 0.5 * (params.r2 - params.r1)
    Cc_star = _cunningham(data.Dmob)
    Zp_star = E_CHARGE * Cc_star / (3.0 * np.pi * AIR_VISC * data.Dmob)
    omega   = data.RPM / 60.0 * 2.0 * np.pi
    Q       = params.Q_a_lpm * 1e-3 / 60.0

    V_term_factor = E_CHARGE / np.log(params.r2 / params.r1)
    num_steps     = int(params.L / params.dz)

    # Discretise the DMA triangular transfer function
    Zp_array  = np.linspace(
        Zp_star * (1.0 - params.beta),
        Zp_star * (1.0 + params.beta),
        params.num_Zp_bins,
    )
    Omega_DMA = np.maximum(
        0.0,
        1.0 - np.abs(Zp_array - Zp_star) / (params.beta * Zp_star),
    )
    sum_Omega = np.sum(Omega_DMA)

    m_min   = params.m_min_fg * 1e-18
    m_max   = params.m_max_fg * 1e-18
    m_array = np.linspace(m_min, m_max, params.J)
    dm      = m_array[1] - m_array[0]

    I, J = data.I, params.J
    K    = np.zeros((I, J))

    print(f"Computing 2D convolution kernel matrix K_eff[{I}x{J}]...")
    for i, V in enumerate(data.V_array):
        V_term = V_term_factor * V
        for j, m in enumerate(m_array):
            K_eff_val = 0.0
            for zp, w_dma in zip(Zp_array, Omega_DMA):
                if w_dma > 0:
                    # Z_p-based coefficient: dz x (8*pi/3e) x Z_p x (delta*rc)/Q
                    # (using the relation Cc/D_mob = 3*pi*eta*Z_p/e)
                    coef_zp = (
                        params.dz * (8.0 / 3.0) * (np.pi * zp / E_CHARGE)
                        * (delta * rc) / Q
                    )
                    K_eff_val += _rk4_transmission(
                        m, V, coef_zp, V_term, omega,
                        rc, delta, params.r1, params.r2,
                        params.nr0, num_steps,
                    ) * w_dma
            K[i, j] = (K_eff_val / sum_Omega) * dm
    print("Computation complete.")

    return K, m_array
