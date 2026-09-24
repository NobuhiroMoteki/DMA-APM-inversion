"""
apm_setpoint.py  ---  APM operating set-point for a spherical particle

Given a spherical particle (diameter d, effective density rho_eff), the APM
aerosol flow rate Q_a and the resolution parameter lambda, determines the
rotation speed and the applied voltage for which the mode of the simulated
APM transfer function coincides with the mass of the singly charged particle.

Rotation speed (definition of lambda, Ehara et al. 1996):
    lambda = 2 tau omega^2 L / v_bar
    tau    = m B,   B = Cc / (3 pi eta d),   m = rho_eff pi d^3 / 6
    v_bar  = Q_a / (pi (r2^2 - r1^2))
    ->  omega = sqrt(lambda v_bar / (2 tau L))

Applied voltage:
    The equation of motion depends on m and V only through
    (m omega^2 r - e V / (r ln(r2/r1))), so at fixed omega, D_mob and Q_a
    the transfer function is a function of m / V alone and its mode scales
    linearly with V. Starting from the force-balance voltage
        V0 = m omega^2 rc^2 ln(r2/r1) / e
    the mode m*_0 of Omega(m; V0) is located numerically and the voltage is
    rescaled to V = V0 m / m*_0, which places the mode exactly at m.

References:
    Ehara et al. (1996) J. Aerosol Sci. doi:10.1016/0021-8502(96)00014-4
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from kernel_simulator import (
    AIR_VISC,
    E_CHARGE,
    _cunningham,
    classifying_mass,
    compute_apm_transfer_function,
)


@dataclass
class APMSetpoint:
    """APM set-point for a singly charged spherical particle.

    Attributes:
        d:          Particle (= mobility) diameter [m]
        rho_eff:    Effective density [kg m^-3]
        Q_a_lpm:    APM aerosol flow rate [L/min]
        lam:        Resolution parameter lambda = 2 tau omega^2 L / v_bar (dimensionless)
        m_p:        Particle mass [kg]
        tau:        Particle relaxation time [s]
        RPM:        Rotation speed [rpm]
        V:          Applied voltage [V] (mode of Omega_APM at m_p)
        V_balance:  Force-balance voltage at the gap centre [V]
        m_array:    Mass grid of the verification simulation [kg], shape (N,)
        Omega:      Transfer function at (RPM, V), shape (N,)
        m_mode:     Mode of the simulated transfer function [kg]
        Omega_peak: Peak transmission efficiency
        fwhm:       Full width at half maximum [kg] (NaN if undefined)
        truncated:  True if Omega >= half maximum at an edge of the mass grid
    """
    d:          float
    rho_eff:    float
    Q_a_lpm:    float
    lam:        float
    m_p:        float
    tau:        float
    RPM:        float
    V:          float
    V_balance:  float
    m_array:    np.ndarray
    Omega:      np.ndarray
    m_mode:     float
    Omega_peak: float
    fwhm:       float
    truncated:  bool


def transfer_function_mode(
    m_array:   np.ndarray,
    Omega:     np.ndarray,
    plateau_frac: float = 0.99,
) -> float:
    """Mode of a (possibly flat-topped) transfer function.

    The mode is defined as the midpoint of the contiguous region around
    argmax(Omega) where Omega >= plateau_frac x max(Omega). This is robust
    against the flat top and the particle-sampling noise of the RK4 simulation.

    Args:
        m_array:      Mass grid [kg], shape (N,)
        Omega:        Transmission efficiency, shape (N,)
        plateau_frac: Relative threshold defining the top plateau

    Returns:
        m_mode: Mode [kg]
    """
    i_max = int(np.argmax(Omega))
    thr   = plateau_frac * Omega[i_max]
    lo, hi = i_max, i_max
    while lo > 0 and Omega[lo - 1] >= thr:
        lo -= 1
    while hi < len(Omega) - 1 and Omega[hi + 1] >= thr:
        hi += 1
    return 0.5 * (m_array[lo] + m_array[hi])


def _fwhm(m_array: np.ndarray, Omega: np.ndarray) -> tuple[float, bool]:
    """FWHM of Omega [kg] and whether Omega is truncated at the grid edge."""
    half      = 0.5 * Omega.max()
    above     = m_array[Omega >= half]
    fwhm      = above[-1] - above[0] if above.size > 1 else np.nan
    truncated = bool(max(Omega[0], Omega[-1]) >= half)
    return fwhm, truncated


def find_apm_setpoint(
    d:              float,
    rho_eff:        float,
    Q_a_lpm:        float,
    lam:            float,
    params,
    rel_halfwidth:  float = 1.0,
    num_m:          int   = 401,
) -> APMSetpoint:
    """Determine the APM rotation speed and voltage for a spherical particle.

    Args:
        d:             Particle (= mobility) diameter [m]
        rho_eff:       Effective density [kg m^-3]
        Q_a_lpm:       APM aerosol flow rate [L/min]
        lam:           Resolution parameter lambda = 2 tau omega^2 L / v_bar
        params:        User configuration (r1, r2, L, dz, nr0)
        rel_halfwidth: Mass grid spans m_p x (1 +/- rel_halfwidth), lower
                       bound >= 0.01 m_p
        num_m:         Number of mass grid points

    Returns:
        APMSetpoint including the transfer function simulated at the set-point
    """
    m_p   = rho_eff * np.pi * d**3 / 6.0                 # [kg]
    B     = _cunningham(d) / (3.0 * np.pi * AIR_VISC * d)  # [s kg^-1]
    tau   = m_p * B                                      # [s]
    Q     = Q_a_lpm * 1e-3 / 60.0                        # [m^3/s]
    v_bar = Q / (np.pi * (params.r2**2 - params.r1**2))  # [m/s]
    omega = np.sqrt(lam * v_bar / (2.0 * tau * params.L))  # [rad/s]
    RPM   = omega * 60.0 / (2.0 * np.pi)

    # Force-balance voltage: m_c(V0) = m_p
    V_balance = m_p / classifying_mass(1.0, RPM, params)

    m_array = np.linspace(
        m_p * max(1.0 - rel_halfwidth, 0.01),
        m_p * (1.0 + rel_halfwidth),
        num_m,
    )   # shape: (num_m,) [kg]

    # Step 1: locate the mode at V0 and rescale V (Omega depends on m/V only)
    Omega0 = compute_apm_transfer_function(m_array, V_balance, RPM, d, params, Q_a_lpm)
    V      = V_balance * m_p / transfer_function_mode(m_array, Omega0)

    # Step 2: simulate at the set-point (verification + output)
    Omega  = compute_apm_transfer_function(m_array, V, RPM, d, params, Q_a_lpm)
    fwhm, truncated = _fwhm(m_array, Omega)

    return APMSetpoint(
        d=d, rho_eff=rho_eff, Q_a_lpm=Q_a_lpm, lam=lam,
        m_p=m_p, tau=tau, RPM=RPM, V=V, V_balance=V_balance,
        m_array=m_array, Omega=Omega,
        m_mode=transfer_function_mode(m_array, Omega),
        Omega_peak=float(Omega.max()),
        fwhm=fwhm, truncated=truncated,
    )


@dataclass
class ClusterTransfer:
    """Transfer function of a charged sphere cluster at a monomer set-point.

    Attributes:
        n_mono:     Number of primary spheres in the cluster
        charge:     Number of elementary charges q
        chi:        Dynamic shape factor (relative to the volume-equivalent sphere)
        d_ve:       Volume-equivalent diameter n^(1/3) d [m]
        d_m:        Mobility-equivalent diameter chi d_ve [m]
        mass:       Cluster mass n m_p [kg]
        lam_eff:    Resolution parameter of the cluster 2 (mass B_c) omega^2 L / v_bar
        Z_ratio:    Electrical mobility relative to the singly charged monomer
        m_array:    Cluster mass grid [kg], shape (N,)
        Omega:      Transfer function at the monomer set-point, shape (N,)
        mq_mode:    Mode of Omega in mass-to-charge ratio m/q [kg per elementary charge]
        Omega_peak: Peak transmission efficiency
        fwhm_mq:    FWHM in m/q [kg per elementary charge] (NaN if undefined)
        truncated:  True if Omega >= half maximum at an edge of the mass grid
    """
    n_mono:     int
    charge:     int
    chi:        float
    d_ve:       float
    d_m:        float
    mass:       float
    lam_eff:    float
    Z_ratio:    float
    m_array:    np.ndarray
    Omega:      np.ndarray
    mq_mode:    float
    Omega_peak: float
    fwhm_mq:    float
    truncated:  bool


def simulate_cluster(
    sp:            APMSetpoint,
    params,
    n_mono:        int   = 2,
    charge:        int   = 2,
    chi:           float = 1.12,
    rel_halfwidth: float = 1.0,
    num_m:         int   = 401,
) -> ClusterTransfer:
    """Transfer function of a randomly oriented, q-fold charged n-sphere cluster.

    The cluster consists of n_mono primary spheres identical to the set-point
    particle (diameter d, density rho_eff). Its mobility is approximated by
        d_ve = n^(1/3) d,   d_m = chi d_ve,   B_c = Cc(d_m) / (3 pi eta d_m),
    with chi the orientation-averaged dynamic shape factor (continuum value;
    Hinds, 1999, Table 3.2: 1.12 for a two-sphere chain).

    With charge q e the equation of motion reads
        dr/dz ~ B_c (M omega^2 r - q e V / (r ln(r2/r1))),
    so the monomer code is used with voltage q V and mobility diameter d_m.
    The cluster passes the APM at the monomer set-point when M/q = m_p.

    Args:
        sp:            Monomer set-point (RPM, V, Q_a from find_apm_setpoint)
        params:        User configuration (r1, r2, L, dz, nr0)
        n_mono:        Number of primary spheres
        charge:        Number of elementary charges
        chi:           Dynamic shape factor of the cluster
        rel_halfwidth: m/q grid spans m_p x (1 +/- rel_halfwidth), lower bound >= 0.01 m_p
        num_m:         Number of mass grid points

    Returns:
        ClusterTransfer
    """
    d_ve = n_mono ** (1.0 / 3.0) * sp.d
    d_m  = chi * d_ve
    M    = n_mono * sp.m_p                                   # [kg]
    B_1  = _cunningham(sp.d) / (3.0 * np.pi * AIR_VISC * sp.d)
    B_c  = _cunningham(d_m) / (3.0 * np.pi * AIR_VISC * d_m)
    omega   = sp.RPM / 60.0 * 2.0 * np.pi
    v_bar   = sp.Q_a_lpm * 1e-3 / 60.0 / (np.pi * (params.r2**2 - params.r1**2))
    lam_eff = 2.0 * M * B_c * omega**2 * params.L / v_bar

    mq_array = np.linspace(
        sp.m_p * max(1.0 - rel_halfwidth, 0.01),
        sp.m_p * (1.0 + rel_halfwidth),
        num_m,
    )   # shape: (num_m,) [kg per elementary charge]
    m_array = charge * mq_array
    Omega   = compute_apm_transfer_function(
        m_array, charge * sp.V, sp.RPM, d_m, params, sp.Q_a_lpm,
    )
    fwhm_mq, truncated = _fwhm(mq_array, Omega)

    return ClusterTransfer(
        n_mono=n_mono, charge=charge, chi=chi, d_ve=d_ve, d_m=d_m, mass=M,
        lam_eff=lam_eff, Z_ratio=charge * B_c / B_1,
        m_array=m_array, Omega=Omega,
        mq_mode=transfer_function_mode(mq_array, Omega),
        Omega_peak=float(Omega.max()),
        fwhm_mq=fwhm_mq, truncated=truncated,
    )
