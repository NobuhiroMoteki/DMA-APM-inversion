"""
visualization.py  ---  Gaussian fitting of the mass distribution and JPEG output

Fits the dominant mode with a Gaussian + linear background model and prints
the peak centre mu and standard deviation sigma to the console.
Output figures are saved as JPEG files at dpi=600.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from data_parser import MeasurementData

matplotlib.rcParams["font.family"] = "sans-serif"


# ------------------------------------------------------------------------------
# Gaussian fitting
# ------------------------------------------------------------------------------

@dataclass
class GaussianFitResult:
    """Result of a Gaussian fit to the dominant mode.

    Attributes:
        mu_fg:     Peak centre position [fg]
        sigma_fg:  Standard deviation [fg]
        amplitude: Peak amplitude A [cm^-3 fg^-1]
        offset:    Constant offset B [cm^-3 fg^-1]
        slope:     Linear slope C [cm^-3 fg^-2]
        r_squared: Coefficient of determination R^2
        success:   Flag indicating whether the fit succeeded
    """
    mu_fg:     float
    sigma_fg:  float
    amplitude: float
    offset:    float
    slope:     float
    r_squared: float
    success:   bool


def _gauss_linear(m: np.ndarray, A: float, mu: float, sigma: float,
                  B: float, C: float) -> np.ndarray:
    """Gaussian + linear background model.

    f(m) = A * exp(-(m - mu)^2 / (2*sigma^2)) + B + C*m
    """
    return A * np.exp(-0.5 * ((m - mu) / sigma) ** 2) + B + C * m


def fit_gaussian_mode(
    m_array:     np.ndarray,
    f_estimated: np.ndarray,
) -> GaussianFitResult:
    """Fit the dominant mode of the mass distribution with a Gaussian + linear background.

    Fit model:
        f(m) = A * exp(-(m - mu)^2 / (2*sigma^2)) + B + C*m

    Initial values: A = max(f),  mu = m[argmax(f)],  sigma = (m_max - m_min)/6,
                    B = C = 0

    If fitting fails, a warning is printed and a result with success=False is returned.

    Args:
        m_array:     Mass grid [kg], shape (J,)
        f_estimated: Estimated mass distribution [cm^-3 kg^-1], shape (J,)

    Returns:
        GaussianFitResult
    """
    m_fg = m_array * 1e18       # [kg] -> [fg]
    f_fg = f_estimated * 1e-18  # [cm^-3 kg^-1] -> [cm^-3 fg^-1]

    peak_idx = int(np.argmax(f_fg))
    A0     = float(f_fg[peak_idx])
    mu0    = float(m_fg[peak_idx])
    sigma0 = float((m_fg[-1] - m_fg[0]) / 6.0)

    try:
        popt, _ = curve_fit(
            _gauss_linear,
            m_fg,
            f_fg,
            p0=[A0, mu0, sigma0, 0.0, 0.0],
            bounds=(
                [0.0,    m_fg[0],  0.0,      -np.inf, -np.inf],
                [np.inf, m_fg[-1], m_fg[-1] - m_fg[0], np.inf, np.inf],
            ),
            maxfev=10000,
        )
        A, mu, sigma, B, C = popt
        sigma = abs(sigma)   # sigma must be positive

        f_fit  = _gauss_linear(m_fg, *popt)
        ss_res = float(np.sum((f_fg - f_fit) ** 2))
        ss_tot = float(np.sum((f_fg - np.mean(f_fg)) ** 2))
        r2     = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        result = GaussianFitResult(
            mu_fg=float(mu),
            sigma_fg=float(sigma),
            amplitude=float(A),
            offset=float(B),
            slope=float(C),
            r_squared=float(r2),
            success=True,
        )
        print("\n=== Gaussian Fit Result ===")
        print(f"  Peak centre  mu    = {mu:.2f} fg")
        print(f"  Std. dev.    sigma = {sigma:.2f} fg")
        print(f"  Amplitude    A     = {A:.4f} cm^-3 fg^-1")
        print(f"  Offset       B     = {B:.4f} cm^-3 fg^-1")
        print(f"  Slope        C     = {C:.6f} cm^-3 fg^-2")
        print(f"  R^2                = {r2:.4f}")
        return result

    except RuntimeError as exc:
        print(f"[Warning] Gaussian fit failed: {exc}")
        return GaussianFitResult(
            mu_fg=mu0, sigma_fg=sigma0, amplitude=A0,
            offset=0.0, slope=0.0, r_squared=0.0, success=False,
        )


# ------------------------------------------------------------------------------
# Plotting and saving
# ------------------------------------------------------------------------------

def plot_and_save(
    data:        MeasurementData,
    K:           np.ndarray,
    m_array:     np.ndarray,
    f_estimated: np.ndarray,
    fit_result:  GaussianFitResult,
    params,
    output_path: str,
) -> None:
    """Plot the results as a two-panel figure and save as JPEG (dpi=600).

    Left panel:  Measured APM spectrum (red dots) vs reconstructed signal (blue line)
    Right panel: Estimated mass distribution dN/dm (blue line) + Gaussian fit (red dashed)

    When the fit succeeds, the legend of the right panel shows the fit parameters
    (mu, sigma, R^2).

    Args:
        data:        Binned measurement data
        K:           Kernel matrix
        m_array:     Mass grid [kg]
        f_estimated: Estimated mass distribution [cm^-3 kg^-1]
        fit_result:  Gaussian fit result
        params:      User configuration
        output_path: Output file path (.jpg)
    """
    n_reconstructed = K @ f_estimated
    m_fg = m_array * 1e18        # [kg] -> [fg]
    f_fg = f_estimated * 1e-18   # [cm^-3 kg^-1] -> [cm^-3 fg^-1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # ---- Left panel: APM spectrum ----
    ax1.plot(data.V_array, data.n_meas, "ro", markersize=5,
             label="Measured (Binned Average)")
    ax1.plot(data.V_array, n_reconstructed, "b-", linewidth=2, alpha=0.8,
             label="Reconstructed Signal")
    ax1.set_xlabel("APM Voltage [V]")
    ax1.set_ylabel("Concentration [cm$^{-3}$]")
    ax1.set_title(
        f"APM Spectrum  "
        f"($D_{{mob}}$ = {data.Dmob * 1e9:.0f} nm,  RPM = {data.RPM:.0f})"
    )
    ax1.legend(fontsize=9)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # ---- Right panel: mass distribution ----
    ax2.plot(m_fg, f_fg, "b-", linewidth=2, label="Estimated Distribution")

    if fit_result.success:
        m_dense = np.linspace(m_fg[0], m_fg[-1], 500)
        f_fit   = _gauss_linear(
            m_dense,
            fit_result.amplitude,
            fit_result.mu_fg,
            fit_result.sigma_fg,
            fit_result.offset,
            fit_result.slope,
        )
        label_fit = (
            f"Gaussian Fit\n"
            f"$\\mu$ = {fit_result.mu_fg:.1f} fg\n"
            f"$\\sigma$ = {fit_result.sigma_fg:.1f} fg\n"
            f"$R^2$ = {fit_result.r_squared:.3f}"
        )
        ax2.plot(m_dense, f_fit, "r--", linewidth=1.5, label=label_fit)

    ax2.set_xlabel("Particle Mass [fg]")
    ax2.set_ylabel("Mass Distribution  $dN/dm$  [cm$^{-3}$ fg$^{-1}$]")
    ax2.set_title("Reconstructed Mass Distribution")
    ax2.legend(fontsize=9)
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()

    out_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(output_path, dpi=600, format="jpeg", bbox_inches="tight")
    plt.close()
    print(f"\nFigure saved: {output_path}")


def plot_transfer_function(
    m_arrays:    list[np.ndarray],
    Omegas:      list[np.ndarray],
    V_list:      list[float],
    m_c_list:    list[float],
    RPM:         float,
    Dmob:        float,
    output_path: str,
) -> None:
    """Plot the standalone APM transfer function and save as JPEG (dpi=600).

    Left panel:  Omega_APM versus particle mass [fg] for each voltage
                 (dotted vertical line: classifying mass m_c)
    Right panel: Omega_APM versus normalised mass m / m_c

    Args:
        m_arrays:    Mass grid for each voltage [kg], list of shape (J,)
        Omegas:      Transmission efficiency for each voltage, list of shape (J,)
        V_list:      Applied voltages [V]
        m_c_list:    Classifying mass for each voltage [kg]
        RPM:         APM rotation speed [rpm]
        Dmob:        Electrical mobility diameter [m]
        output_path: Output file path (.jpg)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    colors = plt.cm.viridis(np.linspace(0.0, 0.85, len(V_list)))

    for m_array, Omega, V, m_c, c in zip(m_arrays, Omegas, V_list, m_c_list, colors):
        label = f"V = {V:.0f} V  ($m_c$ = {m_c * 1e18:.2f} fg)"
        ax1.plot(m_array * 1e18, Omega, "-", color=c, linewidth=2, label=label)
        ax1.axvline(m_c * 1e18, color=c, linestyle=":", linewidth=1)
        ax2.plot(m_array / m_c, Omega, "-", color=c, linewidth=2, label=label)

    title = f"APM Transfer Function  ($D_{{mob}}$ = {Dmob * 1e9:.0f} nm,  RPM = {RPM:.0f})"
    ax1.set_xlabel("Particle Mass [fg]")
    ax1.set_ylabel(r"Transmission Efficiency  $\Omega_{APM}$")
    ax1.set_title(title)
    ax2.set_xlabel(r"Normalised Mass  $m / m_c$")
    ax2.set_ylabel(r"Transmission Efficiency  $\Omega_{APM}$")
    ax2.set_title("Normalised by Classifying Mass")
    for ax in (ax1, ax2):
        ax.set_ylim(0.0, 1.05)
        ax.legend(fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()

    out_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(output_path, dpi=600, format="jpeg", bbox_inches="tight")
    plt.close()
    print(f"\nFigure saved: {output_path}")


def plot_setpoint_transfer_functions(
    setpoints:   list,
    output_path: str,
    overlay:     str = "particle",
) -> None:
    """Plot APM transfer functions simulated at the set-points (JPEG, dpi=600).

    Left panel:  Omega_APM versus particle mass [fg]
                 (dotted vertical line: target particle mass m_p)
    Right panel: Omega_APM versus normalised mass m / m_p

    Args:
        setpoints:   List of apm_setpoint.APMSetpoint
        output_path: Output file path (.jpg)
        overlay:     "particle": curves are different particles at common
                     (Q_a, lambda); "lambda": curves are different lambda for
                     a common particle and Q_a
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    colors = plt.cm.viridis(np.linspace(0.0, 0.85, len(setpoints)))
    sp0    = setpoints[0]

    for sp, c in zip(setpoints, colors):
        setting = (
            f"RPM = {sp.RPM:.0f},  V = {sp.V:.1f} V,  "
            f"$m$/FWHM = {sp.m_p / sp.fwhm:.2f}"
        )
        if overlay == "lambda":
            label = f"$\\lambda$ = {sp.lam:.2f}:  {setting}"
        else:
            label = (
                f"$d$ = {sp.d * 1e9:.0f} nm, $\\rho_{{eff}}$ = {sp.rho_eff:.0f} kg m$^{{-3}}$"
                f"  ($m$ = {sp.m_p * 1e18:.2f} fg)\n{setting}"
            )
        ax1.plot(sp.m_array * 1e18, sp.Omega, "-", color=c, linewidth=2, label=label)
        ax2.plot(sp.m_array / sp.m_p, sp.Omega, "-", color=c, linewidth=2, label=label)
        if overlay != "lambda":
            ax1.axvline(sp.m_p * 1e18, color=c, linestyle=":", linewidth=1)
    if overlay == "lambda":
        ax1.axvline(sp0.m_p * 1e18, color="gray", linestyle=":", linewidth=1)
        title = (
            f"APM Transfer Function at Set-point  ($d$ = {sp0.d * 1e9:.0f} nm, "
            f"$\\rho_{{eff}}$ = {sp0.rho_eff:.0f} kg m$^{{-3}}$, $m$ = {sp0.m_p * 1e18:.2f} fg, "
            f"$Q_a$ = {sp0.Q_a_lpm:.2f} L/min)"
        )
    else:
        title = (
            f"APM Transfer Function at Set-point  "
            f"($Q_a$ = {sp0.Q_a_lpm:.2f} L/min, $\\lambda$ = {sp0.lam:.2f})"
        )

    ax1.set_xlabel("Particle Mass [fg]")
    ax1.set_ylabel(r"Transmission Efficiency  $\Omega_{APM}$")
    ax2.set_xlabel(r"Normalised Mass  $m / m_p$")
    ax2.set_ylabel(r"Transmission Efficiency  $\Omega_{APM}$")
    ax2.set_title("Normalised by Target Particle Mass", fontsize=10)
    fig.suptitle(title, fontsize=11)
    for ax in (ax1, ax2):
        ax.set_ylim(0.0, 1.05)
        ax.legend(fontsize=7)
        ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()

    out_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(output_path, dpi=600, format="jpeg", bbox_inches="tight")
    plt.close()
    print(f"\nFigure saved: {output_path}")


def plot_cluster_transfer_functions(
    setpoints:   list,
    clusters:    list,
    output_path: str,
) -> None:
    """Plot monomer and cluster transfer functions at common set-points (JPEG, dpi=600).

    Curves of the same colour share a set-point (typically different lambda);
    solid: singly charged monomer, dashed: multiply charged cluster.

    Left panel:  Omega_APM versus mass-to-charge ratio m/q [fg per e]
    Right panel: Omega_APM versus (m/q) / m_p

    Args:
        setpoints:   List of apm_setpoint.APMSetpoint (monomers)
        clusters:    List of apm_setpoint.ClusterTransfer (same order)
        output_path: Output file path (.jpg)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    colors = plt.cm.viridis(np.linspace(0.0, 0.85, len(setpoints)))
    sp0, cl0 = setpoints[0], clusters[0]

    for sp, cl, c in zip(setpoints, clusters, colors):
        mq_c = cl.m_array / cl.charge
        lab_m = (f"$\\lambda$ = {sp.lam:.2f}, monomer (q = 1):  RPM = {sp.RPM:.0f}, "
                 f"V = {sp.V:.1f} V")
        lab_c = (f"$\\lambda$ = {sp.lam:.2f}, {cl.n_mono}-sphere cluster (q = {cl.charge}):  "
                 f"$\\lambda_c$ = {cl.lam_eff:.2f}")
        for ax, x_m, x_c in (
            (ax1, sp.m_array * 1e18, mq_c * 1e18),
            (ax2, sp.m_array / sp.m_p, mq_c / sp.m_p),
        ):
            ax.plot(x_m, sp.Omega, "-",  color=c, linewidth=2,   label=lab_m)
            ax.plot(x_c, cl.Omega, "--", color=c, linewidth=1.8, label=lab_c)
    ax1.axvline(sp0.m_p * 1e18, color="gray", linestyle=":", linewidth=1)

    fig.suptitle(
        f"APM Transfer Function: Monomer and {cl0.n_mono}-sphere Cluster  "
        f"($d$ = {sp0.d * 1e9:.0f} nm, $\\rho_{{eff}}$ = {sp0.rho_eff:.0f} kg m$^{{-3}}$, "
        f"$Q_a$ = {sp0.Q_a_lpm:.2f} L/min, cluster $\\chi$ = {cl0.chi:.2f})",
        fontsize=11,
    )
    ax1.set_xlabel(r"Mass-to-charge Ratio  $m/q$  [fg $e^{-1}$]")
    ax2.set_xlabel(r"Normalised Mass-to-charge Ratio  $(m/q) / m_p$")
    ax2.set_title("Normalised by Monomer Mass", fontsize=10)
    for ax in (ax1, ax2):
        ax.set_ylabel(r"Transmission Efficiency  $\Omega_{APM}$")
        ax.set_ylim(0.0, 1.05)
        ax.legend(fontsize=6.5)
        ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()

    out_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(output_path, dpi=600, format="jpeg", bbox_inches="tight")
    plt.close()
    print(f"\nFigure saved: {output_path}")
