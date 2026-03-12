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
