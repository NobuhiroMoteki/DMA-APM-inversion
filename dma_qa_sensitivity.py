"""
dma_qa_sensitivity.py  ---  DMA-only sample-flow sensitivity figure

Question:
    For a fixed DMA sheath flow (3 L/min) and the same source aerosol,
    how does the DMA-outlet mass distribution change when the DMA sample
    flow is changed from 0.30 to 0.43 L/min, given that the inlet
    number-concentration size dependence is negligible within the DMA
    classification window?

Analysis:
    The DMA outlet mass distribution is
        f_out(m; beta) = int n_in(D_mob) h(m|D_mob) Omega(D_mob; D_mob*, beta) dD_mob
    Under the stated assumption n_in(D_mob) ~ const within the window, but
    the conditional mass distribution h(m|D_mob) still depends on D_mob
    (fractal aerosols: <m> ~ D_mob^D_f), so widening the mobility window
    broadens f_out as well.

    Magnitudes (second-moment addition, Gaussian approximation):
        sigma_m^2(beta) ~ sigma_intrinsic^2 + sigma_DMA^2(beta)
        sigma_DMA(beta) ~ D_f * <m> * beta / sqrt(6)        (triangle std)
        N_out         ~ proportional to beta                (unchanged)

    D_f is estimated from a log-log regression of Gaussian-fit centres mu
    versus the three nominal D_mob values across all six datasets.

    The predicted f_out at beta = 0.43/3 is obtained by Gaussian-convolving
    the reference f_out (beta = 0.30/3) with the *increment*
        Delta sigma = sqrt(sigma_DMA(beta_new)^2 - sigma_DMA(beta_old)^2)
    and rescaling the integral by beta_new / beta_old (= 1.433).
"""
from __future__ import annotations

import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d


# --- Plot defaults (paper-quality) -----------------------------------------
matplotlib.rcParams["pdf.fonttype"]    = 42
matplotlib.rcParams["ps.fonttype"]     = 42
matplotlib.rcParams["font.family"]     = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
matplotlib.rcParams["font.size"]       = 9
matplotlib.rcParams["axes.labelsize"]  = 10
matplotlib.rcParams["xtick.labelsize"] = 8
matplotlib.rcParams["ytick.labelsize"] = 8
matplotlib.rcParams["legend.fontsize"] = 7.5
matplotlib.rcParams["axes.linewidth"]  = 0.8
matplotlib.rcParams["lines.linewidth"] = 1.4
matplotlib.rcParams["xtick.direction"] = "in"
matplotlib.rcParams["ytick.direction"] = "in"
matplotlib.rcParams["xtick.top"]       = True
matplotlib.rcParams["ytick.right"]     = True
matplotlib.rcParams["xtick.major.size"] = 3.5
matplotlib.rcParams["ytick.major.size"] = 3.5

DATA_DIR = "./results/data"
OUT_PATH = "./results/paper_figures/dma_qa_sensitivity_3x2.pdf"

Q_REF  = 0.30   # L/min, reference sample flow
Q_NEW  = 0.43   # L/min, perturbed sample flow
Q_SH   = 3.0    # L/min, fixed sheath flow
SCALE  = Q_NEW / Q_REF   # = 1.4333...

# Same six datasets as in the main figures, organised as
# axes[row, col] -> (Dmob, sample-group)
COLUMNS = ["FS", "JetA1"]
DMOBS   = [450, 500, 550]
ROWS = {
    ("FS", 450): "FS_Dmob450nm_try2__20050101014000",
    ("FS", 500): "FS_Dmob500nm_try1__20050101024557",
    ("FS", 550): "FS_Dmob550nm_try1__20050101015918",
    ("JetA1", 450): "JetA1_Dmob450nm_try1__20050101030343",
    ("JetA1", 500): "JetA1_Dmob500nm_try1__20050101032008",
    ("JetA1", 550): "JetA1_Dmob550nm_try1__20050101033753",
}

PANEL_LETTERS = list("abcdef")

COLOR_REF = "#1f77b4"   # blue   -- reference (q_sample = 0.30)
COLOR_NEW = "#2ca02c"   # green  -- predicted (q_sample = 0.43)


def _load(stem: str) -> dict:
    npz = np.load(os.path.join(DATA_DIR, f"{stem}.npz"), allow_pickle=False)
    return {
        "m_fg" : npz["m_array"]     * 1e18,
        "f_fg" : npz["f_estimated"] * 1e-18,
        "mu"   : float(npz["fit_mu_fg"]),
        "sigma": float(npz["fit_sigma_fg"]),
        "rpm"  : float(npz["RPM"]),
    }


def _estimate_Df(records: dict[tuple[str, int], dict]) -> float:
    """Log-log regression of Gaussian-fit mu vs nominal D_mob: m = a * D_mob^Df."""
    log_d  = np.log([dmob for (_, dmob), _ in records.items()])
    log_mu = np.log([rec["mu"] for rec in records.values()])
    Df, _  = np.polyfit(log_d, log_mu, 1)
    return float(Df)


def _broaden_and_scale(
    m_fg:  np.ndarray,
    f_fg:  np.ndarray,
    delta_sigma_fg: float,
    scale: float,
) -> np.ndarray:
    """Convolve f(m) with a Gaussian of std = delta_sigma_fg and rescale by `scale`.

    The mass grid is assumed uniformly spaced (true for np.linspace-generated
    arrays in this code base).
    """
    dm_fg            = float(m_fg[1] - m_fg[0])
    sigma_in_samples = delta_sigma_fg / dm_fg
    return scale * gaussian_filter1d(
        f_fg, sigma=sigma_in_samples, mode="constant", cval=0.0,
    )


def main() -> None:
    # ---- 1. Load every dataset and estimate the mass-mobility exponent ----
    records = {key: _load(stem) for key, stem in ROWS.items()}
    Df      = _estimate_Df(records)
    print(f"Mass-mobility scaling fit:  m proportional to D_mob^{Df:.3f}")

    BETA_OLD = Q_REF / Q_SH
    BETA_NEW = Q_NEW / Q_SH
    print(f"beta_old = {BETA_OLD:.4f},  beta_new = {BETA_NEW:.4f},  "
          f"N-scale = {SCALE:.3f}")

    # ---- 2. Build the figure ------------------------------------------------
    fig, axes = plt.subplots(
        nrows=3, ncols=2,
        figsize=(7.6, 8.4),
        sharex="all", sharey="all",
    )

    # Common y-axis upper bound: max of (reference, predicted) across all panels.
    f_max = 0.0
    for (sample, dmob), rec in records.items():
        sigma_dma_old = Df * rec["mu"] * BETA_OLD / np.sqrt(6.0)
        sigma_dma_new = Df * rec["mu"] * BETA_NEW / np.sqrt(6.0)
        delta_sigma   = float(np.sqrt(sigma_dma_new**2 - sigma_dma_old**2))
        f_predict     = _broaden_and_scale(
            rec["m_fg"], rec["f_fg"], delta_sigma, SCALE,
        )
        f_max = max(f_max, float(np.max(rec["f_fg"])), float(np.max(f_predict)))
    YLIM = (0.0, f_max * 1.50)
    XLIM = (0.0, 90.0)

    for col_idx, sample in enumerate(COLUMNS):
        for row_idx, dmob in enumerate(DMOBS):
            ax  = axes[row_idx, col_idx]
            rec = records[(sample, dmob)]
            m_fg, f_fg, mu = rec["m_fg"], rec["f_fg"], rec["mu"]

            sigma_dma_old = Df * mu * BETA_OLD / np.sqrt(6.0)
            sigma_dma_new = Df * mu * BETA_NEW / np.sqrt(6.0)
            delta_sigma   = float(np.sqrt(sigma_dma_new**2 - sigma_dma_old**2))
            f_predict     = _broaden_and_scale(m_fg, f_fg, delta_sigma, SCALE)

            ax.plot(m_fg, f_fg, "-",
                    color=COLOR_REF, lw=1.6,
                    label=f"$q_\\mathrm{{sample}}$ = {Q_REF:.2f} L/min  (reference)")
            ax.plot(m_fg, f_predict, "--",
                    color=COLOR_NEW, lw=1.6,
                    label=(f"$q_\\mathrm{{sample}}$ = {Q_NEW:.2f} L/min  "
                           f"($\\Delta\\sigma$ = {delta_sigma:.2f} fg, "
                           f"$\\times{SCALE:.2f}$)"))

            ax.legend(loc="upper right", frameon=False)
            panel_idx = row_idx * 2 + col_idx
            ax.text(0.035, 0.955, f"({PANEL_LETTERS[panel_idx]})",
                    transform=ax.transAxes, va="top", ha="left",
                    fontsize=10, fontweight="bold")

            if row_idx == 0:
                ax.set_title(f"{sample}", fontsize=10, pad=4)
            if row_idx == 2:
                ax.set_xlabel("Particle mass [fg]")
            if col_idx == 0:
                ax.set_ylabel(r"$dN/dm$ [cm$^{-3}$ fg$^{-1}$]")

    axes[0, 0].set_xlim(*XLIM)
    axes[0, 0].set_ylim(*YLIM)

    fig.tight_layout(rect=[0.05, 0.0, 1.0, 0.95], h_pad=1.0, w_pad=1.6)

    # Row labels (Dmob).
    for row_idx, dmob in enumerate(DMOBS):
        bbox = axes[row_idx, 0].get_position()
        fig.text(
            x=0.015,
            y=bbox.y0 + bbox.height / 2.0,
            s=f"$D_\\mathrm{{mob}}$ = {dmob} nm",
            rotation=90, ha="center", va="center", fontsize=11,
        )

    fig.suptitle(
        f"DMA outlet $f(m)$ at $q_\\mathrm{{sheath}}$ = {Q_SH:.1f} L/min fixed, "
        f"$q_\\mathrm{{sample}}$: {Q_REF:.2f} $\\rightarrow$ {Q_NEW:.2f} L/min   "
        f"($\\beta$: {BETA_OLD:.3f} $\\rightarrow$ {BETA_NEW:.3f}; "
        f"$D_f$ = {Df:.2f} from data)",
        fontsize=10, y=0.995,
    )

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    fig.savefig(OUT_PATH, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
