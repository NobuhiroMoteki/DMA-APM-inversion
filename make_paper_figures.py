"""
make_paper_figures.py  ---  3x2 panel PDF figures for FS and JetA1

Reads the NPZ dumps produced by run_batch_inversion.py and assembles
publication-quality 3x2 panel figures (rows = Dmob 450/500/550 nm;
cols = APM spectrum / mass distribution) as vector PDFs.

The plotting parameters (font size, panel spacing, legend placement)
are kept in this file so they can be tweaked without re-running the
inversion.
"""
from __future__ import annotations

import os
from typing import Sequence

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# --- Paper-quality defaults ---------------------------------------------------
matplotlib.rcParams["pdf.fonttype"]   = 42   # embed TrueType (editable in Illustrator)
matplotlib.rcParams["ps.fonttype"]    = 42
matplotlib.rcParams["font.family"]    = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
matplotlib.rcParams["font.size"]      = 9
matplotlib.rcParams["axes.labelsize"] = 10
matplotlib.rcParams["axes.titlesize"] = 10
matplotlib.rcParams["xtick.labelsize"] = 8
matplotlib.rcParams["ytick.labelsize"] = 8
matplotlib.rcParams["legend.fontsize"] = 7.5
matplotlib.rcParams["axes.linewidth"]  = 0.8
matplotlib.rcParams["lines.linewidth"] = 1.4
matplotlib.rcParams["xtick.direction"]  = "in"
matplotlib.rcParams["ytick.direction"]  = "in"
matplotlib.rcParams["xtick.major.size"] = 3.5
matplotlib.rcParams["ytick.major.size"] = 3.5
matplotlib.rcParams["xtick.minor.size"] = 2.0
matplotlib.rcParams["ytick.minor.size"] = 2.0
matplotlib.rcParams["xtick.top"]        = True   # mirror ticks on the top
matplotlib.rcParams["ytick.right"]      = True   # mirror ticks on the right

DATA_DIR = "./results/data"
OUT_DIR  = "./results/paper_figures"

GROUPS: dict[str, list[tuple[str, int]]] = {
    "FS": [
        ("FS_Dmob450nm_try2__20050101014000", 450),
        ("FS_Dmob500nm_try1__20050101024557", 500),
        ("FS_Dmob550nm_try1__20050101015918", 550),
    ],
    "JetA1": [
        ("JetA1_Dmob450nm_try1__20050101030343", 450),
        ("JetA1_Dmob500nm_try1__20050101032008", 500),
        ("JetA1_Dmob550nm_try1__20050101033753", 550),
    ],
}

PANEL_LETTERS = list("abcdef")


def _gauss_linear(m: np.ndarray, A: float, mu: float, sigma: float,
                  B: float, C: float) -> np.ndarray:
    return A * np.exp(-0.5 * ((m - mu) / sigma) ** 2) + B + C * m


def _load_group(items: Sequence[tuple[str, int]]) -> list[dict]:
    rows: list[dict] = []
    for stem, dmob_label in items:
        npz = np.load(os.path.join(DATA_DIR, f"{stem}.npz"), allow_pickle=False)
        d = {
            "V"           : npz["V_array"],
            "n"           : npz["n_meas"],
            "K"           : npz["K"],
            "m_arr"       : npz["m_array"],
            "f_est"       : npz["f_estimated"],
            "rpm"         : float(npz["RPM"]),
            "dmob_label"  : dmob_label,
            "fit_success" : bool(npz["fit_success"]),
            "mu"          : float(npz["fit_mu_fg"]),
            "sigma"       : float(npz["fit_sigma_fg"]),
            "A"           : float(npz["fit_amplitude"]),
            "B"           : float(npz["fit_offset"]),
            "C"           : float(npz["fit_slope"]),
            "r2"          : float(npz["fit_r_squared"]),
        }
        d["n_rec"] = d["K"] @ d["f_est"]
        d["m_fg"]  = d["m_arr"] * 1e18
        d["f_fg"]  = d["f_est"] * 1e-18
        rows.append(d)
    return rows


def _group_max(rows: list[dict]) -> tuple[float, float]:
    """Return (max concentration, max dN/dm) for a loaded group."""
    n_max = max(
        float(max(np.max(d["n"]), np.max(d["n_rec"]))) for d in rows
    )
    f_max = 0.0
    for d in rows:
        f_curr = float(np.max(d["f_fg"]))
        if d["fit_success"]:
            m_dense = np.linspace(d["m_fg"][0], d["m_fg"][-1], 600)
            f_fit   = _gauss_linear(
                m_dense, d["A"], d["mu"], d["sigma"], d["B"], d["C"],
            )
            f_curr = max(f_curr, float(np.max(f_fit)))
        f_max = max(f_max, f_curr)
    return n_max, f_max


def _make_panel_figure(
    group_label: str,
    rows:        list[dict],
    output_path: str,
    n_max:       float,
    f_max:       float,
) -> None:
    # Vertical headroom factors above the unified maxima.
    HEADROOM_LEFT  = 1.45   # APM spectrum
    HEADROOM_RIGHT = 1.70   # mass distribution (room for Gaussian-fit legend)

    # Common x-axis ranges.
    XLIM_V_LEFT  = (200.0, 1800.0)
    XLIM_M_RIGHT = (0.0,   90.0)

    YLIM_LEFT  = (0.0, n_max * HEADROOM_LEFT)
    YLIM_RIGHT = (0.0, f_max * HEADROOM_RIGHT)

    # ---- Build the figure -------------------------------------------------
    fig, axes = plt.subplots(
        nrows=3, ncols=2,
        figsize=(7.6, 8.4),
        sharex="col",   # x ticks shown only on the bottom row of each column
        sharey="col",   # y range unified within each column
    )

    for row, d in enumerate(rows):
        # ---- Left column: APM spectrum ------------------------------------
        ax1 = axes[row, 0]
        ax1.plot(d["V"], d["n"], "o", mfc="#d62728", mec="#7a1414",
                 ms=4.5, mew=0.6, label="Measured")
        ax1.plot(d["V"], d["n_rec"], "-", color="#1f77b4",
                 lw=1.6, alpha=0.9, label="Reconstructed")
        ax1.set_ylabel(r"Concentration [cm$^{-3}$]")
        ax1.legend(loc="upper right", frameon=False)
        ax1.text(0.035, 0.955, f"({PANEL_LETTERS[2 * row]})",
                 transform=ax1.transAxes, va="top", ha="left",
                 fontsize=10, fontweight="bold")
        ax1.text(0.035, 0.865,
                 f"$\\omega$ = {d['rpm']:.0f} rpm",
                 transform=ax1.transAxes, va="top", ha="left",
                 fontsize=8.5)
        if row == 2:
            ax1.set_xlabel("APM voltage [V]")

        # ---- Right column: mass distribution ------------------------------
        ax2 = axes[row, 1]
        ax2.plot(d["m_fg"], d["f_fg"], "-", color="#1f77b4",
                 lw=1.6, label=r"Estimated $dN/dm$")
        if d["fit_success"]:
            m_dense = np.linspace(d["m_fg"][0], d["m_fg"][-1], 600)
            f_fit   = _gauss_linear(
                m_dense, d["A"], d["mu"], d["sigma"], d["B"], d["C"],
            )
            label_fit = (
                f"Gaussian fit\n"
                f"$\\mu$ = {d['mu']:.1f} fg\n"
                f"$\\sigma$ = {d['sigma']:.1f} fg\n"
                f"$R^2$ = {d['r2']:.3f}"
            )
            ax2.plot(m_dense, f_fit, "--", color="#d62728",
                     lw=1.2, label=label_fit)
        ax2.set_ylabel(r"$dN/dm$ [cm$^{-3}$ fg$^{-1}$]")
        ax2.legend(loc="upper right", frameon=False)
        ax2.text(0.035, 0.955, f"({PANEL_LETTERS[2 * row + 1]})",
                 transform=ax2.transAxes, va="top", ha="left",
                 fontsize=10, fontweight="bold")
        if row == 2:
            ax2.set_xlabel("Particle mass [fg]")

    # Unified ranges propagate via sharex='col' / sharey='col'.
    axes[0, 0].set_xlim(*XLIM_V_LEFT)
    axes[0, 0].set_ylim(*YLIM_LEFT)
    axes[0, 1].set_xlim(*XLIM_M_RIGHT)
    axes[0, 1].set_ylim(*YLIM_RIGHT)

    fig.tight_layout(rect=[0.05, 0, 1, 1], h_pad=1.0, w_pad=1.6)

    # Row labels (Dmob common across each row), placed at the figure's left
    # edge using the post-layout axes positions of the leftmost column.
    for row, d in enumerate(rows):
        bbox = axes[row, 0].get_position()
        fig.text(
            x=0.015,
            y=bbox.y0 + bbox.height / 2.0,
            s=f"$D_\\mathrm{{mob}}$ = {d['dmob_label']} nm",
            rotation=90, ha="center", va="center", fontsize=11,
        )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def main() -> None:
    # Load every NPZ once and compute per-group maxes so that we can apply
    # cross-group unified axis ranges:
    #   concentration max -> JetA1's max (the larger of the two groups)
    #   dN/dm max         -> FS's max    (the smaller of the two groups)
    loaded = {name: _load_group(items) for name, items in GROUPS.items()}
    maxes  = {name: _group_max(rows)   for name, rows in loaded.items()}

    n_unified = maxes["JetA1"][0]
    f_unified = maxes["FS"][1]

    for name, rows in loaded.items():
        out_path = os.path.join(OUT_DIR, f"panel_{name}_3x2.pdf")
        _make_panel_figure(name, rows, out_path, n_unified, f_unified)


if __name__ == "__main__":
    main()
