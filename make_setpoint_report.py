"""
make_setpoint_report.py  ---  LaTeX/PDF report of APM set-point calculations

Reads the set-point summary (apm_setpoint_summary.csv) and the figures
produced by run_apm_setpoint.py in params.SP_OUTPUT_DIR, writes a LaTeX
report (calculation conditions, method, result table, figures) to
<SP_OUTPUT_DIR>/apm_setpoint_report.tex and builds the PDF with latexmk.

Usage:
    python run_apm_setpoint.py        # first, to produce the results
    python make_setpoint_report.py
"""
from __future__ import annotations

import csv
import datetime
import os
import subprocess

import numpy as np

import params
from kernel_simulator import AIR_VISC, ATM_PRES, E_CHARGE

REPORT_STEM = "apm_setpoint_report"


def _read_summary(path: str) -> list[dict]:
    """Read apm_setpoint_summary.csv into a list of dicts (numeric values as float)."""
    with open(path, encoding="utf-8") as f:
        return [{k: float(v) for k, v in row.items()} for row in csv.DictReader(f)]


def _figure_files(lambdas: list[float]) -> list[tuple[str, str]]:
    """File names (relative to SP_OUTPUT_DIR) and captions of the figures."""
    Q_a = params.SP_Q_a_lpm
    if len(lambdas) == 1:
        name = f"apm_setpoint_Q{Q_a:.2f}lpm_lam{lambdas[0]:.2f}.jpg"
        cap  = (f"APM transfer functions at the set-points for all particles "
                f"($Q_a = \\SI{{{Q_a}}}{{\\liter\\per\\minute}}$, "
                f"$\\lambda = {lambdas[0]}$).")
        return [(name, cap)]
    lam_tag = "-".join(f"{lam:.2f}" for lam in lambdas)
    lam_str = ", ".join(f"{lam}" for lam in lambdas)
    out = []
    for d_nm, rho_eff in params.SP_particles:
        name = f"apm_setpoint_d{d_nm:.0f}nm_rho{rho_eff:.0f}_Q{Q_a:.2f}lpm_lam{lam_tag}.jpg"
        cap  = (
            f"APM transfer function at the set-point for "
            f"$d = \\SI{{{d_nm:g}}}{{\\nano\\meter}}$, "
            f"$\\rho_\\text{{eff}} = \\SI{{{rho_eff / 1000:g}}}{{\\gram\\per\\centi\\meter\\cubed}}$ "
            f"($Q_a = \\SI{{{Q_a}}}{{\\liter\\per\\minute}}$; $\\lambda = {lam_str}$ overlaid). "
            f"Left: $\\Omega_\\text{{APM}}$ versus particle mass (dotted line: $m_p$); "
            f"right: versus $m/m_p$."
        )
        out.append((name, cap))
    return out


def _cluster_section(lambdas: list[float], out_dir: str) -> str:
    """LaTeX section on the multiply charged cluster (empty if not computed)."""
    cfg  = getattr(params, "SP_cluster", None)
    path = os.path.join(out_dir, "apm_cluster_summary.csv")
    if not cfg or not os.path.exists(path):
        return ""
    Q_a  = params.SP_Q_a_lpm
    n, q = cfg["n_mono"], cfg["charge"]
    tag  = f"apm_cluster{n}q{q}"

    lines, figs = [], []
    for r in _read_summary(path):
        d_nm, rho, lam = r["d_nm"], r["rho_eff_kg_m3"], r["lambda"]
        mono = np.loadtxt(os.path.join(
            out_dir, f"apm_setpoint_d{d_nm:.0f}nm_rho{rho:.0f}_Q{Q_a:.2f}lpm_lam{lam:.2f}.csv"),
            delimiter=",")   # shape: (N, 2): mass_fg, Omega
        clus = np.loadtxt(os.path.join(
            out_dir, f"{tag}_d{d_nm:.0f}nm_rho{rho:.0f}_Q{Q_a:.2f}lpm_lam{lam:.2f}.csv"),
            delimiter=",")   # shape: (N, 3): cluster_mass_fg, m/q_fg, Omega
        m_p      = rho * np.pi * (d_nm * 1e-9) ** 3 / 6.0 * 1e18      # [fg]
        om_1     = np.interp(m_p, mono[:, 0], mono[:, 1])
        om_c     = np.interp(m_p, clus[:, 1], clus[:, 2])
        area_rat = np.trapezoid(clus[:, 2], clus[:, 1]) / np.trapezoid(mono[:, 1], mono[:, 0])
        mono_fwhm = next(
            s["FWHM_fg"] for s in _read_summary(os.path.join(out_dir, "apm_setpoint_summary.csv"))
            if s["d_nm"] == d_nm and s["lambda"] == lam
        )
        lines.append(
            f"  \\num{{{d_nm:g}}} & \\num{{{lam:g}}} & \\num{{{r['d_m_nm']:.1f}}} & "
            f"\\num{{{r['lambda_c']:.3f}}} & \\num{{{r['mode_mq_fg'] / m_p - 1.0:+.4f}}} & "
            f"\\num{{{om_1:.3f}}} & \\num{{{om_c:.3f}}} & \\num{{{om_c / om_1:.3f}}} & "
            f"\\num{{{mono_fwhm:.2f}}} & \\num{{{r['FWHM_mq_fg']:.2f}}} & "
            f"\\num{{{area_rat:.3f}}} \\\\"
        )
    for i, (d_nm, rho) in enumerate(params.SP_particles):
        name = f"{tag}_d{d_nm:.0f}nm_rho{rho:.0f}_Q{Q_a:.2f}lpm.jpg"
        if not os.path.exists(os.path.join(out_dir, name)):
            raise FileNotFoundError(f"{name} not found in {out_dir}; run run_apm_setpoint.py first.")
        figs.append(
            "\\begin{figure}[htbp]\n  \\centering\n"
            f"  \\includegraphics[width=\\textwidth]{{{name}}}\n"
            f"  \\caption{{Transfer functions of the singly charged monomer (solid) and the "
            f"{n}-sphere cluster with {q} charges (dashed) at the monomer set-points for "
            f"$d = \\SI{{{d_nm:g}}}{{\\nano\\meter}}$ "
            f"($Q_a = \\SI{{{Q_a}}}{{\\liter\\per\\minute}}$; colours: $\\lambda$). "
            f"Left: versus $m/q$; right: versus $(m/q)/m_p$.}}\n"
            f"  \\label{{fig:cluster-{i + 1}}}\n\\end{{figure}}\n"
        )

    return rf"""
\clearpage
%% ====================================================================
\section{{Multiply charged {n}-sphere cluster}}
\label{{sec:cluster}}
%% ====================================================================

In the present experiment no DMA is placed upstream of the APM. A randomly
oriented cluster of $n = {n}$ primary spheres carrying $q = {q}$ elementary
charges has the same mass-to-charge ratio $M/q = n m_p / q = m_p$ as the
singly charged monomer and is therefore transmitted at the same set-point
$(N, V)$.

\paragraph{{Cluster mobility.}}
The volume-equivalent and mobility-equivalent diameters are approximated as
\begin{{equation}}
  d_\text{{ve}} = n^{{1/3}} d, \qquad
  d_m = \chi\, d_\text{{ve}}, \qquad
  B_c = \frac{{C_c(d_m)}}{{3\pi\eta d_m}},
  \label{{eq:cluster-mobility}}
\end{{equation}}
with the orientation-averaged dynamic shape factor $\chi = {cfg['chi']}$
(two-sphere chain; Hinds, 1999, Table~3.2). This value refers to the
continuum regime; for the present Knudsen numbers (about 0.2--0.45) $d_m$
carries an uncertainty of a few percent.

\paragraph{{Transfer function.}}
For mass $M$ and charge $q e$, Eq.~\eqref{{eq:eom}} becomes
\begin{{equation}}
  \dv{{r}}{{z}} \propto q B_c
  \qty[\frac{{M}}{{q}} \omega^2 r - \frac{{e V}}{{r \ln(r_2/r_1)}}],
  \label{{eq:cluster-eom}}
\end{{equation}}
so the cluster transfer function is simulated with the monomer code using
the voltage $qV$ and the mobility diameter $d_m$. As a function of $M/q$ it
has the shape of a monomer transfer function with the resolution parameter
\begin{{equation}}
  \lambda_c = \frac{{2 M B_c \omega^2 L}}{{\bar{{v}}}}.
  \label{{eq:cluster-lambda}}
\end{{equation}}
Because $B_c/B_1 \approx 0.63$--$0.65$ while $M = {n} m_p$, $\lambda_c$ exceeds
$\lambda$, and the cluster transfer function is narrower with a lower peak.

\paragraph{{Relative transmission.}}
Table~\ref{{tab:cluster}} lists $\Omega_\text{{APM}}$ of the monomer and of
the cluster at $m/q = m_p$; for monodisperse primary particles their ratio
$\Omega_c/\Omega_1$ is the relative counting efficiency of the cluster at a
fixed set-point. The ratio of the areas $\int \Omega_\text{{APM}} \dd(m/q)$,
$A_c/A_1$, approximates the ratio of voltage-scan-integrated signals. The
actual contribution of clusters further depends on their number
concentration and on the fraction carrying ${q}$ charges, which are not
included here.

\begin{{table}}[htbp]
  \centering
  \caption{{Monomer ($q=1$) and {n}-sphere cluster ($q={q}$) at the monomer
  set-points ($Q_a = \SI{{{Q_a}}}{{\liter\per\minute}}$, $\chi = {cfg['chi']}$).
  $\Delta_\text{{mode}}$: relative shift of the cluster mode in $m/q$ from
  $m_p$; $\Omega_1$, $\Omega_c$: transmission at $m/q = m_p$; FWHM in $m/q$.}}
  \label{{tab:cluster}}
  \footnotesize
  \setlength{{\tabcolsep}}{{4pt}}
  \begin{{tabular}}{{ccccccccccc}}
  \toprule
  $d$ & $\lambda$ & $d_m$ & $\lambda_c$ & $\Delta_\text{{mode}}$ & $\Omega_1$ & $\Omega_c$
  & $\Omega_c/\Omega_1$ & FWHM$_1$ & FWHM$_c$ & $A_c/A_1$ \\
  \relax[\si{{\nano\meter}}] & & [\si{{\nano\meter}}] & & & & & & [\si{{\femto\gram}}] & [\si{{\femto\gram}}] & \\
  \midrule
{chr(10).join(lines)}
  \bottomrule
  \end{{tabular}}
\end{{table}}

The transfer functions are compared in
Figs.~\ref{{fig:cluster-1}}--\ref{{fig:cluster-{len(figs)}}}.

{chr(10).join(figs)}"""


def build_tex(rows: list[dict], lambdas: list[float], cluster_tex: str = "") -> str:
    """Assemble the LaTeX source of the report."""
    Q_a   = params.SP_Q_a_lpm
    today = datetime.date.today().isoformat()

    particle_lines = "\n".join(
        f"  \\num{{{d_nm:g}}} & \\num{{{rho_eff / 1000:g}}} & "
        f"\\num{{{rho_eff * np.pi * (d_nm * 1e-9) ** 3 / 6.0 * 1e18:.3f}}} \\\\"
        for d_nm, rho_eff in params.SP_particles
    )
    result_lines = "\n".join(
        f"  \\num{{{r['d_nm']:g}}} & \\num{{{r['m_fg']:.2f}}} & \\num{{{r['lambda']:g}}} & "
        f"\\num{{{r['RPM']:.1f}}} & \\num{{{r['V']:.2f}}} & \\num{{{r['V_force_balance']:.2f}}} & "
        f"\\num{{{r['Omega_peak']:.3f}}} & \\num{{{r['FWHM_fg']:.2f}}} & "
        f"\\num{{{r['m_over_FWHM']:.2f}}} \\\\"
        for r in rows
    )
    max_dev = max(abs(r["mode_fg"] / r["m_fg"] - 1.0) for r in rows)
    figures = "\n".join(
        "\\begin{figure}[htbp]\n  \\centering\n"
        f"  \\includegraphics[width=\\textwidth]{{{name}}}\n"
        f"  \\caption{{{cap}}}\n"
        f"  \\label{{fig:setpoint-{i + 1}}}\n\\end{{figure}}\n"
        for i, (name, cap) in enumerate(_figure_files(lambdas))
    )
    lam_str = ", ".join(f"\\num{{{lam:g}}}" for lam in lambdas)

    return rf"""\documentclass[11pt,a4paper]{{article}}

\usepackage{{amsmath, amssymb, amsthm}}
\usepackage{{physics}}
\usepackage{{bm}}
\usepackage{{siunitx}}
\usepackage{{booktabs}}
\usepackage{{graphicx}}
\usepackage[margin=1in]{{geometry}}
\usepackage{{hyperref}}

\graphicspath{{{{./}}}}

\title{{APM Set-points and Simulated Transfer Functions\\
for Singly Charged Spherical Particles}}
\author{{N.~Moteki}}
\date{{{today}}}

\begin{{document}}
\maketitle

%% ====================================================================
\section{{Calculation conditions}}
\label{{sec:conditions}}
%% ====================================================================

The APM rotation speed and applied voltage are determined for singly charged
spherical particles specified by their diameter $d$ (equal to the mobility
diameter) and effective density $\rho_\text{{eff}}$
(Table~\ref{{tab:particles}}). The operating and numerical conditions are
listed in Table~\ref{{tab:conditions}}.

\begin{{table}}[htbp]
  \centering
  \caption{{Target particles.}}
  \label{{tab:particles}}
  \begin{{tabular}}{{ccc}}
  \toprule
  $d$ [\si{{\nano\meter}}] & $\rho_\text{{eff}}$ [\si{{\gram\per\centi\meter\cubed}}] & $m_p$ [\si{{\femto\gram}}] \\
  \midrule
{particle_lines}
  \bottomrule
  \end{{tabular}}
\end{{table}}

\begin{{table}}[htbp]
  \centering
  \caption{{APM operating conditions, physical constants and numerical settings.}}
  \label{{tab:conditions}}
  \begin{{tabular}}{{ll}}
  \toprule
  Quantity & Value \\
  \midrule
  Aerosol flow rate $Q_a$ & \SI{{{Q_a}}}{{\liter\per\minute}} \\
  Resolution parameter $\lambda$ & {lam_str} \\
  Electrode length $L$ & \SI{{{params.L * 1e3:g}}}{{\milli\meter}} \\
  Inner / outer electrode radius $r_1$, $r_2$ & \SI{{{params.r1 * 1e3:g}}}{{\milli\meter}}, \SI{{{params.r2 * 1e3:g}}}{{\milli\meter}} \\
  Elementary charge $e$ & \SI{{{E_CHARGE:.5e}}}{{\coulomb}} \\
  Air viscosity $\eta$ & \SI{{{AIR_VISC:.3e}}}{{\pascal\second}} \\
  Pressure $P$ & \SI{{{ATM_PRES:.4g}}}{{\pascal}} \\
  RK4 step $\Delta z$ & \SI{{{params.dz * 1e3:g}}}{{\milli\meter}} \\
  Initial radial positions $N_{{r_0}}$ & \num{{{params.nr0}}} \\
  Mass grid & $m_p \times [{max(1.0 - params.SP_rel_halfwidth, 0.01):g},\ {1.0 + params.SP_rel_halfwidth:g}]$, \num{{{params.SP_num_m}}} points \\
  \bottomrule
  \end{{tabular}}
\end{{table}}

%% ====================================================================
\section{{Method}}
\label{{sec:method}}
%% ====================================================================

\paragraph{{Particle properties.}}
The particle mass, mechanical mobility and relaxation time are
\begin{{equation}}
  m_p = \frac{{\pi}}{{6}} \rho_\text{{eff}} d^3, \qquad
  B = \frac{{C_c(d)}}{{3\pi\eta d}}, \qquad
  \tau = m_p B,
  \label{{eq:particle}}
\end{{equation}}
where the Cunningham slip correction factor is
$C_c = 1 + [15.60 + 7.00\exp(-0.059 P d)]/(P d)$ with $P$ in
\si{{\kilo\pascal}} and $d$ in \si{{\micro\meter}} (Hinds, 1999, Eq.~3.22).

\paragraph{{Rotation speed.}}
With the mean axial velocity $\bar{{v}} = Q_a / [\pi (r_2^2 - r_1^2)]$, the
resolution parameter is defined as (Ehara et al., 1996)
\begin{{equation}}
  \lambda = \frac{{2 \tau \omega^2 L}}{{\bar{{v}}}}
  \quad\Longrightarrow\quad
  \omega = \sqrt{{\frac{{\lambda \bar{{v}}}}{{2 \tau L}}}}, \qquad
  N = \frac{{60\,\omega}}{{2\pi}},
  \label{{eq:lambda}}
\end{{equation}}
where $N$ is the rotation speed in rpm.

\paragraph{{Transfer function simulation.}}
Particle trajectories in the annular gap (centre radius $r_c = (r_1+r_2)/2$,
half-width $\delta = (r_2-r_1)/2$) are integrated along $z$ with the
fourth-order Runge--Kutta method under a parabolic axial velocity profile,
\begin{{equation}}
  \dv{{r}}{{z}} = \frac{{8 \delta r_c}}{{9 \eta Q_a}} \frac{{C_c}}{{d}}
  \frac{{m \omega^2 r - e V / [r \ln(r_2/r_1)]}}{{1 - [(r - r_c)/\delta]^2}}.
  \label{{eq:eom}}
\end{{equation}}
The transfer function $\Omega_\text{{APM}}(m; V, \omega)$ is the fraction of
the inlet particle flux, weighted by the parabolic profile over
$N_{{r_0}}$ initial radii, that traverses the electrode length $L$ without
reaching the walls.

\paragraph{{Voltage.}}
The force balance at $r_c$ gives the nominal voltage
\begin{{equation}}
  V_0 = \frac{{m_p \omega^2 r_c^2 \ln(r_2/r_1)}}{{e}}.
  \label{{eq:force-balance}}
\end{{equation}}
Since Eq.~\eqref{{eq:eom}} depends on $m$ and $V$ only through
$m\omega^2 r - eV/[r\ln(r_2/r_1)]$, at fixed $\omega$, $d$ and $Q_a$ the
transfer function is a function of $m/V$ alone, and its mode scales linearly
with $V$. The mode $m_0^*$ of $\Omega_\text{{APM}}(m; V_0)$ is located
numerically and the voltage is set to
\begin{{equation}}
  V = V_0 \, \frac{{m_p}}{{m_0^*}},
  \label{{eq:voltage}}
\end{{equation}}
which places the mode of the transfer function exactly at $m_p$. Because the
transfer function has a flat top, the mode is defined as the midpoint of the
contiguous region where $\Omega_\text{{APM}} \geq 0.99 \max \Omega_\text{{APM}}$.
The transfer function is then re-simulated at $(N, V)$ for verification.

\paragraph{{Invariance of the transfer function shape.}}
Substituting $V \propto m_p\omega^2$ into Eq.~\eqref{{eq:eom}}, the
right-hand side becomes proportional to $\tau\omega^2/Q_a \propto \lambda$
times a function of $m/m_p$ and $r$. Therefore, for a given APM geometry,
$\Omega_\text{{APM}}$ as a function of $m/m_p$ depends only on $\lambda$ and is
independent of $d$, $\rho_\text{{eff}}$ and $Q_a$.

%% ====================================================================
\section{{Results}}
\label{{sec:results}}
%% ====================================================================

The set-points and properties of the simulated transfer functions are listed
in Table~\ref{{tab:results}}. In all cases the mode of the simulated transfer
function coincides with $m_p$ within a relative deviation of
\num{{{max_dev:.1e}}}. As expected from the invariance above, the peak
transmission and the resolution $m_p/\text{{FWHM}}$ depend only on $\lambda$.
The transfer functions are shown in
Figs.~\ref{{fig:setpoint-1}}--\ref{{fig:setpoint-{len(_figure_files(lambdas))}}}.

\begin{{table}}[htbp]
  \centering
  \caption{{APM set-points and properties of the simulated transfer functions
  ($Q_a = \SI{{{Q_a}}}{{\liter\per\minute}}$). $V_0$: force-balance voltage,
  Eq.~\eqref{{eq:force-balance}}; $V$: voltage matching the mode to $m_p$,
  Eq.~\eqref{{eq:voltage}}.}}
  \label{{tab:results}}
  \small
  \begin{{tabular}}{{ccccccccc}}
  \toprule
  $d$ & $m_p$ & $\lambda$ & $N$ & $V$ & $V_0$ & $\max\Omega_\text{{APM}}$ & FWHM & $m_p/\text{{FWHM}}$ \\
  \relax[\si{{\nano\meter}}] & [\si{{\femto\gram}}] & & [rpm] & [\si{{\volt}}] & [\si{{\volt}}] & & [\si{{\femto\gram}}] & \\
  \midrule
{result_lines}
  \bottomrule
  \end{{tabular}}
\end{{table}}

{figures}
{cluster_tex}
\clearpage
\section*{{References}}
\begin{{itemize}}
  \item Ehara, K., Hagwood, C., and Coakley, K.~J. (1996). Novel method to
  classify aerosol particles according to their mass-to-charge ratio---Aerosol
  particle mass analyser. \emph{{J. Aerosol Sci.}}, 27, 217--234.
  \item Hinds, W.~C. (1999). \emph{{Aerosol Technology: Properties, Behavior,
  and Measurement of Airborne Particles}} (2nd ed.). Wiley.
\end{{itemize}}

\section*{{Acknowledgment}}
This document was prepared with the assistance of Claude (Anthropic).
The author assumes full responsibility for the content.

\end{{document}}
"""


def main() -> None:
    out_dir = getattr(params, "SP_OUTPUT_DIR", params.OUTPUT_DIR)
    lambdas = [float(lam) for lam in np.atleast_1d(params.SP_lambda)]
    rows    = _read_summary(os.path.join(out_dir, "apm_setpoint_summary.csv"))

    for name, _ in _figure_files(lambdas):
        if not os.path.exists(os.path.join(out_dir, name)):
            raise FileNotFoundError(
                f"{name} not found in {out_dir}; run run_apm_setpoint.py first."
            )

    tex_path = os.path.join(out_dir, f"{REPORT_STEM}.tex")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(build_tex(rows, lambdas, _cluster_section(lambdas, out_dir)))
    print(f"LaTeX source saved: {tex_path}")

    subprocess.run(
        ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", f"{REPORT_STEM}.tex"],
        cwd=out_dir, check=True, stdout=subprocess.DEVNULL,
    )
    print(f"PDF saved: {os.path.join(out_dir, REPORT_STEM + '.pdf')}")


if __name__ == "__main__":
    main()
