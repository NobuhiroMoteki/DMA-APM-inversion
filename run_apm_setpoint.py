"""
run_apm_setpoint.py  ---  APM set-point (RPM, voltage) for spherical particles

For each (diameter, effective density) pair in params.SP_particles, computes
the rotation speed and voltage at which the mode of the APM transfer function
coincides with the mass of the singly charged particle, for the given aerosol
flow rate and resolution parameter(s) lambda. The simulated transfer
functions are saved as CSV files, the set-points are summarised in
apm_setpoint_summary.csv, and two-panel JPEG figures are produced:
    - single lambda:   one figure, all particles overlaid
    - list of lambdas: one figure per particle, lambdas overlaid
If params.SP_cluster is set, the transfer function of the multiply charged
sphere cluster with the same mass-to-charge ratio is simulated at each
set-point (apm_cluster_summary.csv, one figure per particle).

Usage:
    python run_apm_setpoint.py

Set SP_* parameters (and APM geometry / RK4 settings) in params.py.
"""
import os

import numpy as np

import params
from apm_setpoint  import find_apm_setpoint, simulate_cluster
from visualization import plot_cluster_transfer_functions, plot_setpoint_transfer_functions


def main() -> None:
    out_dir = getattr(params, "SP_OUTPUT_DIR", params.OUTPUT_DIR)
    os.makedirs(out_dir, exist_ok=True)
    lambdas = [float(lam) for lam in np.atleast_1d(params.SP_lambda)]
    Q_a     = params.SP_Q_a_lpm
    print(f"APM set-point search (Q_a = {Q_a:.3f} L/min, lambda = {lambdas})")

    setpoints = {}   # key: (d_nm, rho_eff, lambda)
    for d_nm, rho_eff in params.SP_particles:
        for lam in lambdas:
            sp = find_apm_setpoint(
                d_nm * 1e-9, rho_eff, Q_a, lam, params,
                rel_halfwidth=params.SP_rel_halfwidth, num_m=params.SP_num_m,
            )
            setpoints[(d_nm, rho_eff, lam)] = sp

            print(
                f"  d = {d_nm:6.1f} nm, rho_eff = {rho_eff:7.1f} kg/m3, lambda = {lam:.2f}: "
                f"m = {sp.m_p * 1e18:8.3f} fg, tau = {sp.tau:.3e} s\n"
                f"    -> RPM = {sp.RPM:8.1f} rpm, V = {sp.V:8.2f} V "
                f"(force balance: {sp.V_balance:8.2f} V)\n"
                f"    simulated mode = {sp.m_mode * 1e18:8.3f} fg "
                f"(mode/m - 1 = {sp.m_mode / sp.m_p - 1.0:+.2e}), "
                f"peak Omega = {sp.Omega_peak:.3f}, "
                f"FWHM = {sp.fwhm * 1e18:.3f} fg (m/FWHM = {sp.m_p / sp.fwhm:.2f})"
            )
            if sp.truncated:
                print("    [Warning] Omega is truncated at the mass-range edge; "
                      "increase SP_rel_halfwidth (FWHM is a lower bound).")

            stem     = (f"apm_setpoint_d{d_nm:.0f}nm_rho{rho_eff:.0f}"
                        f"_Q{Q_a:.2f}lpm_lam{lam:.2f}")
            csv_path = os.path.join(out_dir, f"{stem}.csv")
            header = (
                f"d_nm={d_nm}, rho_eff_kg_m3={rho_eff}, Q_a_lpm={Q_a}, "
                f"lambda={lam}, m_fg={sp.m_p * 1e18:.6g}, "
                f"RPM={sp.RPM:.6g}, V={sp.V:.6g}\n"
                f"mass_fg,Omega_APM"
            )
            np.savetxt(
                csv_path,
                np.column_stack([sp.m_array * 1e18, sp.Omega]),   # shape: (num_m, 2)
                delimiter=",", header=header, fmt="%.8g",
            )
            print(f"    Data saved: {csv_path}")

    # ---- Summary table of set-points ----
    summary_path = os.path.join(out_dir, "apm_setpoint_summary.csv")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("d_nm,rho_eff_kg_m3,Q_a_lpm,lambda,m_fg,tau_s,RPM,V,"
                "V_force_balance,mode_fg,Omega_peak,FWHM_fg,m_over_FWHM\n")
        for (d_nm, rho_eff, lam), sp in setpoints.items():
            f.write(
                f"{d_nm},{rho_eff},{Q_a},{lam},{sp.m_p * 1e18:.6g},{sp.tau:.6g},"
                f"{sp.RPM:.6g},{sp.V:.6g},{sp.V_balance:.6g},{sp.m_mode * 1e18:.6g},"
                f"{sp.Omega_peak:.4f},{sp.fwhm * 1e18:.6g},{sp.m_p / sp.fwhm:.4f}\n"
            )
    print(f"\nSummary saved: {summary_path}")

    # ---- Multiply charged cluster at the same set-points ----
    cfg      = getattr(params, "SP_cluster", None)
    clusters = {}
    if cfg:
        print(f"\nCluster: n = {cfg['n_mono']}, q = {cfg['charge']}, chi = {cfg['chi']}")
        for key, sp in setpoints.items():
            cl = simulate_cluster(
                sp, params, n_mono=cfg["n_mono"], charge=cfg["charge"], chi=cfg["chi"],
                rel_halfwidth=params.SP_rel_halfwidth, num_m=params.SP_num_m,
            )
            clusters[key] = cl
            d_nm, rho_eff, lam = key
            print(
                f"  d = {d_nm:6.1f} nm, lambda = {lam:.2f}: d_m = {cl.d_m * 1e9:6.1f} nm, "
                f"lambda_c = {cl.lam_eff:.3f}, Z/Z_1 = {cl.Z_ratio:.3f}, "
                f"mode(m/q) = {cl.mq_mode * 1e18:8.3f} fg, peak Omega = {cl.Omega_peak:.3f}, "
                f"FWHM(m/q) = {cl.fwhm_mq * 1e18:.3f} fg"
            )
            if cl.truncated:
                print("    [Warning] Omega is truncated at the mass-range edge; "
                      "increase SP_rel_halfwidth (FWHM is a lower bound).")
            stem = (f"apm_cluster{cl.n_mono}q{cl.charge}_d{d_nm:.0f}nm_rho{rho_eff:.0f}"
                    f"_Q{Q_a:.2f}lpm_lam{lam:.2f}")
            header = (
                f"cluster n={cl.n_mono}, q={cl.charge}, chi={cl.chi}, d_m_nm={cl.d_m * 1e9:.6g}, "
                f"primary d_nm={d_nm}, rho_eff_kg_m3={rho_eff}, Q_a_lpm={Q_a}, "
                f"set-point RPM={sp.RPM:.6g}, V={sp.V:.6g} (monomer lambda={lam})\n"
                f"cluster_mass_fg,mass_to_charge_fg_per_e,Omega_APM"
            )
            np.savetxt(
                os.path.join(out_dir, f"{stem}.csv"),
                np.column_stack([cl.m_array * 1e18, cl.m_array / cl.charge * 1e18,
                                 cl.Omega]),   # shape: (num_m, 3)
                delimiter=",", header=header, fmt="%.8g",
            )

        cl_path = os.path.join(out_dir, "apm_cluster_summary.csv")
        with open(cl_path, "w", encoding="utf-8") as f:
            f.write("d_nm,rho_eff_kg_m3,Q_a_lpm,lambda,n_mono,charge,chi,d_ve_nm,d_m_nm,"
                    "mass_fg,lambda_c,Z_ratio,mode_mq_fg,Omega_peak,FWHM_mq_fg,"
                    "m_over_FWHM\n")
            for (d_nm, rho_eff, lam), cl in clusters.items():
                f.write(
                    f"{d_nm},{rho_eff},{Q_a},{lam},{cl.n_mono},{cl.charge},{cl.chi},"
                    f"{cl.d_ve * 1e9:.6g},{cl.d_m * 1e9:.6g},{cl.mass * 1e18:.6g},"
                    f"{cl.lam_eff:.6g},{cl.Z_ratio:.6g},{cl.mq_mode * 1e18:.6g},"
                    f"{cl.Omega_peak:.4f},{cl.fwhm_mq * 1e18:.6g},"
                    f"{cl.mq_mode / cl.fwhm_mq:.4f}\n"
                )
        print(f"\nCluster summary saved: {cl_path}")

    # ---- Figures ----
    if len(lambdas) == 1:
        fig_path = os.path.join(out_dir, f"apm_setpoint_Q{Q_a:.2f}lpm_lam{lambdas[0]:.2f}.jpg")
        plot_setpoint_transfer_functions(list(setpoints.values()), fig_path, overlay="particle")
    else:
        lam_tag = "-".join(f"{lam:.2f}" for lam in lambdas)
        for d_nm, rho_eff in params.SP_particles:
            fig_path = os.path.join(
                out_dir,
                f"apm_setpoint_d{d_nm:.0f}nm_rho{rho_eff:.0f}_Q{Q_a:.2f}lpm_lam{lam_tag}.jpg",
            )
            plot_setpoint_transfer_functions(
                [setpoints[(d_nm, rho_eff, lam)] for lam in lambdas],
                fig_path, overlay="lambda",
            )
    if clusters:
        cl0 = next(iter(clusters.values()))
        for d_nm, rho_eff in params.SP_particles:
            keys     = [(d_nm, rho_eff, lam) for lam in lambdas]
            fig_path = os.path.join(
                out_dir,
                f"apm_cluster{cl0.n_mono}q{cl0.charge}_d{d_nm:.0f}nm_rho{rho_eff:.0f}"
                f"_Q{Q_a:.2f}lpm.jpg",
            )
            plot_cluster_transfer_functions(
                [setpoints[k] for k in keys], [clusters[k] for k in keys], fig_path,
            )


if __name__ == "__main__":
    main()
