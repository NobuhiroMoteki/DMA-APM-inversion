"""
run_transfer_function.py  ---  Simulation and plot of the standalone APM transfer function

Computes Omega_APM(m; V, Z_p*) with the RK4 trajectory simulation for each
voltage in params.TF_V_list and saves a two-panel JPEG figure.

Usage:
    python run_transfer_function.py

Set TF_* parameters (and APM geometry / flow / RK4 settings) in params.py.
"""
import os

import numpy as np

import params
from kernel_simulator import classifying_mass, compute_apm_transfer_function
from visualization    import plot_transfer_function


def main() -> None:
    RPM  = params.TF_RPM
    Dmob = params.TF_Dmob_nm * 1e-9   # [nm] -> [m]

    m_arrays, Omegas, m_c_list = [], [], []
    print(f"Computing APM transfer function (Dmob = {params.TF_Dmob_nm:.0f} nm, RPM = {RPM:.0f})...")
    for V in params.TF_V_list:
        m_c     = classifying_mass(V, RPM, params)
        m_array = np.linspace(
            m_c * max(1.0 - params.TF_rel_halfwidth, 0.01),
            m_c * (1.0 + params.TF_rel_halfwidth),
            params.TF_num_m,
        )   # shape: (TF_num_m,) [kg]
        Omega = compute_apm_transfer_function(m_array, V, RPM, Dmob, params)

        # Summary statistics of the transfer function
        dm       = m_array[1] - m_array[0]
        area     = np.sum(Omega) * dm
        m_peak   = m_array[np.argmax(Omega)]
        above    = m_array[Omega >= 0.5 * Omega.max()]
        fwhm     = above[-1] - above[0] if above.size > 1 else np.nan
        print(
            f"  V = {V:7.1f} V: m_c = {m_c * 1e18:8.3f} fg, "
            f"peak Omega = {Omega.max():.3f} at {m_peak * 1e18:8.3f} fg, "
            f"FWHM = {fwhm * 1e18:7.3f} fg (m_c/FWHM = {m_c / fwhm:.2f}), "
            f"area/m_c = {area / m_c:.4f}"
        )

        if max(Omega[0], Omega[-1]) >= 0.5 * Omega.max():
            print("    [Warning] Omega is truncated at the mass-range edge; "
                  "increase TF_rel_halfwidth (FWHM is a lower bound).")

        m_arrays.append(m_array)
        Omegas.append(Omega)
        m_c_list.append(m_c)

    output_path = os.path.join(
        params.OUTPUT_DIR,
        f"apm_transfer_function_Dmob{params.TF_Dmob_nm:.0f}nm_RPM{RPM:.0f}.jpg",
    )
    plot_transfer_function(m_arrays, Omegas, params.TF_V_list, m_c_list, RPM, Dmob, output_path)


if __name__ == "__main__":
    main()
