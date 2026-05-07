"""
run_batch_inversion.py  ---  Batch 1D inversion + NPZ data dump

Iterates over the six CSVs in DMA-APMscan-CPC_20260303BC, runs the standard
1D inversion pipeline, and stores all intermediate / final arrays in
results/data/<stem>.npz so that paper-quality figures can be regenerated
later without re-running the inversion.
"""
from __future__ import annotations

import os

import numpy as np

import params
from data_parser      import load_and_bin
from inversion_solver import solve_chahine_twomey
from kernel_simulator import build_kernel_1d
from visualization    import fit_gaussian_mode


DATA_DIR = "./APM-CPC_data/DMA-APMscan-CPC_20260303BC"
OUT_DIR  = "./results/data"

CSV_FILES = [
    "FS_Dmob450nm_try2__20050101014000.csv",
    "FS_Dmob500nm_try1__20050101024557.csv",
    "FS_Dmob550nm_try1__20050101015918.csv",
    "JetA1_Dmob450nm_try1__20050101030343.csv",
    "JetA1_Dmob500nm_try1__20050101032008.csv",
    "JetA1_Dmob550nm_try1__20050101033753.csv",
]


def run_one(filename: str) -> None:
    params.FILE_PATH = os.path.join(DATA_DIR, filename)

    data            = load_and_bin(params)
    K, m_array      = build_kernel_1d(data, params)
    f_estimated     = solve_chahine_twomey(K, m_array, data, params)
    fit             = fit_gaussian_mode(m_array, f_estimated)

    stem     = os.path.splitext(filename)[0]
    out_path = os.path.join(OUT_DIR, f"{stem}.npz")
    os.makedirs(OUT_DIR, exist_ok=True)

    np.savez(
        out_path,
        # MeasurementData
        V_array        = data.V_array,
        n_meas         = data.n_meas,
        V_sample_array = data.V_sample_array,
        t_meas_array   = data.t_meas_array,
        RPM            = np.float64(data.RPM),
        Dmob           = np.float64(data.Dmob),
        I              = np.int64(data.I),
        # Inversion arrays
        K              = K,
        m_array        = m_array,
        f_estimated    = f_estimated,
        # Gaussian fit
        fit_mu_fg      = np.float64(fit.mu_fg),
        fit_sigma_fg   = np.float64(fit.sigma_fg),
        fit_amplitude  = np.float64(fit.amplitude),
        fit_offset     = np.float64(fit.offset),
        fit_slope      = np.float64(fit.slope),
        fit_r_squared  = np.float64(fit.r_squared),
        fit_success    = np.bool_(fit.success),
        # Provenance
        source_csv     = np.str_(filename),
    )
    print(f"Saved data: {out_path}\n")


def main() -> None:
    for f in CSV_FILES:
        print("=" * 70)
        print(f"Processing: {f}")
        print("=" * 70)
        run_one(f)


if __name__ == "__main__":
    main()
