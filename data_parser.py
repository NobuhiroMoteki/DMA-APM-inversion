"""
data_parser.py  ---  Loading and binning of APM-CPC measurement data

Automatically extracts the electrical mobility diameter Dmob from the CSV
header and performs voltage binning by merging upward and downward scans.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class MeasurementData:
    """Result of binning APM-CPC measurement data.

    Attributes:
        V_array:        Representative voltage of each bin [V],          shape (I,)
        n_meas:         Measured particle concentration [cm⁻³],          shape (I,)
        V_sample_array: Integrated CPC sampling volume [cm³],            shape (I,)
        t_meas_array:   Integrated measurement time per bin [s],         shape (I,)
        RPM:            Mean APM rotation speed [rpm]
        Dmob:           Electrical mobility diameter [m]
        I:              Number of valid voltage bins
    """
    V_array:        np.ndarray
    n_meas:         np.ndarray
    V_sample_array: np.ndarray
    t_meas_array:   np.ndarray
    RPM:            float
    Dmob:           float
    I:              int


def load_and_bin(params) -> MeasurementData:
    """Load an APM control software CSV file and bin the voltage scan.

    CSV format:
        Lines 1-8:  Header information (Dmob is extracted from the line
                    containing 'Electrical Mobility Diameter')
        Line 9:     Column names (Time, Rotation Speed, Applied Voltage,
                    Outlet Particle Concentration)
        Line 10+:   Time-series measurement data

    Physical rationale for binning:
        Intensive variables (voltage, concentration): averaged with mean()
        Extensive variables (time):                  summed with sum()
        -> Ensures accurate evaluation of V_sample = Q_cpc x t_meas
           for Poisson variance weighting.

    Args:
        params: User configuration module defined in params.py

    Returns:
        MeasurementData: Binned measurement data
    """
    file_path = params.FILE_PATH

    # ----------------------------------------------------------------
    # 1. Extract Dmob from header
    # ----------------------------------------------------------------
    Dmob = 450.0e-9  # default [m]
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            for _ in range(8):
                line = f.readline().strip()
                if "Electrical Mobility Diameter" in line:
                    parts = line.split(",")
                    if len(parts) > 1:
                        Dmob = float(parts[1]) * 1e-9  # [nm] -> [m]
    except Exception as exc:
        print(f"[Warning] Failed to read header: {exc}  ->  Using default Dmob={Dmob*1e9:.0f} nm")

    # ----------------------------------------------------------------
    # 2. Load time-series data and compute Δt
    # ----------------------------------------------------------------
    df = pd.read_csv(file_path, skiprows=8)
    df["Time"] = pd.to_datetime(df["Time"])

    # Measurement time interval Δt [s] per row: fill first NaN by back-fill
    df["delta_t"] = df["Time"].diff().dt.total_seconds().bfill()

    RPM = float(df["Rotation Speed"].mean())

    # ----------------------------------------------------------------
    # 3. Voltage binning (merging upward and downward scans)
    # ----------------------------------------------------------------
    min_v = df["Applied Voltage"].min()
    max_v = df["Applied Voltage"].max()
    bins  = np.linspace(min_v - 1, max_v + 1, params.num_bins + 1)

    df["V_bin"] = pd.cut(df["Applied Voltage"], bins=bins, labels=False)
    grouped = df.groupby("V_bin")

    v_grouped = grouped["Applied Voltage"].mean().dropna()
    n_grouped = grouped["Outlet Particle Concentration"].mean().dropna()

    valid_bins   = v_grouped.index
    t_meas_array = grouped["delta_t"].sum().loc[valid_bins].values

    V_array = v_grouped.values
    n_meas  = n_grouped.values
    I       = len(V_array)

    # ----------------------------------------------------------------
    # 4. Compute integrated CPC sampling volume
    # ----------------------------------------------------------------
    Q_cpc_ccps     = params.Q_cpc_lpm * 1000.0 / 60.0   # [L/min] -> [cm³/s]
    V_sample_array = Q_cpc_ccps * t_meas_array

    print("=== Data loading complete ===")
    print(f"  Number of valid voltage bins I = {I}")
    print(f"  APM rotation speed (mean): {RPM:.1f} rpm")
    print(f"  Mobility diameter Dmob = {Dmob * 1e9:.1f} nm")

    return MeasurementData(
        V_array=V_array,
        n_meas=n_meas,
        V_sample_array=V_sample_array,
        t_meas_array=t_meas_array,
        RPM=RPM,
        Dmob=Dmob,
        I=I,
    )
