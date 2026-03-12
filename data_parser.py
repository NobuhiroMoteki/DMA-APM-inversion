"""
data_parser.py  ―  APM-CPC 測定データの読み込みとビニング処理

CSVヘッダーから電気移動度径 Dmob を自動抽出し、
上昇・下降スキャンを統合した電圧ビニングを行う。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class MeasurementData:
    """APM-CPC 測定データのビニング処理結果。

    Attributes:
        V_array:        電圧ビン代表値 [V],          shape (I,)
        n_meas:         測定粒子濃度 [cm⁻³],         shape (I,)
        V_sample_array: CPC 積算吸引体積 [cm³],      shape (I,)
        t_meas_array:   各ビンの積算測定時間 [s],    shape (I,)
        RPM:            APM 平均回転数 [rpm]
        Dmob:           電気移動度径 [m]
        I:              有効電圧ビン数
    """
    V_array:        np.ndarray
    n_meas:         np.ndarray
    V_sample_array: np.ndarray
    t_meas_array:   np.ndarray
    RPM:            float
    Dmob:           float
    I:              int


def load_and_bin(params) -> MeasurementData:
    """APM 制御ソフト出力 CSV を読み込み、電圧スキャンをビニングする。

    CSV フォーマット:
        1〜8行目: ヘッダー情報 ('Electrical Mobility Diameter' を含む行から Dmob を取得)
        9行目:    列名 (Time, Rotation Speed, Applied Voltage, Outlet Particle Concentration)
        10行目〜: 時系列測定データ

    ビニングの物理的根拠:
        示強性変数 (電圧, 濃度): 平均値 mean() を採用
        示量性変数 (時間):       合計値 sum()  を採用
        → 積算吸引体積 V_sample = Q_cpc × t_meas を正確に評価するため

    Args:
        params: params.py で定義されたユーザー設定モジュール

    Returns:
        MeasurementData: ビニング済みの測定データ
    """
    file_path = params.FILE_PATH

    # ----------------------------------------------------------------
    # 1. ヘッダーから Dmob 抽出
    # ----------------------------------------------------------------
    Dmob = 450.0e-9  # デフォルト [m]
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            for _ in range(8):
                line = f.readline().strip()
                if "Electrical Mobility Diameter" in line:
                    parts = line.split(",")
                    if len(parts) > 1:
                        Dmob = float(parts[1]) * 1e-9  # [nm] → [m]
    except Exception as exc:
        print(f"[警告] ヘッダー読み込み失敗: {exc}  →  デフォルト Dmob={Dmob*1e9:.0f} nm を使用")

    # ----------------------------------------------------------------
    # 2. 時系列データ読み込みと Δt 計算
    # ----------------------------------------------------------------
    df = pd.read_csv(file_path, skiprows=8)
    df["Time"] = pd.to_datetime(df["Time"])

    # 各行が代表する測定時間 Δt [s]: 後方補間で最初の NaN を埋める
    df["delta_t"] = df["Time"].diff().dt.total_seconds().bfill()

    RPM = float(df["Rotation Speed"].mean())

    # ----------------------------------------------------------------
    # 3. 電圧スキャンのビニング (上昇・下降を統合)
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
    # 4. CPC 積算吸引体積の計算
    # ----------------------------------------------------------------
    Q_cpc_ccps     = params.Q_cpc_lpm * 1000.0 / 60.0   # [L/min] → [cm³/s]
    V_sample_array = Q_cpc_ccps * t_meas_array

    print("=== データ読み込み完了 ===")
    print(f"  有効電圧ビン数 I = {I}")
    print(f"  APM 回転数 (平均): {RPM:.1f} rpm")
    print(f"  移動度粒径 Dmob  = {Dmob * 1e9:.1f} nm")

    return MeasurementData(
        V_array=V_array,
        n_meas=n_meas,
        V_sample_array=V_sample_array,
        t_meas_array=t_meas_array,
        RPM=RPM,
        Dmob=Dmob,
        I=I,
    )
