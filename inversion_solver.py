"""
inversion_solver.py  ―  Chahine-Twomey 非線形反復逆解析

第一種フレドホルム積分方程式 n = K f を解く。
Chahine 型の加重平均更新式と Markowski 1-2-1 内部スムージングを組み合わせ、
ポアソン計数統計に基づく reduced χ² を収束判定に使用する。

References:
    Twomey (1975) J. Comput. Phys. doi:10.1016/0021-9991(75)90028-5
    Markowski (1987) Aerosol Sci. Technol. doi:10.1080/02786828708959153
"""
from __future__ import annotations

import numpy as np
from data_parser import MeasurementData


def solve_chahine_twomey(
    K:       np.ndarray,
    m_array: np.ndarray,
    data:    MeasurementData,
    params,
) -> np.ndarray:
    """Chahine-Twomey 法で質量分布 f(m) を推定する。

    更新式:
        f_j^(k+1) = f_j^(k) × Σ_i [K_norm(i,j) · (n_meas_i / n_calc_i^(k))]
                                  / Σ_i  K_norm(i,j)

    収束判定 (reduced χ²):
        χ² = (1/I) Σ_i [(n_meas_i − n_calc_i)² / Var(n_i)]  < chi_threshold

    ポアソン分散:
        Var(n_i) = max(n_calc_i / V_sample_i,  1 / V_sample_i²)

    スムージング (Markowski 1-2-1):
        f_j ← 0.25·f_{j-1} + 0.50·f_j + 0.25·f_{j+1}   (内部点のみ)

    Args:
        K:       カーネル行列 [I × J]
        m_array: 質量グリッド [kg], shape (J,)
        data:    ビニング済み測定データ (n_meas, V_sample_array, I)
        params:  ユーザー設定 (J, m_min_fg, m_max_fg, max_iter, chi_threshold)

    Returns:
        f_estimated: 推定質量分布 dN/dm [cm⁻³ kg⁻¹], shape (J,)
    """
    n_meas         = data.n_meas
    V_sample_array = data.V_sample_array
    I              = data.I
    J              = params.J
    m_min          = params.m_min_fg * 1e-18   # [kg]
    m_max          = params.m_max_fg * 1e-18   # [kg]

    # 初期推測: 質量範囲全体に一様分布
    f_est  = np.ones(J) * (np.sum(n_meas) / (m_max - m_min))
    K_norm = K / np.max(K)

    chi_sq = np.inf
    print("逆問題解析 (Chahine-Twomey 反復) を開始します...")

    for k in range(params.max_iter):
        calc_n = K @ f_est
        calc_n = np.maximum(calc_n, 1e-10)   # ゼロ割防止

        # ポアソン統計に基づく reduced χ²
        variance = np.maximum(calc_n / V_sample_array, 1.0 / V_sample_array ** 2)
        chi_sq   = np.sum(((n_meas - calc_n) ** 2) / variance) / I

        if chi_sq < params.chi_threshold:
            print(f"  → {k} 反復で収束 (χ² = {chi_sq:.4f})")
            break

        # Chahine 型加重平均更新
        ratio = n_meas / calc_n
        f_new = np.copy(f_est)
        for j in range(J):
            correction = np.sum(ratio * K_norm[:, j]) / max(
                np.sum(K_norm[:, j]), 1e-10
            )
            f_new[j] = f_est[j] * max(correction, 0.01)

        # Markowski 1-2-1 スムージング (端点は更新しない)
        f_smooth        = np.copy(f_new)
        f_smooth[1:-1]  = (
            0.25 * f_new[:-2] + 0.5 * f_new[1:-1] + 0.25 * f_new[2:]
        )
        f_est = np.copy(f_smooth)

    else:
        print(f"  → 最大反復数 {params.max_iter} に到達 (χ² = {chi_sq:.4f})")

    return f_est
