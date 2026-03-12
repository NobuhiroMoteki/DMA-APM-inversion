"""
kernel_simulator.py  ―  APM 転送関数の数値計算とカーネル行列の構築

放物線流速分布を仮定した APM ギャップ内の粒子軌道を RK4 法で直接追跡し、
透過率関数 Ω_APM(m, V, Z_p) を計算する独自数値シミュレーション手法。

デフォルトの APM 幾何寸法 (L=100 mm, r1=24 mm, r2=25 mm) は
Kanomax APM Model-3601 の公称設計値です。

References:
    Ehara et al. (1996) J. Aerosol Sci. doi:10.1016/0021-8502(96)00014-4
    Hinds, W. C. (1999). Aerosol Technology (2nd ed.). Wiley. Eq. (3.22)
"""
from __future__ import annotations

import numpy as np
from data_parser import MeasurementData

# ------------------------------------------------------------------------------
# 物理定数
# ------------------------------------------------------------------------------
E_CHARGE: float = 1.60219e-19   # 素電荷 [C]
AIR_VISC: float = 1.83e-5       # 空気粘性係数 [Pa·s]  (約 25°C)
ATM_PRES: float = 1.013e5       # 標準大気圧 [Pa]


# ------------------------------------------------------------------------------
# 内部ユーティリティ
# ------------------------------------------------------------------------------

def _cunningham(Dmob: float) -> float:
    """カニンガムすべり補正係数を計算する。

    引用: Hinds, W. C. (1999). Aerosol Technology (2nd ed.). Wiley. Eq. (3.22)

    Args:
        Dmob: 電気移動度径 [m]

    Returns:
        Cc: カニンガム補正係数 (無次元)
    """
    x = ATM_PRES * 1e-3 * Dmob * 1e6   # P [kPa] × d [μm]
    return 1.0 + (15.60 + 7.00 * np.exp(-0.059 * x)) / x


def _rk4_transmission(
    m:         float,
    V:         float,
    coef:      float,
    V_term:    float,
    omega:     float,
    rc:        float,
    delta:     float,
    r1:        float,
    r2:        float,
    nr0:       int,
    num_steps: int,
) -> float:
    """RK4 法による粒子軌道積分と APM 透過率の計算。

    APM ギャップ内 (r1 ≤ r ≤ r2) の粒子群を nr0 点の初期位置から追跡し、
    軸方向 L を通過した粒子のフラックス割合を返す。

    運動方程式 (dr/dz):
        dr/dz = coef × (m·ω²·r − eV/(r·ln(r2/r1))) / (1 − ((r−rc)/δ)²)

    ここで coef = dz × 8/(9η) × (Cc/D_mob) × (δ·rc)/Q が dz を含むため、
    RK4 の各 k_i がそのまま変位量 [m] となる。

    透過率の重み付けは放物線流速分布に比例するフラックス荷重:
        weight(r₀) = 1.5 × (1 − ((r₀−rc)/δ)²)

    Args:
        m:         粒子質量 [kg]
        V:         印加電圧 [V]
        coef:      dr/dz の共通係数 (dz 込み) [m/step]
        V_term:    静電気力項 eV/ln(r2/r1) [N·m]
        omega:     APM 角速度 [rad/s]
        rc:        ギャップ中心半径 [m]
        delta:     ギャップ半値幅 [m]
        r1:        内筒半径 [m]
        r2:        外筒半径 [m]
        nr0:       初期粒子位置数
        num_steps: RK4 積分ステップ数

    Returns:
        透過効率 (0.0 〜 1.0)
    """
    r0      = np.linspace(r1 + 1e-5, r2 - 1e-5, nr0)
    r       = r0.copy()
    weights = 1.5 * (1.0 - ((r0 - rc) / delta) ** 2)
    total_weight = np.sum(weights)
    active  = np.ones(nr0, dtype=bool)

    def dr_dz(rad: np.ndarray) -> np.ndarray:
        return coef * (m * omega**2 * rad - V_term / rad) / (
            1.0 - ((rad - rc) / delta) ** 2
        )

    for _ in range(num_steps):
        if not np.any(active):
            break
        ra = r[active]
        k1 = dr_dz(ra)
        k2 = dr_dz(ra + 0.5 * k1)
        k3 = dr_dz(ra + 0.5 * k2)
        k4 = dr_dz(ra + k3)
        r[active] = ra + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        active[active] = np.abs(r[active] - rc) < (delta - 1e-5)

    return np.sum(weights[active]) / total_weight


# ------------------------------------------------------------------------------
# 公開 API
# ------------------------------------------------------------------------------

def build_kernel_1d(
    data:   MeasurementData,
    params,
) -> tuple[np.ndarray, np.ndarray]:
    """1D 近似カーネル行列 K[I, J] を構築する。

    DMA 出力をデルタ関数と近似し、単一移動度 Z_p* でカーネルを評価する:
        K[i, j] = Ω_APM(m_j, V_i, Z_p*) × Δm

    Args:
        data:   ビニング済み測定データ (V_array, RPM, Dmob, I)
        params: ユーザー設定 (r1, r2, L, Q_a_lpm, dz, nr0, m_min_fg, m_max_fg, J)

    Returns:
        K:       カーネル行列 [I × J]
        m_array: 質量グリッド [kg], shape (J,)
    """
    rc    = 0.5 * (params.r1 + params.r2)
    delta = 0.5 * (params.r2 - params.r1)
    Cc    = _cunningham(data.Dmob)
    omega = data.RPM / 60.0 * 2.0 * np.pi
    Q     = params.Q_a_lpm * 1e-3 / 60.0           # [m³/s]

    # dr/dz の共通係数: dz × 8/(9η) × (Cc/D_mob) × (δ·rc)/Q
    coef_base     = params.dz * 8.0 / (9.0 * AIR_VISC) * (Cc / data.Dmob) * (delta * rc) / Q
    V_term_factor = E_CHARGE / np.log(params.r2 / params.r1)
    num_steps     = int(params.L / params.dz)

    m_min   = params.m_min_fg * 1e-18
    m_max   = params.m_max_fg * 1e-18
    m_array = np.linspace(m_min, m_max, params.J)
    dm      = m_array[1] - m_array[0]

    I, J = data.I, params.J
    K    = np.zeros((I, J))

    print(f"1D カーネル行列 K[{I}×{J}] を計算中...")
    for i, V in enumerate(data.V_array):
        V_term = V_term_factor * V
        for j, m in enumerate(m_array):
            K[i, j] = _rk4_transmission(
                m, V, coef_base, V_term, omega,
                rc, delta, params.r1, params.r2,
                params.nr0, num_steps,
            ) * dm
    print("計算完了！")

    return K, m_array


def build_kernel_2d(
    data:   MeasurementData,
    params,
) -> tuple[np.ndarray, np.ndarray]:
    """2D 畳み込みカーネル行列 K_eff[I, J] を構築する。

    DMA 三角形転送関数 Ω_DMA(Z_p) と APM 転送関数の畳み込みにより
    有効カーネルを計算する:
        K_eff[i,j] = [∫ Ω_APM(m_j,V_i,Z_p) Ω_DMA(Z_p) dZ_p / ∫ Ω_DMA dZ_p] × Δm

    Z_p 積分は params.num_Zp_bins 点の矩形公式で離散化する。

    Args:
        data:   ビニング済み測定データ
        params: ユーザー設定 (beta, num_Zp_bins に加えて 1D と同じパラメータ)

    Returns:
        K:       カーネル行列 [I × J]
        m_array: 質量グリッド [kg], shape (J,)
    """
    rc    = 0.5 * (params.r1 + params.r2)
    delta = 0.5 * (params.r2 - params.r1)
    Cc_star = _cunningham(data.Dmob)
    Zp_star = E_CHARGE * Cc_star / (3.0 * np.pi * AIR_VISC * data.Dmob)
    omega   = data.RPM / 60.0 * 2.0 * np.pi
    Q       = params.Q_a_lpm * 1e-3 / 60.0

    V_term_factor = E_CHARGE / np.log(params.r2 / params.r1)
    num_steps     = int(params.L / params.dz)

    # DMA 三角形転送関数の離散化
    Zp_array  = np.linspace(
        Zp_star * (1.0 - params.beta),
        Zp_star * (1.0 + params.beta),
        params.num_Zp_bins,
    )
    Omega_DMA = np.maximum(
        0.0,
        1.0 - np.abs(Zp_array - Zp_star) / (params.beta * Zp_star),
    )
    sum_Omega = np.sum(Omega_DMA)

    m_min   = params.m_min_fg * 1e-18
    m_max   = params.m_max_fg * 1e-18
    m_array = np.linspace(m_min, m_max, params.J)
    dm      = m_array[1] - m_array[0]

    I, J = data.I, params.J
    K    = np.zeros((I, J))

    print(f"2D 畳み込みカーネル行列 K_eff[{I}×{J}] を計算中...")
    for i, V in enumerate(data.V_array):
        V_term = V_term_factor * V
        for j, m in enumerate(m_array):
            K_eff_val = 0.0
            for zp, w_dma in zip(Zp_array, Omega_DMA):
                if w_dma > 0:
                    # Z_p ベースの係数: dz × (8π/3e) × Z_p × (δ·rc)/Q
                    # (Cc/D_mob = 3π·η·Z_p/e の関係を利用)
                    coef_zp = (
                        params.dz * (8.0 / 3.0) * (np.pi * zp / E_CHARGE)
                        * (delta * rc) / Q
                    )
                    K_eff_val += _rk4_transmission(
                        m, V, coef_zp, V_term, omega,
                        rc, delta, params.r1, params.r2,
                        params.nr0, num_steps,
                    ) * w_dma
            K[i, j] = (K_eff_val / sum_Omega) * dm
    print("計算完了！")

    return K, m_array
