"""
visualization.py  ―  質量分布の Gaussian フィッティングと結果の JPEG 出力

支配的モードを Gauss + 線形背景モデルでフィットし、
中心位置 μ と標準偏差 σ をコンソールに出力する。
出力図は dpi=600 の JPEG ファイルとして保存する。
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from data_parser import MeasurementData

matplotlib.rcParams["font.family"] = "sans-serif"


# ------------------------------------------------------------------------------
# Gaussian フィッティング
# ------------------------------------------------------------------------------

@dataclass
class GaussianFitResult:
    """支配的モードの Gaussian フィット結果。

    Attributes:
        mu_fg:     ピーク中心位置 [fg]
        sigma_fg:  標準偏差 [fg]
        amplitude: ピーク高さ A [cm⁻³ fg⁻¹]
        offset:    定数オフセット B [cm⁻³ fg⁻¹]
        slope:     線形スロープ C [cm⁻³ fg⁻²]
        r_squared: 決定係数 R²
        success:   フィット成功フラグ
    """
    mu_fg:     float
    sigma_fg:  float
    amplitude: float
    offset:    float
    slope:     float
    r_squared: float
    success:   bool


def _gauss_linear(m: np.ndarray, A: float, mu: float, sigma: float,
                  B: float, C: float) -> np.ndarray:
    """Gauss + 線形背景モデル。

    f(m) = A · exp(−(m − μ)² / (2σ²)) + B + C·m
    """
    return A * np.exp(-0.5 * ((m - mu) / sigma) ** 2) + B + C * m


def fit_gaussian_mode(
    m_array:     np.ndarray,
    f_estimated: np.ndarray,
) -> GaussianFitResult:
    """質量分布の支配的モードを Gaussian + 線形背景でフィットする。

    フィットモデル:
        f(m) = A · exp(−(m − μ)² / (2σ²)) + B + C·m

    初期値: A = max(f),  μ = m[argmax(f)],  σ = (m_max − m_min)/6,  B = C = 0

    フィット失敗時は警告を表示し success=False の結果を返す。

    Args:
        m_array:     質量グリッド [kg], shape (J,)
        f_estimated: 推定質量分布 [cm⁻³ kg⁻¹], shape (J,)

    Returns:
        GaussianFitResult
    """
    m_fg = m_array * 1e18       # [kg] → [fg]
    f_fg = f_estimated * 1e-18  # [cm⁻³ kg⁻¹] → [cm⁻³ fg⁻¹]

    peak_idx = int(np.argmax(f_fg))
    A0     = float(f_fg[peak_idx])
    mu0    = float(m_fg[peak_idx])
    sigma0 = float((m_fg[-1] - m_fg[0]) / 6.0)

    try:
        popt, _ = curve_fit(
            _gauss_linear,
            m_fg,
            f_fg,
            p0=[A0, mu0, sigma0, 0.0, 0.0],
            bounds=(
                [0.0,    m_fg[0],  0.0,      -np.inf, -np.inf],
                [np.inf, m_fg[-1], m_fg[-1] - m_fg[0], np.inf, np.inf],
            ),
            maxfev=10000,
        )
        A, mu, sigma, B, C = popt
        sigma = abs(sigma)   # σ は正値

        f_fit  = _gauss_linear(m_fg, *popt)
        ss_res = float(np.sum((f_fg - f_fit) ** 2))
        ss_tot = float(np.sum((f_fg - np.mean(f_fg)) ** 2))
        r2     = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        result = GaussianFitResult(
            mu_fg=float(mu),
            sigma_fg=float(sigma),
            amplitude=float(A),
            offset=float(B),
            slope=float(C),
            r_squared=float(r2),
            success=True,
        )
        print("\n=== Gaussian Fit Result ===")
        print(f"  Peak center  μ  = {mu:.2f} fg")
        print(f"  Std. dev.    σ  = {sigma:.2f} fg")
        print(f"  Amplitude    A  = {A:.4f} cm⁻³ fg⁻¹")
        print(f"  Offset       B  = {B:.4f} cm⁻³ fg⁻¹")
        print(f"  Slope        C  = {C:.6f} cm⁻³ fg⁻²")
        print(f"  R²               = {r2:.4f}")
        return result

    except RuntimeError as exc:
        print(f"[警告] Gaussian フィット失敗: {exc}")
        return GaussianFitResult(
            mu_fg=mu0, sigma_fg=sigma0, amplitude=A0,
            offset=0.0, slope=0.0, r_squared=0.0, success=False,
        )


# ------------------------------------------------------------------------------
# 描画・保存
# ------------------------------------------------------------------------------

def plot_and_save(
    data:        MeasurementData,
    K:           np.ndarray,
    m_array:     np.ndarray,
    f_estimated: np.ndarray,
    fit_result:  GaussianFitResult,
    params,
    output_path: str,
) -> None:
    """結果を 2 パネル図として描画し JPEG (dpi=600) で保存する。

    左図: 実測 APM スペクトル (赤丸) vs 再構成シグナル (青実線)
    右図: 推定質量分布 dN/dm (青実線) + Gaussian フィット曲線 (赤破線)

    フィット成功時は右図凡例にフィット結果 (μ, σ, R²) を表示する。

    Args:
        data:        ビニング済み測定データ
        K:           カーネル行列
        m_array:     質量グリッド [kg]
        f_estimated: 推定質量分布 [cm⁻³ kg⁻¹]
        fit_result:  Gaussian フィット結果
        params:      ユーザー設定
        output_path: 出力ファイルパス (.jpg)
    """
    n_reconstructed = K @ f_estimated
    m_fg = m_array * 1e18        # [kg] → [fg]
    f_fg = f_estimated * 1e-18   # [cm⁻³ kg⁻¹] → [cm⁻³ fg⁻¹]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # ---- 左図: APM スペクトル ----
    ax1.plot(data.V_array, data.n_meas, "ro", markersize=5,
             label="Measured (Binned Average)")
    ax1.plot(data.V_array, n_reconstructed, "b-", linewidth=2, alpha=0.8,
             label="Reconstructed Signal")
    ax1.set_xlabel("APM Voltage [V]")
    ax1.set_ylabel("Concentration [cm$^{-3}$]")
    ax1.set_title(
        f"APM Spectrum  "
        f"($D_{{mob}}$ = {data.Dmob * 1e9:.0f} nm,  RPM = {data.RPM:.0f})"
    )
    ax1.legend(fontsize=9)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # ---- 右図: 質量分布 ----
    ax2.plot(m_fg, f_fg, "b-", linewidth=2, label="Estimated Distribution")

    if fit_result.success:
        m_dense = np.linspace(m_fg[0], m_fg[-1], 500)
        f_fit   = _gauss_linear(
            m_dense,
            fit_result.amplitude,
            fit_result.mu_fg,
            fit_result.sigma_fg,
            fit_result.offset,
            fit_result.slope,
        )
        label_fit = (
            f"Gaussian Fit\n"
            f"$\\mu$ = {fit_result.mu_fg:.1f} fg\n"
            f"$\\sigma$ = {fit_result.sigma_fg:.1f} fg\n"
            f"$R^2$ = {fit_result.r_squared:.3f}"
        )
        ax2.plot(m_dense, f_fit, "r--", linewidth=1.5, label=label_fit)

    ax2.set_xlabel("Particle Mass [fg]")
    ax2.set_ylabel("Mass Distribution  $dN/dm$  [cm$^{-3}$ fg$^{-1}$]")
    ax2.set_title("Reconstructed Mass Distribution")
    ax2.legend(fontsize=9)
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()

    out_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(output_path, dpi=600, format="jpeg", bbox_inches="tight")
    plt.close()
    print(f"\n図を保存しました: {output_path}")
