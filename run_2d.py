"""
run_2d.py  ―  2D 畳み込みモデルによる DMA-APM 質量分布逆解析

1D モデルより厳密だが計算コストは約 num_Zp_bins 倍 (デフォルト 7 倍)。
実用上は 1D モデルと同等の結果が得られる (technical_note.md 参照)。

使い方:
    python run_2d.py

入力パラメータは params.py で設定してください。
"""
import os
import params
from data_parser      import load_and_bin
from kernel_simulator import build_kernel_2d
from inversion_solver import solve_chahine_twomey
from visualization    import fit_gaussian_mode, plot_and_save


def main() -> None:
    # 1. データ読み込み・前処理
    data = load_and_bin(params)

    # 2. 2D 畳み込みカーネル行列の構築
    K, m_array = build_kernel_2d(data, params)

    # 3. 逆問題解析
    f_estimated = solve_chahine_twomey(K, m_array, data, params)

    # 4. Gaussian フィッティング
    fit_result = fit_gaussian_mode(m_array, f_estimated)

    # 5. 結果の JPEG 保存
    stem        = os.path.splitext(os.path.basename(params.FILE_PATH))[0]
    output_path = os.path.join(params.OUTPUT_DIR, f"result_2d_{stem}.jpg")
    plot_and_save(data, K, m_array, f_estimated, fit_result, params, output_path)


if __name__ == "__main__":
    main()
