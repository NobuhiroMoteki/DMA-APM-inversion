"""
run_1d.py  ―  1D 近似モデルによる DMA-APM 質量分布逆解析

使い方:
    python run_1d.py

入力パラメータは params.py で設定してください。
"""
import os
import params
from data_parser      import load_and_bin
from kernel_simulator import build_kernel_1d
from inversion_solver import solve_chahine_twomey
from visualization    import fit_gaussian_mode, plot_and_save


def main() -> None:
    # 1. データ読み込み・前処理
    data = load_and_bin(params)

    # 2. 1D カーネル行列の構築
    K, m_array = build_kernel_1d(data, params)

    # 3. 逆問題解析
    f_estimated = solve_chahine_twomey(K, m_array, data, params)

    # 4. Gaussian フィッティング
    fit_result = fit_gaussian_mode(m_array, f_estimated)

    # 5. 結果の JPEG 保存
    stem        = os.path.splitext(os.path.basename(params.FILE_PATH))[0]
    output_path = os.path.join(params.OUTPUT_DIR, f"result_1d_{stem}.jpg")
    plot_and_save(data, K, m_array, f_estimated, fit_result, params, output_path)


if __name__ == "__main__":
    main()
