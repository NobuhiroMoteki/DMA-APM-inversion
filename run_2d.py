"""
run_2d.py  ---  DMA-APM mass distribution inversion using the 2D convolution model

More rigorous than the 1D model but approximately num_Zp_bins times slower
(7x by default). Yields virtually identical results to the 1D model in practice
(see technical_note.md).

Usage:
    python run_2d.py

Set all input parameters in params.py before running.
"""
import os
import params
from data_parser      import load_and_bin
from kernel_simulator import build_kernel_2d
from inversion_solver import solve_chahine_twomey
from visualization    import fit_gaussian_mode, plot_and_save


def main() -> None:
    # 1. Load and preprocess data
    data = load_and_bin(params)

    # 2. Build the 2D convolution kernel matrix
    K, m_array = build_kernel_2d(data, params)

    # 3. Solve the inverse problem
    f_estimated = solve_chahine_twomey(K, m_array, data, params)

    # 4. Gaussian fitting
    fit_result = fit_gaussian_mode(m_array, f_estimated)

    # 5. Save result as JPEG
    stem        = os.path.splitext(os.path.basename(params.FILE_PATH))[0]
    output_path = os.path.join(params.OUTPUT_DIR, f"result_2d_{stem}.jpg")
    plot_and_save(data, K, m_array, f_estimated, fit_result, params, output_path)


if __name__ == "__main__":
    main()
