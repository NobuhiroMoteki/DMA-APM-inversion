# DMA-APM Data Inversion Tool

**The primary purpose of this tool is to determine the highly resolved mass distribution of aerosol particles that have been pre-classified by their electrical mobility.**

This Python-based data inversion tool evaluates the mass distribution of aerosol particles pre-classified by the DMA from the raw data obtained by the tandem DMA-APM-CPC (Differential Mobility Analyzer - Aerosol Particle Mass Analyzer - Condensation Particle Counter) system.

This tool features a proprietary numerical simulation approach for calculating the APM transfer function by tracing particle trajectories under a parabolic flow profile. It solves the inverse problem using a robust Chahine-Twomey iterative algorithm constrained by Poisson statistics. Both a rapid 1D approximation model and a rigorous 2D convolution model are documented and supported. For theoretical details, see the [Technical Note](technical_note.md).

<p align="center">
<img src="system_schematic.svg" alt="DMA–APM–CPC system schematic"/>
</p>
<p align="center"><em>Schematic diagram of the aerosol measurement system and data inversion procedure</em></p>

## 🌟 Key features
* **Automated Data Preprocessing**: Automatically extracts metadata from raw continuous APM scan data (CSV format) and intelligently bins the time-series data from both upward and downward voltage scans.
* **Rigorous Noise Evaluation**: Calculates the true integration time for each voltage bin by summing the residence times, ensuring accurate weighting of Poisson measurement noise based on the total sampled volume.
* **Original Kernel Simulation**: Uses an unpublished numerical algorithm utilizing the 4th-order Runge-Kutta (RK4) method to directly simulate particle trajectories within the APM gap, accounting for the parabolic flow profile.
* **Robust Inversion Algorithm**: Combines the non-negative Chahine-Twomey inversion method with Markowski's 1-2-1 smoothing to effectively suppress overfitting to experimental noise.
* **Gaussian Mode Analysis**: Automatically fits the dominant mode of the inverted mass distribution with a Gaussian + linear background model, reporting the peak center $\mu$ and standard deviation $\sigma$.

---

## 📖 Theoretical models

The fundamental relationship between the observed CPC concentration $n(V)$ and the target mass distribution $f(m)$ at the **DMA outlet** is an integral equation defined by the instrument's transfer function (kernel).

### The rigorous 2D-integral model
In reality, the raw aerosol $f_{in}(m, Z_p)$ enters the DMA, which transmits a finite distribution of electrical mobilities ($Z_p$). The target distribution $f(m)$ exiting the DMA is $f(m) = f_{in}(m, Z_p^*) \int \Omega_{DMA}(Z_p) dZ_p$.

To directly solve for this $f(m)$, the rigorous 2D effective kernel accounts for the DMA's transmission width by convolving the APM transfer function $\Omega_{APM}$ with the DMA's triangular transfer function $\Omega_{DMA}$:

$$n(V) = \int_{0}^{\infty} K_{eff}^{2D}(V, m) f(m) dm$$
$$K_{eff}^{2D}(V, m) = \frac{\int \Omega_{APM}(m, V, Z_p) \Omega_{DMA}(Z_p) dZ_p}{\int \Omega_{DMA}(Z_p) dZ_p}$$

### The 1D-integral approximation (standard use)
The 1D model approximates the DMA output as a Dirac delta function centered at the target mobility $Z_p^*$. Thus, the kernel is evaluated at a single mobility value:

$$n(V) \approx \int_{0}^{\infty} K^{1D}(V, m) f(m) dm$$
$$K^{1D}(V, m) = \Omega_{APM}(m, V, Z_p^*)$$

### Why the 1D-integral model is recommended
While the 2D model is mathematically more rigorous, **the 1D model is the standard choice for practical data analysis because it yields virtually identical results at a fraction of the computational cost.** This is because the APM acts as a pure mass classifier: its central classification mass ($m_c \propto V/\omega^2$) is fundamentally independent of $Z_p$. Variations in $Z_p$ caused by the DMA's finite resolution only slightly affect the transit time of the particles, causing a negligible change in the *width* of the APM transfer function without shifting its center. Consequently, the 2D convolution kernel $K_{eff}^{2D}$ is almost indistinguishable from $K^{1D}$. The 1D model provides excellent accuracy and stability for almost all experimental conditions.

---

## 🗂 File structure

```text
DMA-APM-inversion/
│
├── params.py              ★ User configuration (edit this file only)
│
├── data_parser.py         # CSV loading and voltage-scan binning
├── kernel_simulator.py    # RK4 particle trajectory simulation; kernel matrix construction
├── inversion_solver.py    # Chahine-Twomey iterative inversion
├── visualization.py       # Gaussian mode fitting and JPEG figure output
│
├── run_1d.py              # Entry point: 1D approximation model (recommended)
├── run_2d.py              # Entry point: 2D convolution model (rigorous)
│
├── test_consistency.py    # Numerical equivalence verification (developer use)
│
├── DMA-APM-CPC_inversion_test.ipynb      # 1D model test notebook (synthetic data)
└── DMA-APM-CPC_2Dinversion_test.ipynb    # 2D model test notebook (synthetic data)
```

### Module responsibilities

| Module | Key functions / classes |
| --- | --- |
| `data_parser.py` | `load_and_bin(params)` → `MeasurementData` dataclass |
| `kernel_simulator.py` | `build_kernel_1d(data, params)`, `build_kernel_2d(data, params)` |
| `inversion_solver.py` | `solve_chahine_twomey(K, m_array, data, params)` |
| `visualization.py` | `fit_gaussian_mode(m_array, f)` → `GaussianFitResult`, `plot_and_save(...)` |

---

## 🛠 Installation & requirements

The following Python libraries are required:
* `numpy`
* `pandas`
* `matplotlib`
* `scipy`

```bash
pip install numpy pandas matplotlib scipy
```

---

## 🚀 Usage

### 1. Data preparation

Prepare the time-series data output from your APM control software (CSV format). The script expects the following structure:

* Lines 1-8: Header information (must include `Electrical Mobility Diameter`).
* Line 9: Column names (`Time`, `Rotation Speed`, `Applied Voltage`, `Outlet Particle Concentration`).
* Line 10+: Time-series measurement data.

### 2. Edit `params.py`

Open `params.py` and set your experimental conditions. **This is the only file you need to edit.**

```python
# --- Input file ---
FILE_PATH  = "./APM-CPC_data/.../your_data_file.csv"
OUTPUT_DIR = "./results"        # Output directory for JPEG figures

# --- APM geometry [m] ---
L   = 100.0e-3    # Electrode length
r1  = 24.0e-3     # Inner cylinder radius
r2  = 25.0e-3     # Outer cylinder radius

# --- Operating conditions ---
Q_a_lpm = 0.3     # APM aerosol flow rate [L/min]
Q_cpc_lpm = 0.3   # CPC flow rate [L/min]

# --- Inversion parameters ---
m_min_fg = 5.0    # Mass range lower bound [fg]
m_max_fg = 80.0   # Mass range upper bound [fg]
J        = 40     # Number of mass bins
```

### 3. Run the analysis

```bash
# 1D model (recommended for standard analysis)
python run_1d.py

# 2D model (rigorous; ~7x slower than 1D)
python run_2d.py
```

### 4. Output

**Console output** — convergence log and Gaussian fit result:

```text
=== データ読み込み完了 ===
  有効電圧ビン数 I = 28
  APM 回転数 (平均): 4010.0 rpm
  移動度粒径 Dmob  = 450.0 nm
1D カーネル行列 K[28×40] を計算中...
逆問題解析 (Chahine-Twomey 反復) を開始します...
  → 312 反復で収束 (χ² = 0.9987)

=== Gaussian Fit Result ===
  Peak center  μ  = 24.35 fg
  Std. dev.    σ  =  8.12 fg
  Amplitude    A  =  0.1273 cm⁻³ fg⁻¹
  R²               =  0.994

図を保存しました: ./results/result_1d_your_data_file.jpg
```

**JPEG figure** (dpi = 600) saved to `OUTPUT_DIR`:

* **Left panel (APM Spectrum)**: Binned measured data (red dots) vs. the reconstructed signal (blue line).
* **Right panel (Mass Distribution)**: Inverted $dN/dm$ in femtograms (blue line) with the Gaussian fit overlaid (red dashed line). The fit reports peak center $\mu$, standard deviation $\sigma$, and $R^2$.

### 5. Test notebooks (synthetic data)

Two Jupyter notebooks are provided for algorithm development and testing **without requiring real measurement data**. Both use synthetic aerosol distributions (log-normal) with simulated Poisson counting noise as input, so no CSV file is needed.

| Notebook | Model |
| --- | --- |
| `DMA-APM-CPC_Test-1D-inversion.ipynb` | 1D approximation |
| `DMA-APM-CPC_Test-2D-inversion.ipynb` | 2D convolution |

These notebooks are self-contained and replicate the full inversion pipeline — kernel construction, Chahine-Twomey iteration, and visualization — entirely within the notebook environment. They are useful for exploring parameter sensitivity and verifying physical correctness before applying the tool to real data.

---

## 📚 References

1. **Ehara et al. (1996).** *J. Aerosol Sci.* doi:10.1016/0021-8502(96)00014-4
2. **Twomey, S. (1975).** *J. Comput. Phys.* doi:10.1016/0021-9991(75)90028-5
3. **Markowski, G. R. (1987).** *Aerosol Sci. Technol.* doi:10.1080/02786828708959153
4. **Hinds, W. C. (1999).** *Aerosol Technology: Properties, Behavior, and Measurement of Airborne Particles* (2nd ed.). Wiley. Eq. (3.22)

## 📝 License

[MIT License](https://www.google.com/search?q=LICENSE)
