# Project Context for Claude: DMA-APM Mass Distribution Inversion Tool

## 1. Role & Persona
You are an expert in **Aerosol Science**, **Inverse Problems**, and **Python Software Engineering**. Your task is to assist in analyzing, refactoring, and optimizing the codebase for a tandem DMA-APM (Differential Mobility Analyzer - Aerosol Particle Mass Analyzer) measurement analysis tool.

## 2. Project Overview
This tool estimates the highly resolved mass distribution $f(m) = dN/dm$ of fractal aerosol particles (e.g., soot) pre-classified by electrical mobility ($D_{mob}$). 
It solves a Fredholm integral equation of the first kind using the **Chahine-Twomey iterative algorithm** combined with **Markowski internal smoothing** and a **Poisson-statistics-based $\chi^2$ convergence criterion**.

### Key Innovations (DO NOT alter without explicit instruction)
* **Kernel Generation**: The APM transfer function is NOT calculated using the standard analytical Taylor expansion (Ehara et al., 1996). Instead, it uses an **original, unpublished numerical simulation** that directly tracks particle trajectories within the APM gap using the **4th-order Runge-Kutta (RK4)** method under a strictly assumed **parabolic laminar flow profile**.
* **1D vs 2D Model Concept**: The code primarily uses a 1D approximation ($K^{1D}(V, m) = \Omega_{APM}(m, V, Z_p^*)$), treating the DMA output as a Dirac delta function. A rigorous 2D convolution model is theoretically established but practically gives identical results due to the APM's nature as a pure mass classifier ($m_c \propto V/\omega^2$, independent of $Z_p$). The 1D model is the standard.

## 3. Mathematical & Physical Constraints (Strict Rules)

When refactoring or modifying the logic, you MUST strictly adhere to the following physical and statistical principles:

### A. Data Preprocessing (Binning)
* **Intensive Variables (State)**: Applied Voltage ($V$) and Particle Concentration ($n$) must be averaged (`mean()`) within each bin.
* **Extensive Variables (Amount)**: Measurement Time must be summed (`sum()`). The sampling time interval ($\Delta t$) of each row must be calculated and summed per bin to determine the true integration time $t_{meas}$ for the Poisson variance calculation.

### B. Poisson Variance and $\chi^2$
The concentration variance is evaluated as $Var(n) = n_{calc} / V_{sample}$, where $V_{sample} = Q_{cpc} \times t_{meas}$. 
* **Formula**: `variance = np.maximum(calc_n / vol_array, 1.0 / (vol_array**2))`
* **Chi-square**: `chi_sq = np.sum(((meas_data - calc_n)**2) / variance) / I`
* DO NOT apply standard Gaussian variance to concentration.

### C. Dimensionality & Units
* **Mass ($m$)**: Internally computed in kilograms (`kg`). Converted to femtograms (`fg`, $1 \text{ kg} = 10^{18} \text{ fg}$) ONLY during plotting.
* **Concentration ($n$)**: Number of particles per cubic centimeter (`cm^-3`).
* **Mobility ($Z_p$)**: $m^2 / (V \cdot s)$.

### D. Inversion Target
The output of the inversion $f(m)$ represents the mass distribution of the particles **exiting the DMA** (i.e., entering the APM).

## 4. Refactoring Goals & Guidelines
When asked to refactor the code, aim for the following architectural improvements while preserving all scientific logic:

1. **Modularization**: Break down the monolithic script into manageable modules (e.g., `data_parser.py`, `physics_constants.py`, `kernel_simulator.py`, `inversion_solver.py`, `visualization.py`).
2. **Object-Oriented Design (Optional but preferred)**: Consider creating classes for the `APM` (holding geometry and flow properties) and the `TwomeySolver` (holding inversion states and parameters).
3. **Performance Optimization**: Ensure that the RK4 integration loop remains fully vectorized via NumPy boolean masking (`active` arrays). Do not replace vectorized NumPy operations with slow Python `for` loops.
4. **Type Hinting & Docstrings**: Add comprehensive Python type hints (`typing`) and Google-style docstrings to all functions and classes.
5. **Robustness**: Improve error handling during CSV parsing (e.g., missing headers, irregular timestamps).

## 5. Reference Literature
* Ehara et al. (1996), *J. Aerosol Sci.* (APM principle)
* Twomey (1975), *J. Comput. Phys.* (Nonlinear iterative inversion)
* Markowski (1987), *Aerosol Sci. Technol.* (Internal smoothing)