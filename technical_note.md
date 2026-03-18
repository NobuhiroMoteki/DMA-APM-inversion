# Technical Note: Data Inversion Theory for the DMA-APM-CPC Tandem Aerosol Measurement System

**Author:** N. Moteki
**Last updated:** 2026-03-18 (corresponds to DMA-APM-inversion v0.1.2)

This document provides a self-contained description of the mathematical theory and numerical algorithms implemented in the DMA-APM mass distribution inversion tool. It covers the formulation of the forward problem (1D and 2D integral models), the physical justification for the 1D approximation, the original RK4-based APM transfer function simulation, the Chahine-Twomey inversion algorithm with Markowski smoothing, and the Poisson-statistics-based convergence criterion.

## Contents

1. [Objective and problem statement](#1-objective-and-problem-statement)
2. [Formulation of the forward problem](#2-formulation-of-the-forward-problem)
3. [Physical justification: why the 1D model is practically sufficient](#3-physical-justification-why-the-1d-model-is-practically-sufficient)
4. [APM transfer function: particle trajectory simulation](#4-apm-transfer-function-particle-trajectory-simulation)
5. [Inversion algorithm: Chahine-Twomey method with internal smoothing](#5-inversion-algorithm-chahine-twomey-method-with-internal-smoothing)
6. [Poisson variance and convergence criterion](#6-poisson-variance-and-convergence-criterion)
7. [Data preprocessing: voltage binning](#7-data-preprocessing-voltage-binning)
8. [Implementation mapping](#8-implementation-mapping)
9. [References](#9-references)

---

## 1. Objective and problem statement

The objective of this analysis is to estimate the highly resolved mass distribution of aerosol particles that have been pre-classified by a specific electrical mobility diameter using a DMA-APM-CPC tandem measurement system. Specifically, we seek the mass distribution function at the DMA outlet:

```math
f(m) = \frac{dN}{dm}, \qquad \textrm{(1)}
```

where $N$ is the particle number concentration and $m$ is the particle mass. The observed CPC concentration $n(V)$ at each APM applied voltage $V$ is related to $f(m)$ through a Fredholm integral equation of the first kind, and recovering $f(m)$ from $n(V)$ constitutes an ill-posed inverse problem.

## 2. Formulation of the forward problem

### 2.1 The rigorous 2D-integral model

The expected particle number concentration $n(V)$ observed downstream of the APM is governed by a double integral over both mass $m$ and electrical mobility $Z_{\mathrm{p}}$:

```math
n(V) = \int_{0}^{\infty} \int_{0}^{\infty} \Omega_{\mathrm{APM}}(m, V, Z_{\mathrm{p}}) \, f_{\mathrm{in}}(m, Z_{\mathrm{p}}) \, \Omega_{\mathrm{DMA}}(Z_{\mathrm{p}}) \, dZ_{\mathrm{p}} \, dm, \qquad \textrm{(2)}
```

where $f_{\mathrm{in}}(m, Z_{\mathrm{p}})$ is the intrinsic 2D mass-mobility distribution of the raw aerosol entering the DMA, $\Omega_{\mathrm{DMA}}(Z_{\mathrm{p}})$ is the DMA transfer function, and $\Omega_{\mathrm{APM}}(m, V, Z_{\mathrm{p}})$ is the APM transfer function.

Assuming that $f_{\mathrm{in}}$ is approximately constant with respect to $Z_{\mathrm{p}}$ over the narrow DMA transmission window, we define the target distribution $f(m)$ as the mass distribution of the aerosol exiting the DMA:

```math
f(m) = f_{\mathrm{in}}(m, Z_{\mathrm{p}}^*) \int_{0}^{\infty} \Omega_{\mathrm{DMA}}(Z_{\mathrm{p}}) \, dZ_{\mathrm{p}}. \qquad \textrm{(3)}
```

Using this definition, the forward problem reduces to:

```math
n(V) = \int_{0}^{\infty} K_{\mathrm{eff}}^{2D}(V, m) \, f(m) \, dm, \qquad \textrm{(4)}
```

where the effective 2D kernel is:

```math
K_{\mathrm{eff}}^{2D}(V, m) = \frac{\int_{0}^{\infty} \Omega_{\mathrm{APM}}(m, V, Z_{\mathrm{p}}) \, \Omega_{\mathrm{DMA}}(Z_{\mathrm{p}}) \, dZ_{\mathrm{p}}}{\int_{0}^{\infty} \Omega_{\mathrm{DMA}}(Z_{\mathrm{p}}) \, dZ_{\mathrm{p}}}. \qquad \textrm{(5)}
```

In the implementation, the DMA transfer function is modelled as a triangular function centered at the target mobility $Z_{\mathrm{p}}^*$ with half-width $\beta Z_{\mathrm{p}}^*$, where $\beta = Q_{\mathrm{sample}}/Q_{\mathrm{sheath}}$ is the DMA flow ratio. The integral in Eq. (5) is evaluated numerically using the rectangle rule with a configurable number of quadrature points.

### 2.2 The 1D-integral approximation

By approximating the DMA output as a Dirac delta function centered at the target mobility $Z_{\mathrm{p}}^*$, the effective kernel simplifies to:

```math
K^{1D}(V, m) = \Omega_{\mathrm{APM}}(m, V, Z_{\mathrm{p}}^*), \qquad \textrm{(6)}
```

and the forward problem becomes:

```math
n(V) \approx \int_{0}^{\infty} K^{1D}(V, m) \, f(m) \, dm. \qquad \textrm{(7)}
```

In both the 1D and 2D models, the inversion algorithm solves for the same target: $f(m)$, the mass distribution of the particles at the DMA outlet.

## 3. Physical justification: why the 1D model is practically sufficient

In practice, the mass distributions obtained from the 1D and 2D models are virtually identical, even when the DMA resolution is relatively broad (e.g., $\beta = 0.2$). This equivalence arises from the fundamental classification principle of the APM. The central classified mass is:

```math
m_{\mathrm{c}} = \frac{e V}{r_{\mathrm{c}}^2 \, \omega^2 \, \ln(r_2 / r_1)}, \qquad \textrm{(8)}
```

where $e$ is the elementary charge, $r_{\mathrm{c}} = (r_1 + r_2)/2$ is the gap centre radius, $\omega$ is the angular velocity, and $r_1$, $r_2$ are the inner and outer cylinder radii. This expression is completely independent of the particle's electrical mobility $Z_{\mathrm{p}}$. A variation in $Z_{\mathrm{p}}$ (due to the DMA's finite transmission width) only alters the particle's radial drift velocity inside the APM, slightly affecting its transit time. Consequently, $Z_{\mathrm{p}}$ variations marginally broaden the width of $\Omega_{\mathrm{APM}}$ but do not shift its central peak. Because the 2D convolution averages these minor, symmetric width changes, the resulting $K_{\mathrm{eff}}^{2D}$ is nearly indistinguishable from $K^{1D}$.

The 1D approximation therefore provides mathematically and physically robust results at approximately $1/N_{Z_{\mathrm{p}}}$ of the computational cost of the 2D model, where $N_{Z_{\mathrm{p}}}$ is the number of $Z_{\mathrm{p}}$ quadrature points.

## 4. APM transfer function: particle trajectory simulation

### 4.1 Equation of motion

To compute $\Omega_{\mathrm{APM}}(m, V, Z_{\mathrm{p}})$, this tool employs an original numerical simulation method that directly tracks particle trajectories within the APM gap using the 4th-order Runge-Kutta (RK4) method.

Inside the APM annular gap (inner radius $r_1$, outer radius $r_2$, centre radius $r_{\mathrm{c}} = (r_1 + r_2)/2$, half-width $\delta = (r_2 - r_1)/2$), a particle of mass $m$ and electrical mobility $Z_{\mathrm{p}}$ experiences centrifugal and electrostatic forces. Under the assumption of a fully developed parabolic laminar flow profile, the ratio of the radial to axial velocity components is (Ehara et al., 1996):

```math
\frac{dr}{dz} = \frac{v_{\mathrm{r}}(r)}{v_{\mathrm{z}}(r)} = \frac{\frac{Z_{\mathrm{p}}}{e} \left( m \omega^2 r - \frac{eV}{r \ln(r_2/r_1)} \right)}{\frac{3Q}{4\pi \delta r_{\mathrm{c}}} \left[ 1 - \left(\frac{r - r_{\mathrm{c}}}{\delta}\right)^2 \right]}, \qquad \textrm{(9)}
```

where $Q$ is the volumetric aerosol flow rate. This can be rearranged to:

```math
\frac{dr}{dz} = \frac{8\pi \delta r_{\mathrm{c}}}{3Q} \frac{Z_{\mathrm{p}}}{e} \cdot \frac{m \omega^2 r - \frac{eV}{r \ln(r_2/r_1)}}{1 - \left(\frac{r - r_{\mathrm{c}}}{\delta}\right)^2}. \qquad \textrm{(10)}
```

### 4.2 RK4 integration and transmission efficiency

The trajectory equation (Eq. 10) is integrated along the axial coordinate $z$ from $z = 0$ to $z = L$ (the electrode length) using the classical 4th-order Runge-Kutta method with a fixed step size $\Delta z$:

```math
k_1 = \Delta z \cdot g(r_n), \qquad \textrm{(11a)}
```

```math
k_2 = \Delta z \cdot g(r_n + k_1/2), \qquad \textrm{(11b)}
```

```math
k_3 = \Delta z \cdot g(r_n + k_2/2), \qquad \textrm{(11c)}
```

```math
k_4 = \Delta z \cdot g(r_n + k_3), \qquad \textrm{(11d)}
```

```math
r_{n+1} = r_n + \frac{1}{6}(k_1 + 2k_2 + 2k_3 + k_4), \qquad \textrm{(11e)}
```

where $g(r) \equiv dr/dz$ as defined in Eq. (10).

A total of $N_{r_0}$ particles are launched from uniformly spaced initial radial positions $r_0 \in [r_1, r_2]$. A particle is considered lost if it hits the inner or outer electrode wall at any step. The transmission efficiency is computed as the flux-weighted fraction of surviving particles:

```math
\Omega_{\mathrm{APM}} = \frac{\sum_{r_0 \in \text{survived}} w(r_0)}{\sum_{r_0 \in \text{all}} w(r_0)}, \qquad \textrm{(12)}
```

where the weight $w(r_0)$ accounts for the parabolic flow profile:

```math
w(r_0) = \frac{3}{2} \left[ 1 - \left(\frac{r_0 - r_{\mathrm{c}}}{\delta}\right)^2 \right]. \qquad \textrm{(13)}
```

### 4.3 Cunningham slip correction

The electrical mobility $Z_{\mathrm{p}}$ is related to the mobility diameter $D_{\mathrm{mob}}$ via:

```math
Z_{\mathrm{p}} = \frac{e \, C_c(D_{\mathrm{mob}})}{3 \pi \eta D_{\mathrm{mob}}}, \qquad \textrm{(14)}
```

where $\eta$ is the dynamic viscosity of air and $C_c$ is the Cunningham slip correction factor (Hinds, 1999):

```math
C_c = 1 + \frac{1}{P d} \left[ 15.60 + 7.00 \exp(-0.059 \, P d) \right], \qquad \textrm{(15)}
```

with $P$ the atmospheric pressure in kPa and $d$ the diameter in $\mu$m.

### 4.4 Kernel matrix discretisation

For the discrete inversion, the integral in Eq. (7) is approximated by the rectangle rule over $J$ mass bins:

```math
n(V_i) \approx \sum_{j=1}^{J} K_{i,j} \, f(m_j), \qquad \textrm{(16)}
```

where $K_{i,j} = \Omega_{\mathrm{APM}}(m_j, V_i, Z_{\mathrm{p}}^*) \cdot \Delta m$ for the 1D model, $m_j$ are uniformly spaced mass grid points, and $\Delta m = (m_{\mathrm{max}} - m_{\mathrm{min}})/(J - 1)$. The kernel matrix $\mathbf{K}$ has shape $I \times J$, where $I$ is the number of voltage bins.

## 5. Inversion algorithm: Chahine-Twomey method with internal smoothing

### 5.1 Update equation

To solve the discrete ill-posed problem $\mathbf{n} = \mathbf{K} \mathbf{f}$, we use the non-negative Chahine-Twomey iterative algorithm (Twomey, 1975). The update rule at iteration $k$ is:

```math
f_j^{(k+1)} = f_j^{(k)} \cdot \frac{\sum_{i=1}^{I} \hat{K}_{i,j} \left( n_{\mathrm{meas},i} \,/\, n_{\mathrm{calc},i}^{(k)} \right)}{\sum_{i=1}^{I} \hat{K}_{i,j}}, \qquad \textrm{(17)}
```

where $\hat{K}_{i,j} = K_{i,j} / \max(\mathbf{K})$ is the normalised kernel, $n_{\mathrm{meas},i}$ is the measured concentration in the $i$-th voltage bin, and $n_{\mathrm{calc},i}^{(k)} = \sum_j K_{i,j} f_j^{(k)}$ is the forward-calculated concentration at iteration $k$.

The initial guess is a uniform distribution:

```math
f_j^{(0)} = \frac{\sum_{i=1}^{I} n_{\mathrm{meas},i}}{m_{\mathrm{max}} - m_{\mathrm{min}}}, \quad j = 1, \ldots, J. \qquad \textrm{(18)}
```

### 5.2 Markowski 1-2-1 smoothing

After each iteration, a three-point moving average filter (Markowski, 1987) is applied to suppress high-frequency oscillations caused by noise amplification:

```math
\tilde{f}_j = \frac{1}{4} f_{j-1} + \frac{1}{2} f_j + \frac{1}{4} f_{j+1}, \quad j = 2, \ldots, J-1. \qquad \textrm{(19)}
```

The endpoints ($j = 1$ and $j = J$) are not smoothed.

## 6. Poisson variance and convergence criterion

### 6.1 Poisson counting statistics

The CPC measures particle counts $N_i$ during the integration time $t_{\mathrm{meas},i}$ at each voltage bin. The measured concentration is $n_{\mathrm{meas},i} = N_i / V_{\mathrm{sample},i}$, where $V_{\mathrm{sample},i} = Q_{\mathrm{CPC}} \cdot t_{\mathrm{meas},i}$ is the total sampled air volume. Because particle counting follows Poisson statistics, the variance of the measured concentration is:

```math
\textrm{Var}(n_i) = \frac{n_{\mathrm{calc},i}}{V_{\mathrm{sample},i}}. \qquad \textrm{(20)}
```

In practice, a floor is applied to prevent division by zero:

```math
\textrm{Var}(n_i) = \max\!\left(\frac{n_{\mathrm{calc},i}}{V_{\mathrm{sample},i}}, \;\frac{1}{V_{\mathrm{sample},i}^2}\right). \qquad \textrm{(21)}
```

### 6.2 Reduced chi-squared criterion

Convergence is determined by the reduced chi-squared statistic:

```math
\chi^2 = \frac{1}{I} \sum_{i=1}^{I} \frac{\left(n_{\mathrm{meas},i} - n_{\mathrm{calc},i}\right)^2}{\textrm{Var}(n_i)}. \qquad \textrm{(22)}
```

The iteration terminates when $\chi^2 < \chi^2_{\mathrm{threshold}}$. The default threshold is $\chi^2_{\mathrm{threshold}} = 1.0$, corresponding to the expectation that residuals should be comparable to the measurement noise. This criterion prevents overfitting: stopping at $\chi^2 \approx 1$ ensures that the solution explains the data to within the precision allowed by Poisson counting noise.

## 7. Data preprocessing: voltage binning

### 7.1 Merging upward and downward scans

The raw APM data consists of time-series measurements during continuous voltage scanning (both upward and downward sweeps). The data are binned into $I$ voltage bins by grouping measurements with similar applied voltages.

### 7.2 Binning rules for physical quantities

The binning procedure strictly distinguishes between intensive and extensive variables:

- **Intensive variables** (state quantities): Applied voltage $V$ and particle concentration $n$ are averaged within each bin using the arithmetic mean.
- **Extensive variables** (amount quantities): The measurement time $t_{\mathrm{meas}}$ for each bin is obtained by summing the individual time intervals $\Delta t$ of each data row assigned to that bin.

This distinction is critical for the correct evaluation of $V_{\mathrm{sample},i} = Q_{\mathrm{CPC}} \cdot t_{\mathrm{meas},i}$, which directly enters the Poisson variance calculation (Eq. 21).

## 8. Implementation mapping

| Theory | Implementation | File |
| --- | --- | --- |
| Cunningham correction, Eq. (15) | `_cunningham(Dmob)` | [kernel_simulator.py](kernel_simulator.py) |
| Trajectory equation, Eq. (10) | `dr_dz(rad)` inside `_rk4_transmission` | [kernel_simulator.py](kernel_simulator.py) |
| RK4 integration, Eq. (11) | `_rk4_transmission(...)` | [kernel_simulator.py](kernel_simulator.py) |
| Transmission efficiency, Eq. (12)-(13) | `_rk4_transmission(...)` | [kernel_simulator.py](kernel_simulator.py) |
| 1D kernel matrix, Eq. (16) | `build_kernel_1d(data, params)` | [kernel_simulator.py](kernel_simulator.py) |
| 2D effective kernel, Eq. (5) | `build_kernel_2d(data, params)` | [kernel_simulator.py](kernel_simulator.py) |
| Chahine-Twomey update, Eq. (17) | `solve_chahine_twomey(...)` | [inversion_solver.py](inversion_solver.py) |
| Markowski smoothing, Eq. (19) | `solve_chahine_twomey(...)` | [inversion_solver.py](inversion_solver.py) |
| Poisson variance, Eq. (21) | `solve_chahine_twomey(...)` | [inversion_solver.py](inversion_solver.py) |
| Chi-squared criterion, Eq. (22) | `solve_chahine_twomey(...)` | [inversion_solver.py](inversion_solver.py) |
| Voltage binning, Sec. 7 | `load_and_bin(params)` | [data_parser.py](data_parser.py) |
| Gaussian mode fitting | `fit_gaussian_mode(m_array, f)` | [visualization.py](visualization.py) |

## 9. References

1. K. Ehara, C. Hagwood, and K. J. Coakley, "Novel method to classify aerosol particles according to their mass-to-charge ratio---Aerosol particle mass analyser," *J. Aerosol Sci.*, vol. 27, no. 2, pp. 217--234, 1996. DOI: [10.1016/0021-8502(96)00014-4](https://doi.org/10.1016/0021-8502(96)00014-4).
2. S. Twomey, "Comparison of constrained linear inversion and an iterative nonlinear algorithm applied to the indirect estimation of particle size distributions," *J. Comput. Phys.*, vol. 18, no. 2, pp. 188--200, 1975. DOI: [10.1016/0021-9991(75)90028-5](https://doi.org/10.1016/0021-9991(75)90028-5).
3. G. R. Markowski, "Improving Twomey's algorithm for inversion of aerosol measurement data," *Aerosol Sci. Technol.*, vol. 7, no. 2, pp. 127--141, 1987. DOI: [10.1080/02786828708959153](https://doi.org/10.1080/02786828708959153).
4. W. C. Hinds, *Aerosol Technology: Properties, Behavior, and Measurement of Airborne Particles*, 2nd ed. Wiley, 1999. Eq. (3.22).

## Acknowledgment

This document was prepared with the assistance of Claude (Anthropic). The author assumes full responsibility for the content.
