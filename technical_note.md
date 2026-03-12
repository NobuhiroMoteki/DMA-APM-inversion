# Technical Note: Data inversion algorithms for the DMA-APM-CPC measurement system

## 1. Objective and problem statement
The objective of this analysis is to estimate the highly resolved mass distribution $f(m) = \frac{dN}{dm}$ of aerosol particles (such as soot fractal aggregates) that have been pre-classified by a specific mobility diameter $D_{mob}$ using a DMA-APM-CPC tandem setup.

While the APM classifies particles based on both their mass $m$ and electrical mobility $Z_p$, the upstream DMA narrows the mobility distribution. The observed particle number concentration measured by the CPC at an APM applied voltage $V$ is mathematically described by a Fredholm integral equation of the first kind as follows.

## 2. Formulation of the forward problem

### The rigorous 2D-integral model and the target distribution $f(m)$
The expected particle number concentration $n(V)$ **observed downstream of the APM (measured by the CPC)** is governed by a double integral over both mass $m$ and electrical mobility $Z_p$:

$$n(V) = \int_{0}^{\infty} \int_{0}^{\infty} \Omega_{APM}(m, V, Z_p) f_{in}(m, Z_p) \Omega_{DMA}(Z_p) dZ_p dm$$

Where:
* $f_{in}(m, Z_p)$ is the intrinsic 2D mass-mobility distribution of the raw aerosol **entering the DMA** (upstream of the DMA).
* $\Omega_{DMA}(Z_p)$ is the transfer function of the DMA.
* $\Omega_{APM}(m, V, Z_p)$ is the transfer function of the APM.

Assuming that $f_{in}$ is relatively constant with respect to $Z_p$ over the narrow transmission window of the DMA, we can factor it out of the $Z_p$ integral. The ultimate goal of our inversion is to find $f(m)$, which is the mass distribution of the aerosol **exiting the DMA** (entering the APM). This target distribution is mathematically defined as:

$$f(m) = f_{in}(m, Z_p^*) \int_{0}^{\infty} \Omega_{DMA}(Z_p) dZ_p$$

Using this definition, the 2D forward problem can be elegantly rewritten as a single integral over $m$:

$$n(V) = \int_{0}^{\infty} K_{eff}^{2D}(V, m) f(m) dm$$

Where the rigorous 2D effective kernel is defined by convolving the APM transfer function with the DMA transfer function:

$$K_{eff}^{2D}(V, m) = \frac{\int_{0}^{\infty} \Omega_{APM}(m, V, Z_p) \Omega_{DMA}(Z_p) dZ_p}{\int_{0}^{\infty} \Omega_{DMA}(Z_p) dZ_p}$$

### The 1D-integral approximation
The 1D model is derived as a practical approximation of the 2D model. By assuming the DMA acts as a perfect Dirac delta-function filter centered at the target mobility $Z_p^*$, the effective kernel simplifies to:

$$K^{1D}(V, m) = \Omega_{APM}(m, V, Z_p^*)$$

Thus, the 1D forward problem simplifies to:

$$n(V) \approx \int_{0}^{\infty} K^{1D}(V, m) f(m) dm$$

**In both the 1D and 2D models, the inversion algorithm solves for the exact same target: $f(m)$, the mass distribution of the particles at the DMA outlet.**

## 3. Physical justification: why the 1D model is practically sufficient
In practice, the mass distributions inverted using the 1D model and the 2D model are virtually identical, even when the DMA resolution is relatively broad (e.g., flow ratio $\beta = 0.2$). 

This is due to the fundamental classification principle of the APM. The central classified mass $m_c$ is determined solely by the balance between the centrifugal and electrostatic forces inside the APM:

$$m_c = \frac{e V}{r^2 \omega^2 \ln(r_2/r_1)}$$

**This equation is completely independent of the particle's electrical mobility $Z_p$.** A variation in $Z_p$ (due to the DMA's finite transmission width) only alters the particle's radial velocity, slightly affecting its transit time through the APM gap. Consequently, $Z_p$ variations marginally broaden the *width* (resolution) of $\Omega_{APM}$ but do not shift its central peak location. Because the convolution in the 2D model averages these symmetrical, minor changes in width, the resulting $K_{eff}^{2D}$ is nearly indistinguishable from $K^{1D}$. 

Therefore, the 1D approximation provides mathematically and physically robust results with a significantly reduced computational load (approximately 1/15th the calculation time of the 2D model), making it the optimal standard for practical data analysis.

## 4. Kernel matrix construction: particle trajectory simulation (original method)
To calculate $\Omega_{APM}(m, V, Z_p)$, this analysis employs an **unpublished, original numerical simulation method**. It directly tracks particle trajectories within the APM gap using the 4th-order Runge-Kutta (RK4) method, assuming a parabolic axial laminar flow profile.

Inside the APM gap (inner radius $r_1$, outer radius $r_2$, center radius $r_c$, and half-width $\delta$), the ratio of radial to axial velocity components is given by:

$$\frac{dr}{dz} = \frac{8 \delta r_c}{3 Q} \frac{\pi Z_p}{e} \frac{m \omega^2 r - \frac{e V}{r \ln(r_2/r_1)}}{1 - \left(\frac{r-r_c}{\delta}\right)^2}$$

The transmission efficiency is computed as the fraction of the initial flux (weighted by the parabolic velocity profile) of particles that traverse the APM length $L$ without colliding with the electrodes.

## 5. Inversion algorithm: Chahine-Twomey method with internal smoothing
To solve the ill-posed matrix equation $\mathbf{n} = \mathbf{K}\mathbf{f}$, we use the non-negative Chahine-Twomey iterative algorithm combined with Markowski internal smoothing (Twomey, 1975; Markowski, 1987).

**1. Update Equation:**
$$f_j^{(k+1)} = f_j^{(k)} \frac{\sum_{i=1}^{I} K_{i,j} \left( \frac{n_{meas,i}}{n_{calc,i}^{(k)}} \right)}{\sum_{i=1}^{I} K_{i,j}}$$

**2. Smoothing & Convergence:**
A 1-2-1 moving average filter is applied between adjacent mass bins after each iteration. Convergence is determined using a $\chi^2$ test based on Poisson counting statistics, utilizing the true integrated sampling volume $V_{sample,i}$ to calculate the variance $Var(n_i) = n_{calc,i} / V_{sample,i}$. The integration time used for this volume calculation is rigorously obtained by summing the time intervals $\Delta t$ spent in each voltage bin.

## 6. References
1. **Ehara, K., Hagwood, C., & Coakley, K. J. (1996).** Novel method to classify aerosol particles according to their mass-to-charge ratio—Aerosol particle mass analyser. *Journal of Aerosol Science*, 27(2), 217-234. doi:10.1016/0021-8502(96)00014-4
2. **Twomey, S. (1975).** Comparison of constrained linear inversion and an iterative nonlinear algorithm applied to the indirect estimation of particle size distributions. *Journal of Computational Physics*, 18(2), 188-200. doi:10.1016/0021-9991(75)90028-5
3. **Markowski, G. R. (1987).** Improving Twomey's algorithm for inversion of aerosol measurement data. *Aerosol Science and Technology*, 7(2), 127-141. doi:10.1080/02786828708959153
4. **Hinds, W. C. (1999).** *Aerosol Technology: Properties, Behavior, and Measurement of Airborne Particles* (2nd ed.). Wiley. Eq. (3.22)
