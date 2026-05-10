# DMA Sample-Flow Sensitivity: Mass-Distribution Width Correction

This note records the derivation and numerical correction used in
`dma_qa_sensitivity.py` to estimate how the DMA-outlet mass distribution
$f_{\text{out}}(m)$ changes when the DMA sample flow is varied while the
sheath flow is held fixed.

## 1. Setup

| Quantity | Reference | Perturbed |
| --- | --- | --- |
| $q_{\text{sample}}$ [L/min] | 0.30 | 0.43 |
| $q_{\text{sheath}}$ [L/min] | 3.0 | 3.0 |
| $\beta = q_{\text{sample}}/q_{\text{sheath}}$ | 0.100 | 0.143 |

Inlet assumption (per problem statement):
$n_{\text{in}}(D_{\text{mob}}) \approx \mathrm{const}$ within the DMA
classification window. The conditional mass distribution
$h(m \mid D_{\text{mob}})$ is allowed to depend on $D_{\text{mob}}$.

The reference inversion result $f_{\text{out}}(m;\beta_{\text{ref}}=0.100)$
for each of six datasets (FS / JetA1 $\times$ $D_{\text{mob}}^*$ = 450, 500,
550 nm) is taken as the starting point.

## 2. Derivation

### 2.1 Triangular DMA transfer function

For a balanced-flow DMA the (non-diffusive) transfer function is a
triangle peaking at the centre mobility $Z_p^*$ with base half-width
$\beta Z_p^*$:

$$
\Omega(Z_p;\, Z_p^*,\beta) \;=\; \max\!\Big(0,\; 1 - \frac{|Z_p - Z_p^*|}{\beta\, Z_p^*}\Big)
$$

The variance of a symmetric triangular distribution with half-base $a$ is
$a^2/6$, so

$$
\sigma_{Z_p}^2 \;=\; \frac{(\beta Z_p^*)^2}{6}, \qquad
\sigma_{Z_p} \;=\; \frac{\beta Z_p^*}{\sqrt{6}}.
$$

### 2.2 Mapping to mobility-diameter space

For a narrow window we linearise around $D_{\text{mob}}^*$. With
$Z_p \propto 1/D_{\text{mob}}$ to leading order,
$\delta Z_p / Z_p^* = -\,\delta D_{\text{mob}}/D_{\text{mob}}^*$, hence

$$
\frac{\sigma_{D_{\text{mob}}}}{D_{\text{mob}}^*} \;\approx\; \frac{\sigma_{Z_p}}{Z_p^*} \;=\; \frac{\beta}{\sqrt{6}}.
$$

(Cunningham slip merely changes the prefactor by a small amount; for our
narrow windows this correction is sub-percent and is neglected here.)

### 2.3 Mapping to mass space (mass--mobility scaling)

For fractal aerosols we adopt a single power law
$\langle m \rangle \propto D_{\text{mob}}^{D_f}$, so for narrow windows

$$
\frac{\delta m}{\langle m \rangle} \;=\; D_f \cdot \frac{\delta D_{\text{mob}}}{D_{\text{mob}}^*}.
$$

The DMA-window contribution to the mass-distribution standard deviation
at $D_{\text{mob}}^*$ is therefore

$$
\boxed{\;\sigma_{\text{DMA}}(\beta) \;=\; D_f \,\mu\, \frac{\beta}{\sqrt{6}}\;}
$$

where $\mu = \langle m \rangle$ at $D_{\text{mob}}^*$ (taken from the
Gaussian-fit centre of the inversion result).

### 2.4 Quadrature decomposition of the observed width

The DMA outlet mass distribution is the convolution (over
$D_{\text{mob}}$) of the conditional intrinsic distribution at
$D_{\text{mob}}^*$ with the DMA window. To second-moment order, variances
add in quadrature:

$$
\sigma_{\text{out}}^2(\beta) \;=\; \sigma_{\text{intrinsic}}^2 + \sigma_{\text{DMA}}^2(\beta).
$$

The intrinsic component (independent of $\beta$) is recovered from the
reference fit:

$$
\sigma_{\text{intrinsic}}^2 \;=\; \sigma_{\text{out}}^2(\beta_{\text{ref}}) - \sigma_{\text{DMA}}^2(\beta_{\text{ref}}).
$$

### 2.5 Predicted width and amplitude

The predicted width at $\beta_{\text{new}}$ is

$$
\boxed{\;
\sigma_{\text{out}}^2(\beta_{\text{new}}) \;=\; \sigma_{\text{out}}^2(\beta_{\text{ref}}) \;+\; \big[\sigma_{\text{DMA}}^2(\beta_{\text{new}}) - \sigma_{\text{DMA}}^2(\beta_{\text{ref}})\big].
\;}
$$

This is equivalent, on the curve $f_{\text{out}}(m)$ itself, to a
Gaussian convolution with the increment

$$
\Delta\sigma \;=\; \sqrt{\sigma_{\text{DMA}}^2(\beta_{\text{new}}) - \sigma_{\text{DMA}}^2(\beta_{\text{ref}})},
$$

followed by a uniform amplitude rescaling

$$
f_{\text{out}}(m;\beta_{\text{new}}) \;=\; \frac{\beta_{\text{new}}}{\beta_{\text{ref}}} \cdot \big(\,f_{\text{out}}(\,\cdot\,;\beta_{\text{ref}}) * G(\Delta\sigma)\,\big)(m).
$$

The integrated number $N = \int f_{\text{out}}(m)\,\mathrm{d}m$ scales by
$\beta_{\text{new}}/\beta_{\text{ref}}$ exactly (independent of shape).

## 3. Estimating $D_f$ from the reference data

A log--log regression of the Gaussian-fit centres $\mu$ versus the three
nominal $D_{\text{mob}}^*$ values across all six datasets yields

$$
D_f \;\approx\; 2.63 \qquad (\text{from data}).
$$

This value is used uniformly across the six panels in the script.

## 4. Correction results

All values in fg. Reference $\sigma_{\text{out}}(0.100)$ is taken from the
Gaussian fit stored in `results/data/*.npz`.

| Sample | $D_{\text{mob}}^*$ [nm] | $\mu$ | $\sigma_{\text{out}}(0.100)$ | $\sigma_{\text{DMA}}(0.100)$ | $\sigma_{\text{DMA}}(0.143)$ | $\sigma_{\text{intrinsic}}$ | $\Delta\sigma$ | $\sigma_{\text{out}}(0.143)$ | ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FS    | 450 | 26.1 | 4.6 | 2.80 | 4.01 | 3.65 | 2.88 | 5.42 | **1.18** |
| FS    | 500 | 34.7 | 6.1 | 3.73 | 5.33 | 4.83 | 3.83 | 7.19 | **1.18** |
| FS    | 550 | 44.2 | 6.7 | 4.75 | 6.79 | 4.73 | 4.86 | 8.27 | **1.23** |
| JetA1 | 450 | 26.4 | 4.4 | 2.83 | 4.05 | 3.37 | 2.91 | 5.27 | **1.20** |
| JetA1 | 500 | 34.9 | 5.8 | 3.75 | 5.36 | 4.42 | 3.84 | 6.95 | **1.20** |
| JetA1 | 550 | 44.8 | 6.3 | 4.81 | 6.88 | 4.07 | 4.93 | 7.99 | **1.27** |

The "ratio" column is
$\sigma_{\text{out}}(\beta=0.143) / \sigma_{\text{out}}(\beta=0.100)$ —
the *observed* width broadening. It is uniformly smaller than the
flow-rate ratio $\beta_{\text{new}}/\beta_{\text{ref}} = 1.433$ because
of the quadrature with $\sigma_{\text{intrinsic}}$.

Limiting cases:

- $\sigma_{\text{intrinsic}} \gg \sigma_{\text{DMA}}$: the observed width
  is essentially unchanged.
- $\sigma_{\text{intrinsic}} \ll \sigma_{\text{DMA}}$: the observed width
  scales as $\beta$, i.e. by 1.433.

For these datasets the two contributions are comparable, giving an
intermediate broadening of roughly **1.18--1.27**.

## 5. Assumptions and limitations

- Single power-law mass--mobility relation $m \propto D_{\text{mob}}^{D_f}$
  with a global $D_f$ fitted across both samples and three diameters.
- Triangle-window contribution approximated by a Gaussian of equal
  variance. The tail shapes differ, but for $\Delta\beta/\beta \sim 0.43$
  the second-moment treatment dominates.
- $\sigma_{\text{intrinsic}}$ at each $D_{\text{mob}}^*$ is treated as
  unchanged when $\beta$ is varied (i.e. the conditional shape
  $h(m \mid D_{\text{mob}}^*)$ is preserved).
- Diffusion broadening of the DMA transfer function is not included.
- Cunningham slip in the $Z_p \leftrightarrow D_{\text{mob}}$
  transformation is neglected (sub-percent effect for these narrow
  windows).

## 6. Reproducibility

- Script: [`dma_qa_sensitivity.py`](../dma_qa_sensitivity.py)
- Output figure: [`results/paper_figures/dma_qa_sensitivity_3x2.pdf`](../results/paper_figures/dma_qa_sensitivity_3x2.pdf)
- Source data: NPZ dumps under `results/data/` produced by
  [`run_batch_inversion.py`](../run_batch_inversion.py).
