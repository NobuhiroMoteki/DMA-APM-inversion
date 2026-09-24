# ==============================================================================
# params.py  ---  User configuration file
# Edit only this file to change the analysis settings.
# ==============================================================================

# --- Input file ---
FILE_PATH  = "./APM-CPC_data/DMA-APMscan-CPC_20260303BC/JetA1_Dmob450nm_try1__20050101030343.csv"
OUTPUT_DIR = "./results"        # Output directory for JPEG figures

# --- APM geometry [m]  (nominal dimensions of Kanomax APM Model-3601) ---
# L=100 mm, r1=24 mm, r2=25 mm are the nominal dimensions of the Kanomax APM Model-3601.
# Change these values if you use a different instrument model.
L   = 100.0e-3    # Electrode length
r1  = 24.0e-3     # Inner cylinder radius
r2  = 25.0e-3     # Outer cylinder radius

# --- APM operating conditions ---
Q_a_lpm = 0.3     # Aerosol flow rate [L/min]

# --- CPC conditions ---
Q_cpc_lpm = 0.3   # CPC flow rate [L/min]

# --- Data preprocessing ---
num_bins = 30     # Number of voltage bins

# --- RK4 numerical integration parameters ---
dz  = 1.0e-4      # Integration step size [m]
nr0 = 1000        # Number of initial particle positions

# --- Mass range for inversion ---
m_min_fg = 5.0    # Lower bound [fg]
m_max_fg = 80.0   # Upper bound [fg]
J        = 40     # Number of mass bins

# --- Convergence criteria ---
max_iter      = 2000
chi_threshold = 1.0

# --- 2D model only ---
beta        = 0.1   # DMA resolution β = q_sample / q_sheath
num_Zp_bins = 7     # Number of Z_p integration points

# --- Standalone APM transfer function (run_transfer_function.py only) ---
TF_RPM          = 2661.0                      # APM rotation speed [rpm]
TF_Dmob_nm      = 350.0                       # Electrical mobility diameter [nm]
TF_V_list       = [50.0, 100.0, 200.0, 400.0] # Applied voltages [V]
TF_rel_halfwidth = 1.0                        # Mass range: m_c x (1 +/- this), lower bound >= 0.01 m_c
TF_num_m        = 200                         # Number of mass points per voltage

# --- APM set-point for spherical particles (run_apm_setpoint.py only) ---
# Resolution parameter: lambda = 2 tau omega^2 L / v_bar  (Ehara et al., 1996)
SP_particles     = [(303.0, 1050.0), (345.0, 1050.0), (401.0, 1050.0), (453.0, 1050.0)]  # (d [nm], rho_eff [kg/m3])
SP_Q_a_lpm       = 0.46          # APM aerosol flow rate [L/min]
SP_lambda        = [0.2, 0.5]    # Resolution parameter lambda (dimensionless); float or list
                                 # (list -> one figure per particle, lambdas overlaid)
SP_OUTPUT_DIR    = "./results/exp_plan"   # Output directory for set-point results
SP_rel_halfwidth = 1.0    # Mass range: m_p x (1 +/- this), lower bound >= 0.01 m_p
SP_num_m         = 401    # Number of mass points
# Multiply charged sphere cluster transmitted at the same set-point (M/q = m_p).
# Set SP_cluster = None to skip.
SP_cluster = {
    "n_mono": 2,     # Number of primary spheres
    "charge": 2,     # Number of elementary charges
    "chi":    1.12,  # Dynamic shape factor, random orientation (Hinds 1999, Table 3.2)
}
