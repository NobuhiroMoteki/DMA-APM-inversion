# ==============================================================================
# params.py  ---  User configuration file
# Edit only this file to change the analysis settings.
# ==============================================================================

# --- Input file ---
FILE_PATH  = "./APM-CPC_data/DMA-APMscan-CPC_20260303BC/FS_Dmob450nm_try2__20050101014000.csv"
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
