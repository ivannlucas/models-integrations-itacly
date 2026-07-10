"""
DATAGIA project constants.

Groups physical constants from the simulator, genetic algorithm parameters,
and the feature/target lists used throughout the pipeline.
"""

# =============================================================================
# FEATURES AND TARGETS
# =============================================================================
FEATURES = ["T_in_leche", "F_flow", "T_servicio", "t_ciclo", "Delta_P"]
TARGETS = ["E_consumo", "T_out_leche"]

# =============================================================================
# PHYSICAL CONSTANTS — SIMULATOR (data_creation.ipynb)
# =============================================================================
U_CLEAN = 3500       # W/m²K — Clean heat transfer coefficient
AREA = 15            # m² — Effective plate heat exchanger area
CP_MILK = 3890       # J/(kg·K) — Specific heat capacity of milk
RHO_MILK = 1030      # kg/m³ — Milk density

ETA_BOILER = 0.90    # Boiler efficiency
ETA_PUMP = 0.75      # Pump efficiency

NUM_DAYS = 180       # Simulation days
SAMPLING_FREQ = 5    # Minutes between samples

# =============================================================================
# GENETIC ALGORITHM PARAMETERS (Single-objective GA v4)
# =============================================================================
BOUNDS = {
    "F_flow":     (3500.0, 5500.0),   # L/h — Pump operational limits
    "T_servicio": (76.0,   95.0),     # °C — Boiler limits
}

# Thermodinamic Safety Margin + Instrumentation Tolerance:
# Legal pasteurization limit is 72.0°C, but the algorithm setpoint must
# account for measurement uncertainty of PT100 Class A sensors (±0.3°C
# at 72°C per IEC 60751) and FDA PMO thermometer tolerance (±0.5°F ≈ ±0.28°C).
# A +0.3°C safety margin (+0.3 over 72.0) prevents the FDV (Flow Diverting Valve)
# from triggering unnecessarily due to sensor noise and dynamic fluctuations.
T_OUT_MIN = 72.3      # °C — Minimum pasteurization temperature with safety margin
PENALTY_FACTOR = 10.0  # Penalty factor for infeasible solutions

# Default GA configuration
GA_DEFAULT_CONFIG = {
    "pop_size": 150,
    "n_gen": 15,
    "cxpb": 0.8,
    "mutpb": 0.2,
}

# =============================================================================
# FOOD SAFETY
# =============================================================================
# Legal pasteurization limit is 72.0°C, but with +0.3°C safety margin
# to absorb PT100 Class A sensor uncertainty (±0.3°C per IEC 60751)
# and FDA PMO tolerance (±0.5°F ≈ ±0.28°C).
T_SAFETY = 72.3       # °C — Legal pasteurization constraint with safety margin
