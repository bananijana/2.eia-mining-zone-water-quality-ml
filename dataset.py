"""
Dataset Generator — Mining Zone Water Quality
-----------------------------------------------
Generates a synthetic hydrogeochemical dataset simulating
water quality conditions across three zones relative to a
mineral exploration / mining source area.

Zones:
  Impact Zone   (0–500 m)   — 30 samples
  Buffer Zone   (500–2000 m) — 30 samples
  Background Zone (>2000 m) — 30 samples

Parameters reflect conditions typical of uranium-bearing
metasedimentary terrains with sulphide-associated mineralisation.
Values parameterised from published literature for similar
geological settings (AMD-type environments).
"""

import numpy as np
import pandas as pd

np.random.seed(7)

def generate_zone(n, zone_label, ph_range, ec_range, tds_range,
                  fe_range, mn_range, as_range, pb_range,
                  cu_range, zn_range, so4_range, no3_range,
                  cl_range, hco3_range, temp_range):
    data = {
        "pH":    np.round(np.random.uniform(*ph_range, n), 2),
        "EC":    np.round(np.random.uniform(*ec_range, n), 1),
        "TDS":   np.round(np.random.uniform(*tds_range, n), 1),
        "Temp":  np.round(np.random.uniform(*temp_range, n), 1),
        "SO4":   np.round(np.random.uniform(*so4_range, n), 2),
        "NO3":   np.round(np.random.uniform(*no3_range, n), 2),
        "Cl":    np.round(np.random.uniform(*cl_range, n), 2),
        "HCO3":  np.round(np.random.uniform(*hco3_range, n), 2),
        "Fe":    np.round(np.random.uniform(*fe_range, n), 3),
        "Mn":    np.round(np.random.uniform(*mn_range, n), 3),
        "As":    np.round(np.random.uniform(*as_range, n), 4),
        "Pb":    np.round(np.random.uniform(*pb_range, n), 4),
        "Cu":    np.round(np.random.uniform(*cu_range, n), 3),
        "Zn":    np.round(np.random.uniform(*zn_range, n), 3),
        "Zone":  [zone_label] * n,
    }
    return pd.DataFrame(data)

# ── Zone parameters ───────────────────────────────────────────────────────────
# Impact zone — acidic, high metals, high SO4 (sulphide oxidation signature)
impact = generate_zone(
    n=30, zone_label="Impact",
    ph_range=(4.5, 6.2),
    ec_range=(1800, 4500),
    tds_range=(1200, 3000),
    temp_range=(24, 32),
    so4_range=(250, 800),
    no3_range=(15, 55),
    cl_range=(80, 250),
    hco3_range=(20, 120),
    fe_range=(1.5, 8.0),
    mn_range=(0.5, 3.5),
    as_range=(0.02, 0.12),
    pb_range=(0.01, 0.08),
    cu_range=(0.05, 0.8),
    zn_range=(0.1, 2.5),
)

# Buffer zone — transitional, moderate contamination
buffer = generate_zone(
    n=30, zone_label="Buffer",
    ph_range=(6.2, 7.4),
    ec_range=(800, 1800),
    tds_range=(520, 1200),
    temp_range=(22, 30),
    so4_range=(80, 250),
    no3_range=(5, 20),
    cl_range=(30, 100),
    hco3_range=(100, 280),
    fe_range=(0.3, 1.8),
    mn_range=(0.1, 0.6),
    as_range=(0.005, 0.025),
    pb_range=(0.003, 0.015),
    cu_range=(0.01, 0.12),
    zn_range=(0.02, 0.3),
)

# Background zone — near-natural baseline
background = generate_zone(
    n=30, zone_label="Background",
    ph_range=(7.2, 8.5),
    ec_range=(200, 800),
    tds_range=(130, 520),
    temp_range=(20, 27),
    so4_range=(10, 80),
    no3_range=(0.5, 8),
    cl_range=(10, 50),
    hco3_range=(200, 400),
    fe_range=(0.01, 0.35),
    mn_range=(0.001, 0.08),
    as_range=(0.0005, 0.006),
    pb_range=(0.0005, 0.004),
    cu_range=(0.001, 0.02),
    zn_range=(0.005, 0.05),
)

# ── Combine ───────────────────────────────────────────────────────────────────
df = pd.concat([impact, buffer, background], ignore_index=True)
df.insert(0, "SampleID", [f"MW{str(i+1).zfill(3)}" for i in range(len(df))])

df.to_csv("/home/claude/projects/eia-mining-zone-water-quality-ml/dataset.csv",
          index=False)
print(f"Dataset saved: {df.shape[0]} samples, {df.shape[1]} columns")
print(df["Zone"].value_counts())
print(df.describe().T[["mean","min","max"]].round(3))
