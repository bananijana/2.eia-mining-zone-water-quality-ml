"""
Environmental Impact Assessment — Mining Zone Water Quality
------------------------------------------------------------
Computes three geochemical contamination indices:
  1. Heavy Metal Pollution Index (HPI)
  2. Contamination Index (Cd)
  3. Environmental Quality Index (EQI) — composite score

Then classifies each sample into an impact severity class:
  Low / Moderate / High / Very High

Reference standards: WHO (2022), BIS 10500:2012, USEPA MCL

Authors:
  Lead   : Banani Jana
  Support: Anikate Chowdhury
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
import os

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH  = "dataset.csv"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Permissible limits (mg/L) — WHO/BIS/USEPA
LIMITS = {
    "Fe":  0.30,   # WHO 2022
    "Mn":  0.10,   # WHO 2022
    "As":  0.01,   # WHO 2022
    "Pb":  0.01,   # WHO 2022
    "Cu":  2.00,   # WHO 2022
    "Zn":  3.00,   # WHO 2022
    "SO4": 250.0,  # BIS 10500
    "NO3": 45.0,   # BIS 10500
    "Cl":  250.0,  # BIS 10500
    "TDS": 500.0,  # BIS 10500
}

# Ideal/background values (mg/L) for HPI
IDEAL = {
    "Fe":  0.05,
    "Mn":  0.01,
    "As":  0.001,
    "Pb":  0.001,
    "Cu":  0.005,
    "Zn":  0.01,
}

HEAVY_METALS = list(IDEAL.keys())

# Unit weights for HPI (inverse of permissible limit)
unit_weights = {m: 1 / LIMITS[m] for m in HEAVY_METALS}
W_total = sum(unit_weights.values())
WEIGHTS = {m: unit_weights[m] / W_total for m in HEAVY_METALS}

PALETTE = {
    "Impact":     "#D32F2F",
    "Buffer":     "#F57C00",
    "Background": "#388E3C",
}
IMPACT_PALETTE = {
    "Low":       "#388E3C",
    "Moderate":  "#F9A825",
    "High":      "#F57C00",
    "Very High": "#D32F2F",
}

# ── Load ──────────────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
print(f"Loaded: {df.shape[0]} samples\n")

# ── 1. Heavy Metal Pollution Index (HPI) ─────────────────────────────────────
# HPI = Σ [Wi × (Ci/Si)] × 100
# Wi = unit weight, Ci = observed conc, Si = permissible limit
hpi_scores = []
for _, row in df.iterrows():
    hpi = sum(WEIGHTS[m] * (row[m] / LIMITS[m]) for m in HEAVY_METALS) * 100
    hpi_scores.append(round(hpi, 3))
df["HPI"] = hpi_scores

# ── 2. Contamination Index (Cd) ───────────────────────────────────────────────
# Cd = Σ (Ci/Bi - 1)  where Bi = ideal/background value
cd_scores = []
for _, row in df.iterrows():
    cd = sum((row[m] / IDEAL[m]) - 1 for m in HEAVY_METALS)
    cd_scores.append(round(cd, 3))
df["Cd"] = cd_scores

# ── 3. Physicochemical Deviation Score (PDS) ─────────────────────────────────
# Normalised deviation of key physicochemical params from permissible limits
PHYSICO = ["SO4", "NO3", "Cl", "TDS"]
pds_scores = []
for _, row in df.iterrows():
    pds = np.mean([row[p] / LIMITS[p] for p in PHYSICO])
    pds_scores.append(round(pds, 3))
df["PDS"] = pds_scores

# ── 4. Composite EQI ─────────────────────────────────────────────────────────
# EQI = 0.40×HPI_norm + 0.40×Cd_norm + 0.20×PDS_norm  (scaled 0–100)
def minmax(series):
    return (series - series.min()) / (series.max() - series.min()) * 100

df["HPI_norm"] = minmax(df["HPI"])
df["Cd_norm"]  = minmax(df["Cd"])
df["PDS_norm"] = minmax(df["PDS"])

df["EQI"] = (0.40 * df["HPI_norm"] +
             0.40 * df["Cd_norm"]  +
             0.20 * df["PDS_norm"]).round(3)

# ── 5. Impact Classification ──────────────────────────────────────────────────
def classify_eqi(score):
    if score < 25:   return "Low"
    elif score < 50: return "Moderate"
    elif score < 75: return "High"
    else:            return "Very High"

df["ImpactClass"] = df["EQI"].apply(classify_eqi)

# ── Print Summary ─────────────────────────────────────────────────────────────
print("── EQI Summary by Zone ──")
summary = df.groupby("Zone")[["HPI","Cd","EQI"]].agg(["mean","min","max"]).round(3)
print(summary)
print()
print("── Impact Class Distribution ──")
print(df.groupby(["Zone","ImpactClass"]).size().unstack(fill_value=0))
print()

# Save full results
df.to_csv(f"{OUTPUT_DIR}/eqi_results.csv", index=False)
print("Saved: eqi_results.csv")

# ── Plot 1: EQI Score by Zone (Boxplot) ───────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
order = ["Impact", "Buffer", "Background"]
sns.boxplot(data=df, x="Zone", y="EQI", order=order,
            palette=PALETTE, ax=ax, linewidth=1.5,
            flierprops=dict(marker="o", markersize=5,
                            markerfacecolor="grey", alpha=0.6))
ax.axhline(25, color="gold",   linestyle="--", linewidth=1.2, alpha=0.8,
           label="Low/Moderate (25)")
ax.axhline(50, color="orange", linestyle="--", linewidth=1.2, alpha=0.8,
           label="Moderate/High (50)")
ax.axhline(75, color="red",    linestyle="--", linewidth=1.2, alpha=0.8,
           label="High/Very High (75)")
ax.set_title("Environmental Quality Index (EQI) by Zone",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Sampling Zone", fontsize=11)
ax.set_ylabel("EQI Score", fontsize=11)
ax.legend(fontsize=8, loc="upper right")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/eqi_by_zone.png", dpi=180)
plt.close()
print("Saved: eqi_by_zone.png")

# ── Plot 2: HPI Bar Chart per Sample ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 5))
colors = [PALETTE[z] for z in df["Zone"]]
bars = ax.bar(df["SampleID"], df["HPI"], color=colors,
              edgecolor="white", width=0.7)
ax.axhline(100, color="red", linestyle="--", linewidth=1.5,
           label="HPI = 100 (critical threshold)")
ax.set_xlabel("Sample ID", fontsize=10)
ax.set_ylabel("HPI Score", fontsize=11)
ax.set_title("Heavy Metal Pollution Index (HPI) — All Samples",
             fontsize=13, fontweight="bold")
ax.tick_params(axis="x", rotation=90, labelsize=6)
patches = [mpatches.Patch(color=v, label=k) for k, v in PALETTE.items()]
patches.append(mpatches.Patch(color="red", label="HPI=100 threshold"))
ax.legend(handles=patches, fontsize=9)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/hpi_all_samples.png", dpi=180)
plt.close()
print("Saved: hpi_all_samples.png")

# ── Plot 3: Impact Class Distribution (Stacked bar) ───────────────────────────
impact_counts = df.groupby(["Zone","ImpactClass"]).size().unstack(fill_value=0)
impact_order  = ["Low","Moderate","High","Very High"]
impact_counts = impact_counts.reindex(columns=impact_order, fill_value=0)
zone_order    = ["Impact","Buffer","Background"]
impact_counts = impact_counts.reindex(zone_order)

fig, ax = plt.subplots(figsize=(8, 5))
bottom = np.zeros(3)
for cls in impact_order:
    vals = impact_counts[cls].values
    bars = ax.bar(zone_order, vals, bottom=bottom,
                  color=IMPACT_PALETTE[cls], label=cls,
                  edgecolor="white", width=0.5)
    for bar, val in zip(bars, vals):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_y() + bar.get_height()/2,
                    str(val), ha="center", va="center",
                    fontsize=10, fontweight="bold", color="white")
    bottom += vals

ax.set_ylabel("Number of Samples", fontsize=11)
ax.set_title("Impact Class Distribution by Zone", fontsize=13,
             fontweight="bold")
ax.legend(title="Impact Class", fontsize=9, title_fontsize=9,
          loc="upper right")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/impact_class_distribution.png", dpi=180)
plt.close()
print("Saved: impact_class_distribution.png")

# ── Plot 4: Index Comparison (HPI vs Cd vs EQI) ───────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
indices = ["HPI", "Cd", "EQI"]
titles  = ["Heavy Metal Pollution Index\n(HPI)",
           "Contamination Index\n(Cd)",
           "Environmental Quality Index\n(EQI — Composite)"]

for ax, idx, title in zip(axes, indices, titles):
    sns.boxplot(data=df, x="Zone", y=idx, order=order,
                palette=PALETTE, ax=ax, linewidth=1.3,
                flierprops=dict(marker="o", markersize=4,
                                markerfacecolor="grey", alpha=0.5))
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel(idx, fontsize=10)
    ax.tick_params(axis="x", labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

fig.suptitle("Geochemical Index Comparison Across Sampling Zones",
             fontsize=13, fontweight="bold")
plt.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/index_comparison.png", dpi=180)
plt.close()
print("Saved: index_comparison.png")

# ── Plot 5: Heavy Metal Exceedance Heatmap ────────────────────────────────────
# Ratio of observed to permissible limit per metal per zone
exceed_data = {}
for m in HEAVY_METALS:
    exceed_data[m] = df.groupby("Zone")[m].mean() / LIMITS[m]
exceed_df = pd.DataFrame(exceed_data).reindex(zone_order)

fig, ax = plt.subplots(figsize=(9, 4))
sns.heatmap(exceed_df, annot=True, fmt=".2f", cmap="RdYlGn_r",
            linewidths=0.5, ax=ax, center=1.0,
            cbar_kws={"label": "Ci / Permissible Limit"})
ax.set_title("Heavy Metal Exceedance Ratio (Mean Ci / WHO Permissible Limit)",
             fontsize=12, fontweight="bold")
ax.set_xlabel("Parameter", fontsize=10)
ax.set_ylabel("Zone", fontsize=10)
plt.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/exceedance_heatmap.png", dpi=180)
plt.close()
print("Saved: exceedance_heatmap.png")

# ── Plot 6: EQI vs pH scatter ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
for zone, grp in df.groupby("Zone"):
    ax.scatter(grp["pH"], grp["EQI"], color=PALETTE[zone],
               label=zone, s=55, alpha=0.8,
               edgecolors="white", linewidths=0.4)
ax.set_xlabel("pH", fontsize=11)
ax.set_ylabel("EQI Score", fontsize=11)
ax.set_title("pH vs EQI — Environmental Degradation Gradient",
             fontsize=12, fontweight="bold")
ax.legend(fontsize=9)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/pH_vs_EQI.png", dpi=180)
plt.close()
print("Saved: pH_vs_EQI.png")

print(f"\n✓ All EQI outputs saved to: {OUTPUT_DIR}")
