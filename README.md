# Environmental Impact Assessment of Mining Zone Water Quality Using Geochemical Indices and Machine Learning

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

---

## Overview

This project presents a Python-based framework for the environmental impact assessment (EIA) of water quality in and around mineral exploration and mining-influenced zones. The framework integrates three standard geochemical contamination indices into a composite **Environmental Quality Index (EQI)**, which is then validated and extended using a **Random Forest machine learning classifier**.

The study is motivated by field observations from uranium-bearing metasedimentary terrains in the Singhbhum Shear Zone, Jharkhand, where sulphide-associated mineralisation and processing activities pose significant risks to local water quality.

> **Note:** As hydrogeochemical data from active mineral exploration zones are restricted under AMD/UCIL confidentiality protocols, a synthetic dataset was generated to simulate conditions typical of such settings, parameterised from published literature values for analogous AMD-type environments.

---

## Objectives

1. Compute Heavy Metal Pollution Index (HPI), Contamination Index (Cd), and a composite EQI for 90 sampling stations across three zones
2. Classify each station into an impact severity class (Low / Moderate / High / Very High)
3. Train a Random Forest model to predict impact class directly from raw water quality parameters
4. Validate ML predictions against index-based classifications
5. Identify the most diagnostic water quality parameters for impact discrimination

---

## Methodology

### Zone Framework

| Zone | Distance from Source | Samples |
|---|---|---|
| Impact Zone | 0 – 500 m | 30 |
| Buffer Zone | 500 m – 2 km | 30 |
| Background Zone | > 2 km | 30 |

### Index Computation

**1. Heavy Metal Pollution Index (HPI)**

$$HPI = \sum_{i=1}^{n} W_i \cdot \frac{C_i}{S_i} \times 100$$

Where $W_i$ = unit weight (inverse of permissible limit), $C_i$ = observed concentration, $S_i$ = WHO/BIS permissible limit.

**2. Contamination Index (Cd)**

$$C_d = \sum_{i=1}^{n} \left(\frac{C_i}{B_i} - 1\right)$$

Where $B_i$ = ideal/background concentration.

**3. Environmental Quality Index (EQI) — Composite**

$$EQI = 0.40 \times HPI_{norm} + 0.40 \times C_{d,norm} + 0.20 \times PDS_{norm}$$

Where PDS = Physicochemical Deviation Score from permissible limits (SO₄, NO₃, Cl, TDS).

### Impact Classification

| EQI Score | Impact Class |
|---|---|
| < 25 | Low |
| 25 – 50 | Moderate |
| 50 – 75 | High |
| > 75 | Very High |

### Reference Standards

| Parameter | Permissible Limit | Standard |
|---|---|---|
| Fe | 0.30 mg/L | WHO 2022 |
| Mn | 0.10 mg/L | WHO 2022 |
| As | 0.01 mg/L | WHO 2022 |
| Pb | 0.01 mg/L | WHO 2022 |
| SO₄ | 250 mg/L | BIS 10500:2012 |
| NO₃ | 45 mg/L | BIS 10500:2012 |
| TDS | 500 mg/L | BIS 10500:2012 |

---

## Parameters

| Category | Parameters |
|---|---|
| Physical | pH, EC (µS/cm), TDS (mg/L), Temperature (°C) |
| Major ions | SO₄, NO₃, Cl, HCO₃ (mg/L) |
| Heavy metals | Fe, Mn, As, Pb, Cu, Zn (mg/L) |
| Derived indices | HPI, Cd, PDS, EQI |

---

## Project Structure

```
eia-mining-zone-water-quality-ml/
│
├── dataset.py              # Synthetic dataset generator
├── dataset.csv             # Generated dataset (90 samples, 15 parameters)
├── eqi_analysis.py         # Index computation and visualisation (lead script)
├── ml_classify.py          # Random Forest ML pipeline
│
├── outputs/                # EQI analysis outputs
│   ├── eqi_results.csv
│   ├── eqi_by_zone.png
│   ├── hpi_all_samples.png
│   ├── impact_class_distribution.png
│   ├── index_comparison.png
│   ├── exceedance_heatmap.png
│   └── pH_vs_EQI.png
│
├── ml_outputs/             # ML outputs
│   ├── classification_report.csv
│   ├── confusion_matrix.png
│   ├── feature_importance.png
│   ├── roc_curves.png
│   ├── cv_scores.png
│   └── ml_vs_index_agreement.png
│
└── README.md
```

---

## Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
```

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

---

## Usage

```bash
# Step 1: Generate dataset
python dataset.py

# Step 2: Compute EQI indices and generate plots
python eqi_analysis.py

# Step 3: Run ML classification
python ml_classify.py
```

---

## Key Results

### EQI Summary

| Zone | Mean HPI | Mean EQI | Dominant Impact Class |
|---|---|---|---|
| Impact | ~659 | ~64 | High / Very High |
| Buffer | ~135 | ~13 | Low |
| Background | ~29 | ~2 | Low |

### ML Performance
- **5-Fold CV Accuracy: 90.0% ± 2.2%**
- High agreement between ML predictions and index-based classification across all three zones

### Key Finding
pH and Fe are the most diagnostic parameters for impact class discrimination, consistent with sulphide oxidation-driven acidification and iron mobilisation in mining-influenced environments.

---

## Outputs Explained

**EQI by Zone (Boxplot)** — Shows clear separation of EQI scores across Impact, Buffer, and Background zones with threshold lines at 25, 50, and 75.

**HPI Bar Chart** — Per-sample HPI scores with the critical threshold at HPI = 100. Impact zone samples consistently exceed the threshold.

**Exceedance Heatmap** — Mean concentration / WHO permissible limit ratio per metal per zone. Red cells indicate parameters exceeding safe limits.

**Feature Importance** — Identifies which raw water quality parameters most strongly predict impact class. Guides cost-effective monitoring by prioritising key indicators.

**ML vs Index Agreement** — Validates that the Random Forest model independently recovers the same classification as the geochemical index framework.

---

## Geoscience Context

Environmental quality assessment of mining-influenced water bodies is a critical component of EIA baseline studies. This framework is applicable to:

- Pre-mining baseline water quality characterisation
- Post-mining impact monitoring
- AMD (Acid Mine Drainage) risk screening
- Regulatory compliance assessment against BIS/WHO standards

---

## Contributors

| Role | Contributor |
|---|---|
| **Lead Author** | Banani Jana — conceptualisation, EQI framework design, index computation, environmental standards integration, visualisation, documentation |
| **Contributor** | Anikate Chowdhury — ML classification pipeline, statistical validation |

---

## Affiliation

Department of Geology, Presidency University, Kolkata
Supervisor: Asst. Prof. Dr. Aditya Sarkar

---

## Contact

**Banani Jana**
M.Sc. Applied Geology (2nd Year), Presidency University, Kolkata
Email: bananijana2002@gmail.com

---

## License

MIT License

---

## Citation

If you use this framework, please cite:

> Jana, B., & Chowdhury, A. (2026). *Environmental Impact Assessment of Mining Zone Water Quality Using Geochemical Indices and Machine Learning*. GitHub Repository.
