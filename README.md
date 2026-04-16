# OD Matrix Estimation — Citi Bike Jersey City

A Spatial Graph Convolutional Network (**SGCN**) for predicting **Origin-Destination (OD) matrices** of the Citi Bike sharing system in Jersey City, NJ. The model forecasts bike trip flows between stations across four time periods and benchmarks against a gravity model baseline.

---

## Overview

Bike-sharing OD matrices encode how many trips travel from each origin station to each destination station in a given time window. This project:

1. Processes raw Citi Bike trip data (2021–2024, Jersey City)
2. Constructs station-level spatial graphs
3. Trains an SGCN to predict OD flows per time period
4. Visualizes results as heatmaps and interactive Folium flow maps

**Time periods analyzed:** Morning peak (7–10 am) · Afternoon (10 am–4 pm) · Evening peak (4–7 pm) · All-day

---

## Results (V2 — 50 stations, tested on 2024)

| Period | SGCN MAE | Gravity Model MAE |
|--------|----------|-------------------|
| All Day | **166.1** | 408.8 |
| Morning Peak | — | — |

The SGCN consistently outperforms the gravity model baseline across all periods.

---

## Model Architecture (V2)

```
Station embeddings (256-dim)
    ↓
SGCN × 3 layers  (residual connections + batch normalization)
    ↓
Dropout (0.3)
    ↓
OD matrix output  (N × N predicted flows)
```

**Key hyperparameters:**

| Parameter | Value |
|-----------|-------|
| Embedding dim | 256 |
| GCN layers | 3 |
| Dropout | 0.3 |
| Learning rate | 0.003 |
| Batch size | 128 |
| Epochs | 15 (early stopping) |
| L2 regularization | 1e-5 |

---

## Data

- **Source:** [Citi Bike System Data](https://citibikenyc.com/system-data) — Jersey City monthly CSV files
- **Period:** January 2021 – December 2024
- **Split:** Train 2021–2022 · Validation last 2 months of 2023 · Test 2024
- **Stations:** 50 (V2), 20 (V1)

Place monthly CSV files in `data/citibike/`.

---

## Usage

**Run V2 (latest):**
```bash
python sgcn_od_predictor_V2.py
```

**Run V1 (baseline):**
```bash
python sgcn_od_predictor_V1.py
```

---

## Output Structure

```
results_sgcn_improved/          # V2 outputs (latest)
    *.pth                       # trained model weights
    heatmap_actual_*.png/svg    # ground-truth OD heatmaps
    heatmap_predicted_*.png/svg # predicted OD heatmaps
    training_curves_*.png       # loss curves
    station_map_50.html         # interactive station map
    metrics_summary.csv         # MAE / RMSE per period

results_sgcn/                   # V1 outputs
results/                        # zone classification & flow analysis
```

---

## Zone Classification

Stations are automatically classified as **residential** or **commercial** based on station name heuristics. V1 analysis on 2024 data found:

- 57% residential → residential flows
- 19% residential → commercial flows

---

## Installation

```bash
pip install torch pandas numpy scikit-learn matplotlib seaborn folium
```

---

## Version History

| Version | Stations | Training data | Key improvements |
|---------|----------|---------------|-----------------|
| V2 (latest) | 50 | 2021–2023 | Residual connections, batch norm, validation split |
| V1 | 20 | 2023 | Initial SGCN implementation |
| V1–V8 notebooks | — | Exploratory | Data exploration, prototyping |

---

## Requirements

- Python 3.8+
- PyTorch ≥ 1.9
- Pandas, NumPy, scikit-learn, Matplotlib, Seaborn, Folium
