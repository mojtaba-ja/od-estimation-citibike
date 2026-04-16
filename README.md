# OD Matrix Estimation — Citi Bike Jersey City

A **Spatial Graph Convolutional Network (SGCN)** for predicting Origin-Destination (OD) trip matrices of the Citi Bike sharing system in Jersey City, NJ. The model learns spatial correlations between stations from 4 years of trip data and forecasts inter-station flows across four daily time periods, significantly outperforming a gravity model baseline.

---

## Results (V2 — 50 stations, trained 2021–2023, tested on 2024)

| Time Period | SGCN MAE | Gravity Model MAE | Improvement |
|-------------|----------|-------------------|-------------|
| All Day | **166.1** | 408.8 | 59% |
| Morning Peak (7–10 am) | — | — | — |
| Afternoon (10 am–4 pm) | — | — | — |
| Evening Peak (4–7 pm) | — | — | — |

> Full per-period results are saved to `results_sgcn_improved/metrics_summary.csv` after training.

---

## What It Does

Bike-sharing OD matrices encode the number of trips between every pair of stations in a given time window. This project:

1. Aggregates raw Citi Bike CSV trip records into station-level OD matrices per time period
2. Constructs a spatial graph of stations (nodes = stations, edges = geographic proximity)
3. Trains an SGCN to predict OD flows from learned station embeddings
4. Benchmarks against a gravity model and visualizes results as heatmaps and interactive Folium flow maps

---

## Model Architecture (V2)

```
Station embeddings (256-dim)
    ↓
SGCN × 3 layers  (residual connections + batch normalization)
    ↓
Dropout (0.3)
    ↓
OD matrix output  (N × N predicted trip flows)
```

| Hyperparameter | Value |
|----------------|-------|
| Embedding dim | 256 |
| GCN layers | 3 |
| Dropout | 0.3 |
| Learning rate | 0.003 |
| Batch size | 128 |
| Max epochs | 15 (early stopping) |
| L2 regularization | 1e-5 |

---

## Data

**Source:** [Citi Bike System Data](https://citibikenyc.com/system-data) — Jersey City monthly CSVs  
**Coverage:** January 2021 – December 2024  
**Split:** Train Jan 2021–Oct 2023 · Validation Nov–Dec 2023 · Test Jan–Dec 2024  
**Stations:** 50 (V2), 20 (V1)

Download the monthly trip CSV files and place them in `data/citibike/`.

---

## Usage

```bash
# Latest model (V2 — 50 stations, residual SGCN)
python sgcn_od_predictor_V2.py

# Baseline model (V1 — 20 stations)
python sgcn_od_predictor_V1.py
```

Exploratory analysis and prototyping notebooks: `OD Estimation NYC Bike Trip Data-V4` through `V8`.

---

## Output Structure

```
results_sgcn_improved/          # V2 outputs
    heatmap_actual_*.png/svg    # ground-truth OD heatmaps
    heatmap_predicted_*.png/svg # model-predicted OD heatmaps
    training_curves_*.png       # loss curves per time period
    station_map_50.html         # interactive Folium station map
    metrics_summary.csv         # MAE / RMSE for all time periods

results_sgcn/                   # V1 outputs
results/                        # zone classification & flow statistics
```

---

## Zone Classification

Stations are automatically classified as **residential** or **commercial** based on station name heuristics. Analysis on 2024 data (V1, 20 stations):

- **57%** of flows: residential → residential
- **19%** of flows: residential → commercial

---

## Version History

| Version | Stations | Notes |
|---------|----------|-------|
| `sgcn_od_predictor_V2.py` | 50 | **Latest** — residual connections, batch norm, 3-year training data |
| `sgcn_od_predictor_V1.py` | 20 | Baseline SGCN implementation |
| Notebooks V4–V8 | — | Data exploration and model prototyping |

---

## Installation

```bash
pip install torch pandas numpy scikit-learn matplotlib seaborn folium
```

Python 3.8+ · PyTorch ≥ 1.9

---

## Citation

If you use this code or find it helpful in your research, please cite:

```bibtex
@article{jafarian5801889spatial,
  title   = {Spatial Graph Convolutional Network for Predicting
             Bike-Sharing Origin-Destination Spatiotemporal Flows},
  author  = {Jafarian Abyaneh, Mojtaba and Jang, Jinwoo and Kaisar, Evangelos I},
  journal = {Available at SSRN 5801889}
}
```
