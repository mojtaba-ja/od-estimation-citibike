"""
Improved Spatial Graph Convolutional Network (SGCN) for Bike Sharing OD Matrix Prediction
WITH COMPLETE VISUALIZATION SUITE + IMPROVEMENTS
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import warnings
import glob
import math
import gc
import time  # NEW - for timing

# Import matplotlib BEFORE using it
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import folium
from folium.plugins import MarkerCluster

warnings.filterwarnings("ignore")

###########################################
# CONFIGURATION - IMPROVED
###########################################
CONFIG = {
    "DATA_DIR": "data/citibike",
    # To use 2022+2023 data for training: ["JC2022*-citibike-tripdata.csv", "JC2023*-citibike-tripdata.csv"]
    "TRAIN_PATTERNS": ["JC-2021*-citibike-tripdata.csv",
                   "JC-2022*-citibike-tripdata.csv",
                   "JC-2023*-citibike-tripdata.csv"],
    "TEST_PATTERN": "JC2024*citibiketripdata.csv",
    "OUTPUT_DIR": "results_sgcn_improved",
    "NUM_STATIONS": 50,  # INCREASED from 20 to 50
    "TIME_PERIODS": {
        "morning_peak": (7, 10),
        "afternoon": (10, 16),
        "evening_peak": (16, 20),
        "all_day": (0, 24),
    },
    "EMBEDDING_DIM": 256,  # INCREASED from 128
    "NUM_LAYERS": 3,  # Back to 3 with residual connections
    "DROPOUT": 0.3,
    "LEARNING_RATE": 0.003,  # ADJUSTED
    "BATCH_SIZE": 128,  # INCREASED from 64
    "NUM_EPOCHS": 15,
    "L2_REG": 1e-5,
    "PATIENCE": 50,
    "CHUNK_SIZE": 10000,
    "SAMPLE_RATE": 1.0,
    "MIN_TRIPS_THRESHOLD": 3,  # REDUCED for more connections
    "VALIDATION_MONTHS": 2,  # Use last 2 months of 2023 for validation
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
}

ZONE_CLASSIFICATION = {
    "commercial_keywords": [
        "downtown", "exchange", "newport", "harborside", "grove", "path", "plaza",
        "office", "center", "station", "transit", "christopher columbus", "essex",
        "liberty", "morris canal", "hudson", "waterfront", "financial",
    ],
    "residential_keywords": [
        "heights", "bergen", "lafayette", "greenville", "mcginley", "park", "school",
        "housing", "residential", "neighborhood", "home", "communipaw", "westside",
        "journal square", "marion", "duncan", "terrace", "garden",
    ],
}

NEIGHBORHOOD_LABELS = {
    "The Heights": (40.7489, -74.0464),
    "Downtown JC": (40.7178, -74.0431),
    "Journal Square": (40.7324, -74.0635),
    "Bergen-Lafayette": (40.7089, -74.0665),
    "Greenville": (40.7067, -74.0774),
    "Hoboken": (40.7439, -74.0324),
}

###########################################
# HELPER FUNCTIONS
###########################################

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance in km"""
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    return 6371 * c


def classify_zone_type(station_name):
    """Classify as residential (0) or commercial (1)"""
    name_lower = station_name.lower()
    is_commercial = any(kw in name_lower for kw in ZONE_CLASSIFICATION["commercial_keywords"])
    return 1 if is_commercial else 0


def create_temporal_features(hour, day_of_week):
    """Create cyclical temporal encodings"""
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    day_sin = np.sin(2 * np.pi * day_of_week / 7)
    day_cos = np.cos(2 * np.pi * day_of_week / 7)
    return np.array([hour_sin, hour_cos, day_sin, day_cos])


def create_curved_path_with_direction(lat1, lng1, lat2, lng2, curve_height=0.3, num_points=20):
    """Create curved path with direction info"""
    mid_lat = (lat1 + lat2) / 2
    mid_lng = (lng1 + lng2) / 2
    distance = math.sqrt((lat2 - lat1) ** 2 + (lng2 - lng1) ** 2)
    
    dx = lng2 - lng1
    dy = lat2 - lat1
    perp_dx = -dy
    perp_dy = dx
    
    perp_length = math.sqrt(perp_dx**2 + perp_dy**2)
    if perp_length > 0:
        perp_dx /= perp_length
        perp_dy /= perp_length
    
    curve_offset = distance * curve_height
    control_lat = mid_lat + perp_dy * curve_offset
    control_lng = mid_lng + perp_dx * curve_offset
    
    path_points = []
    for i in range(num_points + 1):
        t = i / num_points
        lat = (1 - t) ** 2 * lat1 + 2 * (1 - t) * t * control_lat + t**2 * lat2
        lng = (1 - t) ** 2 * lng1 + 2 * (1 - t) * t * control_lng + t**2 * lng2
        path_points.append([lat, lng])
    
    mid_idx = len(path_points) // 2
    if mid_idx > 0 and mid_idx < len(path_points) - 1:
        before_point = path_points[mid_idx - 1]
        after_point = path_points[mid_idx + 1]
        dir_dx = after_point[1] - before_point[1]
        dir_dy = after_point[0] - before_point[0]
        direction_angle = math.degrees(math.atan2(dir_dy, dir_dx))
        midpoint = path_points[mid_idx]
    else:
        direction_angle = math.degrees(math.atan2(lat2 - lat1, lng2 - lng1))
        midpoint = [mid_lat, mid_lng]
    
    return path_points, midpoint, direction_angle


def create_draggable_html_legend(legend_items, title="Legend", position="top-right"):
    """Create draggable HTML legend"""
    position_styles = {
        "top-right": "top: 10px; right: 10px;",
        "top-left": "top: 10px; left: 10px;",
        "bottom-right": "bottom: 10px; right: 10px;",
        "bottom-left": "bottom: 10px; left: 10px;",
    }
    
    legend_html = f"""
    <div id="draggable-legend" style="position: absolute; 
                                     {position_styles.get(position, position_styles["top-right"])}
                                     cursor: move; width: auto; 
                                     background-color: rgba(255, 255, 255, 0.95); 
                                     border: 2px solid #333; border-radius: 8px;
                                     z-index: 9999; font-size: 14px; padding: 15px;
                                     box-shadow: 0 4px 12px rgba(0,0,0,0.3);
                                     font-family: Arial, sans-serif; user-select: none;">
        <div style="font-weight: bold; margin-bottom: 10px; text-align: center; 
                    font-size: 16px; color: #333; border-bottom: 1px solid #ccc;
                    padding-bottom: 8px; cursor: move;">{title}</div>
        <div id="legend-content">
    """
    
    for item in legend_items:
        if item["type"] == "circle":
            legend_html += f"""
            <div style="margin: 8px 0; display: flex; align-items: center;">
                <div style="width: 16px; height: 16px; background-color: {item['color']}; 
                           border-radius: 50%; border: 2px solid black; margin-right: 10px;"></div>
                <span style="color: #333; font-weight: 500;">{item['label']}</span>
            </div>"""
        elif item["type"] == "line":
            legend_html += f"""
            <div style="margin: 8px 0; display: flex; align-items: center;">
                <div style="width: 20px; height: {item.get('width', 3)}px; 
                           background-color: {item['color']}; margin-right: 10px;"></div>
                <span style="color: #333; font-weight: 500;">{item['label']}</span>
            </div>"""
        elif item["type"] == "arrow":
            legend_html += f"""
            <div style="margin: 8px 0; display: flex; align-items: center;">
                <div style="width: 0; height: 0; border-left: 8px solid {item['color']};
                           border-top: 6px solid transparent; border-bottom: 6px solid transparent;
                           margin-right: 12px;"></div>
                <span style="color: #333; font-weight: 500;">{item['label']}</span>
            </div>"""
    
    legend_html += """
        </div></div>
    <script>
    (function() {
        var legend = document.getElementById('draggable-legend');
        var isDragging = false, currentX, currentY, initialX, initialY, xOffset = 0, yOffset = 0;
        function dragStart(e) {
            if (e.type === "touchstart") {
                initialX = e.touches[0].clientX - xOffset;
                initialY = e.touches[0].clientY - yOffset;
            } else {
                initialX = e.clientX - xOffset;
                initialY = e.clientY - yOffset;
            }
            if (e.target === legend || e.target.parentNode === legend || e.target.parentNode.parentNode === legend) {
                isDragging = true;
            }
        }
        function dragEnd(e) { initialX = currentX; initialY = currentY; isDragging = false; }
        function drag(e) {
            if (isDragging) {
                e.preventDefault();
                if (e.type === "touchmove") {
                    currentX = e.touches[0].clientX - initialX;
                    currentY = e.touches[0].clientY - initialY;
                } else {
                    currentX = e.clientX - initialX;
                    currentY = e.clientY - initialY;
                }
                xOffset = currentX; yOffset = currentY;
                legend.style.transform = "translate3d(" + currentX + "px, " + currentY + "px, 0)";
            }
        }
        if (legend) {
            legend.addEventListener("mousedown", dragStart, false);
            legend.addEventListener("touchstart", dragStart, false);
            document.addEventListener("mouseup", dragEnd, false);
            document.addEventListener("touchend", dragEnd, false);
            document.addEventListener("mousemove", drag, false);
            document.addEventListener("touchmove", drag, false);
        }
    })();
    </script>"""
    
    return legend_html


def make_matplotlib_legend_draggable(ax, *args, **kwargs):
    """Create draggable matplotlib legend"""
    kwargs.setdefault("frameon", True)
    kwargs.setdefault("fancybox", True)
    kwargs.setdefault("shadow", True)
    kwargs.setdefault("framealpha", 0.9)
    kwargs.setdefault("facecolor", "white")
    kwargs.setdefault("edgecolor", "black")
    
    legend = ax.legend(*args, **kwargs)
    if kwargs.get("frameon") and legend.get_frame():
        legend.get_frame().set_linewidth(1.5)
    legend.set_draggable(True)
    return legend


###########################################
# DATA LOADING - WITH TRAIN/VAL SPLIT
###########################################

def load_citibike_data_multi_pattern(data_dir, patterns, year_label):
    """Load Citibike data from multiple patterns"""
    print(f"\nLoading {year_label} data...")
    all_data = []
    
    if isinstance(patterns, str):
        patterns = [patterns]
    
    for pattern in patterns:
        file_path = os.path.join(data_dir, pattern)
        files = glob.glob(file_path)
        
        if not files:
            print(f"  Warning: No files found for pattern: {pattern}")
            continue
        
        files.sort()
        for file in files:
            print(f"  Loading: {os.path.basename(file)}")
            df = pd.read_csv(file)
            all_data.append(df)
    
    if not all_data:
        raise FileNotFoundError(f"No files found for any pattern in {year_label}")
    
    trips_df = pd.concat(all_data, ignore_index=True)
    trips_df["started_at"] = pd.to_datetime(trips_df["started_at"], errors="coerce")
    trips_df["ended_at"] = pd.to_datetime(trips_df["ended_at"], errors="coerce")
    trips_df = trips_df.dropna(subset=["started_at", "ended_at"])
    
    trips_df["hour"] = trips_df["started_at"].dt.hour
    trips_df["day_of_week"] = trips_df["started_at"].dt.dayofweek
    trips_df["month"] = trips_df["started_at"].dt.month
    trips_df["year"] = trips_df["started_at"].dt.year
    
    required_cols = [
        "start_station_name", "end_station_name", 
        "start_lat", "start_lng", "end_lat", "end_lng",
    ]
    trips_df = trips_df.dropna(subset=required_cols)
    
    print(f"After cleaning: {len(trips_df):,} trips")
    print(f"Date range: {trips_df['started_at'].min()} to {trips_df['started_at'].max()}")
    return trips_df


def load_citibike_data(data_dir, file_pattern, year_label):
    """Load Citibike data (wrapper for backward compatibility)"""
    return load_citibike_data_multi_pattern(data_dir, file_pattern, year_label)


def split_train_validation(trips_df, validation_months=2):
    """Split data into train and validation sets"""
    max_month = trips_df["month"].max()
    validation_threshold = max_month - validation_months + 1
    
    train_df = trips_df[trips_df["month"] < validation_threshold].copy()
    val_df = trips_df[trips_df["month"] >= validation_threshold].copy()
    
    print(f"\nTrain/Validation Split:")
    print(f"  Train: {len(train_df):,} trips (months < {validation_threshold})")
    print(f"  Validation: {len(val_df):,} trips (months >= {validation_threshold})")
    
    return train_df, val_df


def get_top_stations(trips_df, num_stations):
    """Get top N stations"""
    start_counts = trips_df["start_station_name"].value_counts()
    end_counts = trips_df["end_station_name"].value_counts()
    total_counts = start_counts.add(end_counts, fill_value=0)
    
    top_stations = total_counts.nlargest(num_stations).index.tolist()
    
    stations_info = []
    for station in top_stations:
        station_data = trips_df[trips_df["start_station_name"] == station].iloc[0]
        zone_type = classify_zone_type(station)
        
        stations_info.append({
            "name": station,
            "lat": station_data["start_lat"],
            "lng": station_data["start_lng"],
            "zone_type": zone_type,
            "zone_label": "Commercial" if zone_type == 1 else "Residential",
            "popularity": int(total_counts[station]),
        })
    
    return pd.DataFrame(stations_info)


def build_od_matrices(trips_df, stations_df, time_periods):
    """Build OD matrices"""
    station_names = stations_df["name"].tolist()
    station_to_idx = {name: idx for idx, name in enumerate(station_names)}
    num_stations = len(station_names)
    
    od_data = {}
    
    for period_name, (start_hour, end_hour) in time_periods.items():
        print(f"Processing {period_name}...")
        
        if period_name == "all_day":
            period_trips = trips_df[
                (trips_df["start_station_name"].isin(station_names))
                & (trips_df["end_station_name"].isin(station_names))
            ].copy()
        else:
            period_trips = trips_df[
                (trips_df["hour"] >= start_hour)
                & (trips_df["hour"] < end_hour)
                & (trips_df["start_station_name"].isin(station_names))
                & (trips_df["end_station_name"].isin(station_names))
            ].copy()
        
        od_matrix = np.zeros((num_stations, num_stations))
        trip_details = []
        
        for _, trip in period_trips.iterrows():
            try:
                orig_idx = station_to_idx[trip["start_station_name"]]
                dest_idx = station_to_idx[trip["end_station_name"]]
                od_matrix[orig_idx, dest_idx] += 1
                
                trip_details.append({
                    "origin_idx": orig_idx,
                    "dest_idx": dest_idx,
                    "hour": trip["hour"],
                    "day_of_week": trip["day_of_week"],
                })
            except KeyError:
                continue
        
        od_data[period_name] = {
            "od_matrix": od_matrix,
            "trip_details": pd.DataFrame(trip_details),
            "total_trips": od_matrix.sum(),
        }
    
    return od_data


def diagnose_data_quality(train_od, val_od, test_od, period_name):
    """Diagnose data quality and distribution"""
    print(f"\n{'='*60}")
    print(f"Data Quality Diagnostics: {period_name}")
    print(f"{'='*60}")
    
    # Sparsity
    train_sparsity = (train_od == 0).sum() / train_od.size * 100
    val_sparsity = (val_od == 0).sum() / val_od.size * 100
    test_sparsity = (test_od == 0).sum() / test_od.size * 100
    
    print(f"Sparsity (% zeros):")
    print(f"  Train: {train_sparsity:.1f}%")
    print(f"  Val:   {val_sparsity:.1f}%")
    print(f"  Test:  {test_sparsity:.1f}%")
    
    # Flow statistics
    print(f"\nFlow Statistics:")
    print(f"  Train: mean={train_od.mean():.2f}, max={train_od.max():.0f}, total={train_od.sum():.0f}")
    print(f"  Val:   mean={val_od.mean():.2f}, max={val_od.max():.0f}, total={val_od.sum():.0f}")
    print(f"  Test:  mean={test_od.mean():.2f}, max={test_od.max():.0f}, total={test_od.sum():.0f}")
    
    # Correlations
    train_flat = train_od.flatten()
    val_flat = val_od.flatten()
    test_flat = test_od.flatten()
    
    train_val_corr = np.corrcoef(train_flat, val_flat)[0, 1] if train_flat.std() > 0 and val_flat.std() > 0 else 0
    train_test_corr = np.corrcoef(train_flat, test_flat)[0, 1] if train_flat.std() > 0 and test_flat.std() > 0 else 0
    
    print(f"\nPattern Correlations:")
    print(f"  Train-Val:  {train_val_corr:.3f}")
    print(f"  Train-Test: {train_test_corr:.3f}")
    
    print(f"{'='*60}\n")


###########################################
# FEATURE ENGINEERING
###########################################

def compute_network_features(od_matrix, stations_df):
    """Compute network features"""
    num_stations = len(stations_df)
    outflow = od_matrix.sum(axis=1)
    inflow = od_matrix.sum(axis=0)
    degree = ((od_matrix > 0).sum(axis=1) + (od_matrix > 0).sum(axis=0)) / (num_stations - 1)
    return outflow, inflow, degree


def create_node_features(stations_df, od_matrix):
    """Create node features"""
    outflow, inflow, degree = compute_network_features(od_matrix, stations_df)
    return np.column_stack([
        stations_df["zone_type"].values,
        stations_df["lat"].values,
        stations_df["lng"].values,
        outflow,
        inflow,
        degree,
    ])


def create_edge_features(stations_df, node_features):
    """Create edge features"""
    num_stations = len(stations_df)
    edge_features = []
    
    for i in range(num_stations):
        for j in range(num_stations):
            lat_i, lng_i = stations_df.iloc[i][["lat", "lng"]]
            lat_j, lng_j = stations_df.iloc[j][["lat", "lng"]]
            
            distance = haversine_distance(lat_i, lng_i, lat_j, lng_j)
            dir_x = (lng_j - lng_i) / distance if distance > 0 else 0
            dir_y = (lat_j - lat_i) / distance if distance > 0 else 0
            
            edge_feat = np.concatenate([
                node_features[i],
                node_features[j],
                [distance, dir_x, dir_y]
            ])
            edge_features.append(edge_feat)
    
    return np.array(edge_features).reshape(num_stations, num_stations, -1)


###########################################
# PYTORCH DATASET & IMPROVED MODEL
###########################################

class ODDataset(Dataset):
    def __init__(self, od_matrix, edge_features, trip_details):
        self.od_matrix = torch.FloatTensor(od_matrix)
        self.edge_features = torch.FloatTensor(edge_features)
        
        if len(trip_details) > 0:
            grouped = trip_details.groupby(["origin_idx", "dest_idx"]).agg({
                "hour": "mean",
                "day_of_week": lambda x: x.mode()[0] if len(x) > 0 else 0,
            }).reset_index()
            
            num_stations = od_matrix.shape[0]
            temporal_features = np.zeros((num_stations, num_stations, 4))
            
            for _, row in grouped.iterrows():
                i, j = int(row["origin_idx"]), int(row["dest_idx"])
                temporal_features[i, j] = create_temporal_features(
                    row["hour"], row["day_of_week"]
                )
        else:
            temporal_features = np.zeros((od_matrix.shape[0], od_matrix.shape[1], 4))
        
        self.temporal_features = torch.FloatTensor(temporal_features)
        self.od_pairs = [
            (i, j) for i in range(od_matrix.shape[0]) for j in range(od_matrix.shape[1])
        ]
    
    def __len__(self):
        return len(self.od_pairs)
    
    def __getitem__(self, idx):
        i, j = self.od_pairs[idx]
        return {
            "edge_features": self.edge_features[i, j],
            "temporal_features": self.temporal_features[i, j],
            "origin_idx": i,
            "dest_idx": j,
            "flow": self.od_matrix[i, j],
        }


class ImprovedMessagePassingLayer(nn.Module):
    """Improved with batch normalization and residual connections"""
    def __init__(self, hidden_dim):
        super(ImprovedMessagePassingLayer, self).__init__()
        self.msg_nn = nn.Linear(hidden_dim + 3, hidden_dim)
        self.update_nn = nn.Linear(hidden_dim * 2, hidden_dim)
        self.batch_norm = nn.BatchNorm1d(hidden_dim)  # NEW
        self.activation = nn.ReLU()
        self.msg_dropout = nn.Dropout(0.2)  # NEW
    
    def forward(self, node_embeddings, edge_relations, adjacency):
        batch_size, num_nodes, hidden_dim = node_embeddings.shape
        messages = torch.zeros_like(node_embeddings)
        
        for j in range(num_nodes):
            neighbors = torch.where(adjacency[j] > 0)[0]
            if len(neighbors) == 0:
                continue
            
            for i in neighbors:
                msg_input = torch.cat([node_embeddings[:, i, :], edge_relations[:, i, j, :]], dim=-1)
                msg = self.msg_nn(msg_input)
                msg = self.msg_dropout(msg)  # NEW
                messages[:, j, :] += msg
        
        update_input = torch.cat([node_embeddings, messages], dim=-1)
        updated = self.activation(self.update_nn(update_input))
        
        # Batch normalization (reshape for BatchNorm1d)
        updated_flat = updated.reshape(-1, hidden_dim)
        updated_flat = self.batch_norm(updated_flat)
        updated = updated_flat.reshape(batch_size, num_nodes, hidden_dim)
        
        # Residual connection
        return updated + node_embeddings  # NEW


class ImprovedSGCNModel(nn.Module):
    """Improved SGCN with residual connections and batch norm"""
    def __init__(
        self,
        node_feature_dim=6,
        edge_feature_dim=15,
        temporal_dim=4,
        hidden_dim=256,
        num_layers=3,
        dropout=0.3,
    ):
        super(ImprovedSGCNModel, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.node_embedding = nn.Linear(node_feature_dim, hidden_dim)
        self.node_bn = nn.BatchNorm1d(hidden_dim)  # NEW
        
        self.message_layers = nn.ModuleList([
            ImprovedMessagePassingLayer(hidden_dim) for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        
        # Improved flow predictor - Using LayerNorm instead of BatchNorm
        # (LayerNorm works with batch_size=1, BatchNorm doesn't)
        self.flow_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 3 + temporal_dim, 128),
            nn.LayerNorm(128),  # Changed from BatchNorm1d
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.LayerNorm(64),  # Changed from BatchNorm1d
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.ReLU(),
        )
    
    def forward(self, node_features, edge_features, temporal_features, adjacency):
        batch_size, num_nodes = node_features.shape[0], node_features.shape[1]
        
        # Node embedding with batch norm
        node_embeddings = self.node_embedding(node_features)
        node_embeddings_flat = node_embeddings.reshape(-1, self.hidden_dim)
        node_embeddings_flat = self.node_bn(node_embeddings_flat)
        node_embeddings = node_embeddings_flat.reshape(batch_size, num_nodes, self.hidden_dim)
        node_embeddings = torch.relu(node_embeddings)
        node_embeddings = self.dropout(node_embeddings)
        
        edge_relations = edge_features[:, :, :, -3:]
        
        # Message passing with residual connections
        for layer in self.message_layers:
            node_embeddings = layer(node_embeddings, edge_relations, adjacency)
            node_embeddings = self.dropout(node_embeddings)
        
        flows = torch.zeros(batch_size, num_nodes, num_nodes, device=node_features.device)
        
        # Predict flows
        for i in range(num_nodes):
            for j in range(num_nodes):
                flow_input = torch.cat([
                    node_embeddings[:, i, :],
                    node_embeddings[:, j, :],
                    edge_relations[:, i, j, :],
                    temporal_features[:, i, j, :],
                ], dim=-1)
                
                flows[:, i, j] = self.flow_predictor(flow_input).squeeze(-1)
        
        return flows


###########################################
# BASELINE
###########################################

def train_gravity_model(train_od, stations_df):
    """Train gravity model"""
    num_stations = len(stations_df)
    outflow, inflow, _ = compute_network_features(train_od, stations_df)
    zone_types = stations_df["zone_type"].values
    
    A_orig = outflow + 0.1 * zone_types
    A_dest = inflow + 0.1 * zone_types
    
    distances = np.zeros((num_stations, num_stations))
    for i in range(num_stations):
        for j in range(num_stations):
            if i != j:
                distances[i, j] = haversine_distance(
                    stations_df.iloc[i]["lat"], stations_df.iloc[i]["lng"],
                    stations_df.iloc[j]["lat"], stations_df.iloc[j]["lng"],
                )
            else:
                distances[i, j] = 0.1
    
    beta = 2.0
    pred_od = np.outer(A_orig, A_dest) / (distances ** beta)
    K = train_od.sum() / pred_od.sum()
    return pred_od * K


###########################################
# TRAINING - WITH VALIDATION
###########################################

def create_adjacency_matrix(stations_df, od_matrix):
    """Create adjacency matrix"""
    num_stations = len(stations_df)
    adjacency = np.zeros((num_stations, num_stations))
    
    for i in range(num_stations):
        for j in range(num_stations):
            if i == j:
                adjacency[i, j] = 1
                continue
            
            dist = haversine_distance(
                stations_df.iloc[i]["lat"], stations_df.iloc[i]["lng"],
                stations_df.iloc[j]["lat"], stations_df.iloc[j]["lng"],
            )
            
            flow = od_matrix[i, j] + od_matrix[j, i]
            flow_threshold = np.percentile(od_matrix[od_matrix > 0], 70) if (od_matrix > 0).any() else 0
            
            # More lenient connectivity
            if dist < 3.0 or flow > flow_threshold:
                adjacency[i, j] = 1
    
    return adjacency


def train_sgcn(model, train_loader, node_features, edge_features, temporal_features, 
               adjacency, optimizer, criterion, device, epoch):
    """Train SGCN with progress indicators"""
    model.train()
    total_loss = 0
    num_batches = len(train_loader)
    
    node_feat_tensor = torch.FloatTensor(node_features).unsqueeze(0).to(device)
    edge_feat_tensor = torch.FloatTensor(edge_features).unsqueeze(0).to(device)
    temporal_feat_tensor = torch.FloatTensor(temporal_features).unsqueeze(0).to(device)
    adjacency_tensor = torch.FloatTensor(adjacency).to(device)
    
    # Progress indicator every 20% of batches
    report_interval = max(1, num_batches // 5)
    
    for batch_idx, batch in enumerate(train_loader):
        flows = batch["flow"].to(device)
        orig_idx = batch["origin_idx"].to(device)
        dest_idx = batch["dest_idx"].to(device)
        
        optimizer.zero_grad()
        pred_od_matrix = model(node_feat_tensor, edge_feat_tensor, temporal_feat_tensor, adjacency_tensor)
        predictions = pred_od_matrix[0, orig_idx, dest_idx]
        
        loss = criterion(predictions, flows)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        
        # Progress indicator
        if (batch_idx + 1) % report_interval == 0 or (batch_idx + 1) == num_batches:
            progress = (batch_idx + 1) / num_batches * 100
            avg_loss_so_far = total_loss / (batch_idx + 1)
            print(f"    Epoch {epoch+1} - Batch {batch_idx+1}/{num_batches} ({progress:.0f}%) - Avg Loss: {avg_loss_so_far:.4f}")
    
    return total_loss / num_batches


def evaluate_model(model, test_loader, node_features, edge_features, temporal_features, adjacency, device):
    """Evaluate model"""
    model.eval()
    predictions, actuals = [], []
    
    node_feat_tensor = torch.FloatTensor(node_features).unsqueeze(0).to(device)
    edge_feat_tensor = torch.FloatTensor(edge_features).unsqueeze(0).to(device)
    temporal_feat_tensor = torch.FloatTensor(temporal_features).unsqueeze(0).to(device)
    adjacency_tensor = torch.FloatTensor(adjacency).to(device)
    
    with torch.no_grad():
        pred_od_matrix = model(node_feat_tensor, edge_feat_tensor, temporal_feat_tensor, adjacency_tensor)
        
        for batch in test_loader:
            flows = batch["flow"].to(device)
            orig_idx = batch["origin_idx"].to(device)
            dest_idx = batch["dest_idx"].to(device)
            
            pred = pred_od_matrix[0, orig_idx, dest_idx]
            predictions.extend(pred.cpu().numpy())
            actuals.extend(flows.cpu().numpy())
    
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    mae = mean_absolute_error(actuals, predictions)
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    
    return mae, rmse, predictions, actuals


###########################################
# VISUALIZATIONS (keeping existing functions)
###########################################

def create_complete_folium_station_map(stations_df, output_dir):
    """Complete station map with all features"""
    center_lat = stations_df["lat"].mean()
    center_lng = stations_df["lng"].mean()
    
    m = folium.Map(location=[center_lat, center_lng], zoom_start=13, tiles=None)
    
    folium.TileLayer("cartodbpositron", name="Light", control=True, show=True).add_to(m)
    folium.TileLayer("openstreetmap", name="Standard", control=True, show=False).add_to(m)
    folium.TileLayer("cartodbdark_matter", name="Dark", control=True, show=False).add_to(m)
    
    legend_items = [
        {"type": "circle", "color": "#e74c3c", "label": "Residential"},
        {"type": "circle", "color": "#3498db", "label": "Commercial"},
    ]
    m.get_root().html.add_child(folium.Element(create_draggable_html_legend(legend_items, "Station Map")))
    
    res_group = folium.FeatureGroup(name="Residential", show=True)
    comm_group = folium.FeatureGroup(name="Commercial", show=True)
    labels_group = folium.FeatureGroup(name="Neighborhoods", show=True)
    
    for _, station in stations_df.iterrows():
        color = "#e74c3c" if station["zone_type"] == 0 else "#3498db"
        group = res_group if station["zone_type"] == 0 else comm_group
        
        folium.CircleMarker(
            location=[station["lat"], station["lng"]],
            radius=8,
            color="black",
            fill=True,
            fill_color=color,
            fill_opacity=0.8,
            weight=2,
            popup=f"<b>{station['name']}</b><br>{station['zone_label']}<br>Trips: {station['popularity']:,}",
        ).add_to(group)
    
    for name, (lat, lng) in NEIGHBORHOOD_LABELS.items():
        folium.Marker(
            location=[lat, lng],
            draggable=True,
            icon=folium.DivIcon(
                html=f'<div style="font-size: 14px; font-weight: bold; '
                     f'background-color: rgba(155, 89, 182, 0.9); color: white; '
                     f'border: 2px solid #8e44ad; padding: 6px 12px; border-radius: 8px; '
                     f'box-shadow: 0 2px 4px rgba(0,0,0,0.3); '
                     f'cursor: grab; white-space: nowrap; display: inline-block;">{name}</div>'
            ),
        ).add_to(labels_group)
    
    res_group.add_to(m)
    comm_group.add_to(m)
    labels_group.add_to(m)
    folium.LayerControl(position="topleft", collapsed=False).add_to(m)
    
    m.save(os.path.join(output_dir, "complete_station_map.html"))


def create_complete_heatmap(od_matrix, stations_df, period_name, output_dir):
    """Complete heatmap with draggable legend"""
    plt.figure(figsize=(24, 20))
    
    labels = [
        f"[{'C' if z==1 else 'R'}] {n[:20]}"
        for n, z in zip(stations_df["name"], stations_df["zone_type"])
    ]
    
    ax = plt.subplot(111)
    sns.heatmap(
        od_matrix,
        cmap="YlOrRd",
        square=True,
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={"label": "Trips", "shrink": 0.5},
        ax=ax,
    )
    
    plt.title(
        f"OD Matrix: {period_name}\n[R]=Residential | [C]=Commercial",
        fontsize=22,
        pad=30,
        fontweight="bold",
    )
    plt.xlabel("Destination", fontsize=18, labelpad=20)
    plt.ylabel("Origin", fontsize=18, labelpad=20)
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    
    legend_elements = [
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#e74c3c",
               markersize=16, label="[R] Residential", markeredgecolor="black"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#3498db",
               markersize=16, label="[C] Commercial", markeredgecolor="black"),
    ]
    
    make_matplotlib_legend_draggable(ax, handles=legend_elements, loc="upper left",
                                    fontsize=16, title="Zone Types", title_fontsize=18)
    
    plt.tight_layout()
    
    safe_name = period_name.lower().replace(" ", "_")
    plt.savefig(os.path.join(output_dir, f"heatmap_{safe_name}.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(output_dir, f"heatmap_{safe_name}.svg"), bbox_inches="tight")
    plt.close()


def plot_training_curves(train_losses, val_losses, period_name, output_dir):
    """Plot training and validation loss curves"""
    plt.figure(figsize=(12, 6))
    epochs = range(1, len(train_losses) + 1)
    
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.title(f'Training Curves: {period_name}', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    safe_name = period_name.lower().replace(" ", "_")
    plt.savefig(os.path.join(output_dir, f"training_curves_{safe_name}.png"), dpi=300, bbox_inches="tight")
    plt.close()


###########################################
# MAIN - WITH VALIDATION
###########################################

def main():
    """Main execution with validation"""
    os.makedirs(CONFIG["OUTPUT_DIR"], exist_ok=True)
    
    print("=" * 80)
    print("IMPROVED SGCN WITH VALIDATION")
    print("=" * 80)
    
    # Load data
    train_val_trips = load_citibike_data(CONFIG["DATA_DIR"], CONFIG["TRAIN_PATTERNS"], "2022-2023" if len(CONFIG["TRAIN_PATTERNS"]) > 1 else "2023")
    test_trips = load_citibike_data(CONFIG["DATA_DIR"], CONFIG["TEST_PATTERN"], "2024")
    
    # Split train/validation
    train_trips, val_trips = split_train_validation(train_val_trips, CONFIG["VALIDATION_MONTHS"])
    
    # Get stations (based on all 2023 data)
    stations_df = get_top_stations(train_val_trips, CONFIG["NUM_STATIONS"])
    stations_df.to_csv(os.path.join(CONFIG["OUTPUT_DIR"], "stations.csv"), index=False)
    
    print(f"\n{'='*80}")
    print(f"Using {len(stations_df)} stations")
    print(f"Residential: {(stations_df['zone_type'] == 0).sum()}")
    print(f"Commercial:  {(stations_df['zone_type'] == 1).sum()}")
    print(f"{'='*80}")
    
    # Create complete station map
    create_complete_folium_station_map(stations_df, CONFIG["OUTPUT_DIR"])
    
    # Build OD matrices
    train_od_data = build_od_matrices(train_trips, stations_df, CONFIG["TIME_PERIODS"])
    val_od_data = build_od_matrices(val_trips, stations_df, CONFIG["TIME_PERIODS"])
    test_od_data = build_od_matrices(test_trips, stations_df, CONFIG["TIME_PERIODS"])
    
    # Train and evaluate
    results_summary = []
    
    for period_name in CONFIG["TIME_PERIODS"].keys():
        print(f"\n{'='*80}")
        print(f"PROCESSING: {period_name.upper()}")
        print(f"{'='*80}")
        
        train_od = train_od_data[period_name]["od_matrix"]
        val_od = val_od_data[period_name]["od_matrix"]
        test_od = test_od_data[period_name]["od_matrix"]
        
        # Diagnose data quality
        diagnose_data_quality(train_od, val_od, test_od, period_name)
        
        # Normalize OD matrices
        max_flow = max(train_od.max(), val_od.max(), test_od.max())
        if max_flow > 0:
            train_od_normalized = train_od / max_flow
            val_od_normalized = val_od / max_flow
            test_od_normalized = test_od / max_flow
        else:
            train_od_normalized = train_od
            val_od_normalized = val_od
            test_od_normalized = test_od
        
        print(f"\nNormalization: max_flow={max_flow:.0f}")
        
        # Create features
        train_node_feat = create_node_features(stations_df, train_od)
        val_node_feat = create_node_features(stations_df, val_od)
        test_node_feat = create_node_features(stations_df, test_od)
        
        # Normalize node features
        node_scaler = StandardScaler()
        train_node_feat_norm = node_scaler.fit_transform(train_node_feat)
        val_node_feat_norm = node_scaler.transform(val_node_feat)
        test_node_feat_norm = node_scaler.transform(test_node_feat)
        
        # Create edge features
        train_edge_feat = create_edge_features(stations_df, train_node_feat)
        val_edge_feat = create_edge_features(stations_df, val_node_feat)
        test_edge_feat = create_edge_features(stations_df, test_node_feat)
        
        # Normalize edge features
        num_stations = len(stations_df)
        train_edge_flat = train_edge_feat.reshape(-1, train_edge_feat.shape[-1])
        val_edge_flat = val_edge_feat.reshape(-1, val_edge_feat.shape[-1])
        test_edge_flat = test_edge_feat.reshape(-1, test_edge_feat.shape[-1])
        
        edge_scaler = StandardScaler()
        train_edge_flat_norm = edge_scaler.fit_transform(train_edge_flat)
        val_edge_flat_norm = edge_scaler.transform(val_edge_flat)
        test_edge_flat_norm = edge_scaler.transform(test_edge_flat)
        
        train_edge_feat_norm = train_edge_flat_norm.reshape(num_stations, num_stations, -1)
        val_edge_feat_norm = val_edge_flat_norm.reshape(num_stations, num_stations, -1)
        test_edge_feat_norm = test_edge_flat_norm.reshape(num_stations, num_stations, -1)
        
        adjacency = create_adjacency_matrix(stations_df, train_od)
        
        # Datasets
        train_dataset = ODDataset(train_od_normalized, train_edge_feat_norm,
                                 train_od_data[period_name]["trip_details"])
        val_dataset = ODDataset(val_od_normalized, val_edge_feat_norm,
                               val_od_data[period_name]["trip_details"])
        test_dataset = ODDataset(test_od_normalized, test_edge_feat_norm,
                                test_od_data[period_name]["trip_details"])
        
        train_loader = DataLoader(train_dataset, batch_size=CONFIG["BATCH_SIZE"], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=CONFIG["BATCH_SIZE"], shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=CONFIG["BATCH_SIZE"], shuffle=False)
        
        # Model
        device = torch.device(CONFIG["DEVICE"])
        model = ImprovedSGCNModel(
            hidden_dim=CONFIG["EMBEDDING_DIM"],
            num_layers=CONFIG["NUM_LAYERS"],
            dropout=CONFIG["DROPOUT"],
        ).to(device)
        
        optimizer = optim.Adam(model.parameters(), lr=CONFIG["LEARNING_RATE"],
                              weight_decay=CONFIG["L2_REG"])
        criterion = nn.MSELoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min",
                                                         factor=0.5, patience=20)
        
        print(f"\nTraining Improved SGCN...")
        print(f"  Model: {sum(p.numel() for p in model.parameters()):,} parameters")
        print(f"  Training batches: {len(train_loader)}, Validation batches: {len(val_loader)}")
        print(f"  Device: {device}")
        print(f"  Starting training loop...\n")
        
        best_val_loss = float("inf")
        patience_counter = 0
        train_temporal_feat = train_dataset.temporal_features.numpy()
        val_temporal_feat = val_dataset.temporal_features.numpy()
        
        train_losses = []
        val_losses = []
        
        training_start_time = time.time()
        
        for epoch in range(CONFIG["NUM_EPOCHS"]):
            epoch_start_time = time.time()
            
            # Train
            train_loss = train_sgcn(
                model, train_loader, train_node_feat_norm, train_edge_feat_norm,
                train_temporal_feat, adjacency, optimizer, criterion, device, epoch
            )
            train_losses.append(train_loss)
            
            # Validate
            print(f"    Validating...")
            val_mae, val_rmse, _, _ = evaluate_model(
                model, val_loader, val_node_feat_norm, val_edge_feat_norm,
                val_temporal_feat, adjacency, device
            )
            val_losses.append(val_mae)
            
            scheduler.step(val_mae)
            
            epoch_time = time.time() - epoch_start_time
            
            if (epoch + 1) % 10 == 0:
                print(f"  ✓ Epoch {epoch+1}/{CONFIG['NUM_EPOCHS']} completed in {epoch_time:.1f}s")
                print(f"    Train Loss: {train_loss:.4f}, Val MAE: {val_mae:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}\n")
            
            # Save best model based on validation
            if val_mae < best_val_loss:
                best_val_loss = val_mae
                patience_counter = 0
                torch.save(
                    model.state_dict(),
                    os.path.join(CONFIG["OUTPUT_DIR"], f"sgcn_{period_name}.pth")
                )
                if (epoch + 1) % 10 != 0:  # Don't print if we just printed above
                    print(f"    ✓ New best model saved (Val MAE: {val_mae:.4f})")
            else:
                patience_counter += 1
            
            if patience_counter >= CONFIG["PATIENCE"]:
                print(f"\n  ⚠ Early stopping at epoch {epoch+1} (no improvement for {CONFIG['PATIENCE']} epochs)")
                print(f"  Best validation MAE: {best_val_loss:.4f}\n")
                break
        
        total_training_time = time.time() - training_start_time
        print(f"  Training completed in {total_training_time:.1f}s ({total_training_time/60:.1f} min)")
        print(f"  Total epochs: {len(train_losses)}")
        
        # Plot training curves
        plot_training_curves(train_losses, val_losses, period_name, CONFIG["OUTPUT_DIR"])
        
        # Load best model
        model.load_state_dict(
            torch.load(os.path.join(CONFIG["OUTPUT_DIR"], f"sgcn_{period_name}.pth"))
        )
        
        # Evaluate on test set
        test_temporal_feat = test_dataset.temporal_features.numpy()
        mae, rmse, predictions, actuals = evaluate_model(
            model, test_loader, test_node_feat_norm, test_edge_feat_norm,
            test_temporal_feat, adjacency, device
        )
        
        # Denormalize
        predictions_denorm = predictions * max_flow
        actuals_denorm = actuals * max_flow
        
        mae_denorm = mean_absolute_error(actuals_denorm, predictions_denorm)
        rmse_denorm = np.sqrt(mean_squared_error(actuals_denorm, predictions_denorm))
        
        print(f"\nImproved SGCN Results:")
        print(f"  MAE:  {mae_denorm:.2f} trips")
        print(f"  RMSE: {rmse_denorm:.2f} trips")
        
        # Baseline
        gravity_pred = train_gravity_model(train_od, stations_df).flatten()
        gravity_mae = mean_absolute_error(actuals_denorm, gravity_pred)
        gravity_rmse = np.sqrt(mean_squared_error(actuals_denorm, gravity_pred))
        print(f"Gravity Model:")
        print(f"  MAE:  {gravity_mae:.2f} trips")
        print(f"  RMSE: {gravity_rmse:.2f} trips")
        
        improvement = ((gravity_mae - mae_denorm) / gravity_mae) * 100
        print(f"Improvement over Gravity: {improvement:.1f}%")
        
        results_summary.append({
            "Period": period_name,
            "SGCN_MAE": mae_denorm,
            "SGCN_RMSE": rmse_denorm,
            "Gravity_MAE": gravity_mae,
            "Gravity_RMSE": gravity_rmse,
            "Improvement_%": improvement,
        })
        
        # Visualizations
        print(f"\nGenerating visualizations...")
        pred_od = predictions_denorm.reshape(CONFIG["NUM_STATIONS"], CONFIG["NUM_STATIONS"])
        
        create_complete_heatmap(test_od, stations_df, f"{period_name}_actual", CONFIG["OUTPUT_DIR"])
        create_complete_heatmap(pred_od, stations_df, f"{period_name}_predicted", CONFIG["OUTPUT_DIR"])
    
    # Save results
    results_df = pd.DataFrame(results_summary)
    results_df.to_csv(os.path.join(CONFIG["OUTPUT_DIR"], "results_summary.csv"), index=False)
    
    print("\n" + "=" * 80)
    print("COMPLETE!")
    print("=" * 80)
    print(results_df.to_string(index=False))
    print(f"\nResults in: {CONFIG['OUTPUT_DIR']}")


if __name__ == "__main__":
    main()