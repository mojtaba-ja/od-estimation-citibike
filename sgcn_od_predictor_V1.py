"""
Spatial Graph Convolutional Network (SGCN) for Bike Sharing OD Matrix Prediction
WITH COMPLETE VISUALIZATION SUITE (All features from previous code)
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

# FIXED: Import matplotlib BEFORE using it
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
# CONFIGURATION
###########################################
CONFIG = {
    "DATA_DIR": "data/citibike",
    "TRAIN_PATTERN": "JC2023*-citibike-tripdata.csv",
    "TEST_PATTERN": "JC2024*citibiketripdata.csv",
    "OUTPUT_DIR": "results_sgcn",
    "NUM_STATIONS": 20,
    "TIME_PERIODS": {
        "morning_peak": (7, 10),
        "afternoon": (10, 16),
        "evening_peak": (16, 20),
        "all_day": (0, 24),
    },
    "EMBEDDING_DIM": 128,
    "NUM_LAYERS": 2,  # REDUCED from 3 to 2 for stability
    "DROPOUT": 0.3,
    "LEARNING_RATE": 0.0001,  # REDUCED from 0.001 to 0.0001
    "BATCH_SIZE": 64,
    "NUM_EPOCHS": 200,  # INCREASED from 100
    "L2_REG": 1e-5,
    "PATIENCE": 50,  # INCREASED from 30 to 50 to allow LR scheduler to work
    "CHUNK_SIZE": 10000,
    "SAMPLE_RATE": 1.0,
    "MIN_TRIPS_THRESHOLD": 5,
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
}

ZONE_CLASSIFICATION = {
    "commercial_keywords": [
        "downtown",
        "exchange",
        "newport",
        "harborside",
        "grove",
        "path",
        "plaza",
        "office",
        "center",
        "station",
        "transit",
        "christopher columbus",
        "essex",
        "liberty",
        "morris canal",
        "hudson",
        "waterfront",
        "financial",
    ],
    "residential_keywords": [
        "heights",
        "bergen",
        "lafayette",
        "greenville",
        "mcginley",
        "park",
        "school",
        "housing",
        "residential",
        "neighborhood",
        "home",
        "communipaw",
        "westside",
        "journal square",
        "marion",
        "duncan",
        "terrace",
        "garden",
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
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.asin(math.sqrt(a))
    return 6371 * c


def classify_zone_type(station_name):
    """Classify as residential (0) or commercial (1)"""
    name_lower = station_name.lower()
    is_commercial = any(
        kw in name_lower for kw in ZONE_CLASSIFICATION["commercial_keywords"]
    )
    return 1 if is_commercial else 0


def create_temporal_features(hour, day_of_week):
    """Create cyclical temporal encodings"""
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    day_sin = np.sin(2 * np.pi * day_of_week / 7)
    day_cos = np.cos(2 * np.pi * day_of_week / 7)
    return np.array([hour_sin, hour_cos, day_sin, day_cos])


def create_curved_path_with_direction(
    lat1, lng1, lat2, lng2, curve_height=0.3, num_points=20
):
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
# DATA LOADING
###########################################


def load_citibike_data(data_dir, file_pattern, year_label):
    """Load Citibike data"""
    print(f"\nLoading {year_label} data...")
    file_path = os.path.join(data_dir, file_pattern)
    files = glob.glob(file_path)

    if not files:
        raise FileNotFoundError(f"No files found: {file_path}")

    files.sort()
    all_data = []
    for file in files:
        print(f"  Loading: {os.path.basename(file)}")
        df = pd.read_csv(file)
        all_data.append(df)

    trips_df = pd.concat(all_data, ignore_index=True)
    trips_df["started_at"] = pd.to_datetime(trips_df["started_at"], errors="coerce")
    trips_df["ended_at"] = pd.to_datetime(trips_df["ended_at"], errors="coerce")
    trips_df = trips_df.dropna(subset=["started_at", "ended_at"])

    trips_df["hour"] = trips_df["started_at"].dt.hour
    trips_df["day_of_week"] = trips_df["started_at"].dt.dayofweek

    required_cols = [
        "start_station_name",
        "end_station_name",
        "start_lat",
        "start_lng",
        "end_lat",
        "end_lng",
    ]
    trips_df = trips_df.dropna(subset=required_cols)

    print(f"After cleaning: {len(trips_df):,} trips")
    return trips_df


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

        stations_info.append(
            {
                "name": station,
                "lat": station_data["start_lat"],
                "lng": station_data["start_lng"],
                "zone_type": zone_type,
                "zone_label": "Commercial" if zone_type == 1 else "Residential",
                "popularity": int(total_counts[station]),
            }
        )

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

                trip_details.append(
                    {
                        "origin_idx": orig_idx,
                        "dest_idx": dest_idx,
                        "hour": trip["hour"],
                        "day_of_week": trip["day_of_week"],
                    }
                )
            except KeyError:
                continue

        od_data[period_name] = {
            "od_matrix": od_matrix,
            "trip_details": pd.DataFrame(trip_details),
            "total_trips": od_matrix.sum(),
        }

    return od_data


###########################################
# FEATURE ENGINEERING
###########################################


def compute_network_features(od_matrix, stations_df):
    """Compute network features"""
    num_stations = len(stations_df)
    outflow = od_matrix.sum(axis=1)
    inflow = od_matrix.sum(axis=0)
    degree = ((od_matrix > 0).sum(axis=1) + (od_matrix > 0).sum(axis=0)) / (
        num_stations - 1
    )
    return outflow, inflow, degree


def create_node_features(stations_df, od_matrix):
    """Create node features"""
    outflow, inflow, degree = compute_network_features(od_matrix, stations_df)
    return np.column_stack(
        [
            stations_df["zone_type"].values,
            stations_df["lat"].values,
            stations_df["lng"].values,
            outflow,
            inflow,
            degree,
        ]
    )


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

            edge_feat = np.concatenate(
                [node_features[i], node_features[j], [distance, dir_x, dir_y]]
            )
            edge_features.append(edge_feat)

    return np.array(edge_features).reshape(num_stations, num_stations, -1)


###########################################
# PYTORCH DATASET & MODEL
###########################################


class ODDataset(Dataset):
    def __init__(self, od_matrix, edge_features, trip_details):
        self.od_matrix = torch.FloatTensor(od_matrix)
        self.edge_features = torch.FloatTensor(edge_features)

        if len(trip_details) > 0:
            grouped = (
                trip_details.groupby(["origin_idx", "dest_idx"])
                .agg(
                    {
                        "hour": "mean",
                        "day_of_week": lambda x: x.mode()[0] if len(x) > 0 else 0,
                    }
                )
                .reset_index()
            )

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


class MessagePassingLayer(nn.Module):
    def __init__(self, hidden_dim):
        super(MessagePassingLayer, self).__init__()
        self.msg_nn = nn.Linear(hidden_dim + 3, hidden_dim)
        self.update_nn = nn.Linear(hidden_dim * 2, hidden_dim)
        self.activation = nn.ReLU()

    def forward(self, node_embeddings, edge_relations, adjacency):
        batch_size, num_nodes, hidden_dim = node_embeddings.shape
        messages = torch.zeros_like(node_embeddings)

        for j in range(num_nodes):
            neighbors = torch.where(adjacency[j] > 0)[0]
            if len(neighbors) == 0:
                continue

            for i in neighbors:
                msg_input = torch.cat(
                    [node_embeddings[:, i, :], edge_relations[:, i, j, :]], dim=-1
                )
                msg = self.msg_nn(msg_input)
                messages[:, j, :] += msg

        update_input = torch.cat([node_embeddings, messages], dim=-1)
        return self.activation(self.update_nn(update_input))


class SGCNModel(nn.Module):
    def __init__(
        self,
        node_feature_dim=6,
        edge_feature_dim=15,
        temporal_dim=4,
        hidden_dim=128,
        num_layers=3,
        dropout=0.3,
    ):
        super(SGCNModel, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.node_embedding = nn.Linear(node_feature_dim, hidden_dim)
        self.message_layers = nn.ModuleList(
            [MessagePassingLayer(hidden_dim) for _ in range(num_layers)]
        )
        self.dropout = nn.Dropout(dropout)

        self.flow_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 3 + temporal_dim, 64),
            nn.ReLU(),
            self.dropout,
            nn.Linear(64, 1),
            nn.ReLU(),
        )

    def forward(self, node_features, edge_features, temporal_features, adjacency):
        batch_size, num_nodes = node_features.shape[0], node_features.shape[1]

        node_embeddings = torch.relu(self.node_embedding(node_features))
        node_embeddings = self.dropout(node_embeddings)

        edge_relations = edge_features[:, :, :, -3:]

        for layer in self.message_layers:
            node_embeddings = layer(node_embeddings, edge_relations, adjacency)
            node_embeddings = self.dropout(node_embeddings)

        flows = torch.zeros(
            batch_size, num_nodes, num_nodes, device=node_features.device
        )

        for i in range(num_nodes):
            for j in range(num_nodes):
                flow_input = torch.cat(
                    [
                        node_embeddings[:, i, :],
                        node_embeddings[:, j, :],
                        edge_relations[:, i, j, :],
                        temporal_features[:, i, j, :],
                    ],
                    dim=-1,
                )

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
                    stations_df.iloc[i]["lat"],
                    stations_df.iloc[i]["lng"],
                    stations_df.iloc[j]["lat"],
                    stations_df.iloc[j]["lng"],
                )
            else:
                distances[i, j] = 0.1

    beta = 2.0
    pred_od = np.outer(A_orig, A_dest) / (distances**beta)
    K = train_od.sum() / pred_od.sum()
    return pred_od * K


###########################################
# TRAINING
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
                stations_df.iloc[i]["lat"],
                stations_df.iloc[i]["lng"],
                stations_df.iloc[j]["lat"],
                stations_df.iloc[j]["lng"],
            )

            flow = od_matrix[i, j] + od_matrix[j, i]
            flow_threshold = (
                np.percentile(od_matrix[od_matrix > 0], 80)
                if (od_matrix > 0).any()
                else 0
            )

            if dist < 2.0 or flow > flow_threshold:
                adjacency[i, j] = 1

    return adjacency


def train_sgcn(
    model,
    train_loader,
    node_features,
    edge_features,
    temporal_features,
    adjacency,
    optimizer,
    criterion,
    device,
    epoch,
):
    """Train SGCN"""
    model.train()
    total_loss = 0
    num_batches = 0

    node_feat_tensor = torch.FloatTensor(node_features).unsqueeze(0).to(device)
    edge_feat_tensor = torch.FloatTensor(edge_features).unsqueeze(0).to(device)
    temporal_feat_tensor = torch.FloatTensor(temporal_features).unsqueeze(0).to(device)
    adjacency_tensor = torch.FloatTensor(adjacency).to(device)

    for batch in train_loader:
        flows = batch["flow"].to(device)
        orig_idx = batch["origin_idx"].to(device)
        dest_idx = batch["dest_idx"].to(device)

        optimizer.zero_grad()
        pred_od_matrix = model(
            node_feat_tensor, edge_feat_tensor, temporal_feat_tensor, adjacency_tensor
        )
        predictions = pred_od_matrix[0, orig_idx, dest_idx]

        loss = criterion(predictions, flows)
        loss.backward()

        # Gradient clipping to prevent explosion
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


def evaluate_model(
    model,
    test_loader,
    node_features,
    edge_features,
    temporal_features,
    adjacency,
    device,
):
    """Evaluate model"""
    model.eval()
    predictions, actuals = [], []

    node_feat_tensor = torch.FloatTensor(node_features).unsqueeze(0).to(device)
    edge_feat_tensor = torch.FloatTensor(edge_features).unsqueeze(0).to(device)
    temporal_feat_tensor = torch.FloatTensor(temporal_features).unsqueeze(0).to(device)
    adjacency_tensor = torch.FloatTensor(adjacency).to(device)

    with torch.no_grad():
        pred_od_matrix = model(
            node_feat_tensor, edge_feat_tensor, temporal_feat_tensor, adjacency_tensor
        )

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
# COMPLETE VISUALIZATION SUITE
###########################################


def create_complete_folium_station_map(stations_df, output_dir):
    """Complete station map with all features"""
    center_lat = stations_df["lat"].mean()
    center_lng = stations_df["lng"].mean()

    m = folium.Map(location=[center_lat, center_lng], zoom_start=13, tiles=None)

    folium.TileLayer("cartodbpositron", name="Light", control=True, show=True).add_to(m)
    folium.TileLayer("openstreetmap", name="Standard", control=True, show=False).add_to(
        m
    )
    folium.TileLayer(
        "cartodbdark_matter", name="Dark", control=True, show=False
    ).add_to(m)
    folium.TileLayer(
        "stamenterrain",
        name="Terrain",
        control=True,
        show=False,
        attr="Map tiles by Stamen Design, under CC BY 3.0. Data by OpenStreetMap, under ODbL.",
    ).add_to(m)

    legend_items = [
        {"type": "circle", "color": "#e74c3c", "label": "Residential"},
        {"type": "circle", "color": "#3498db", "label": "Commercial"},
    ]
    m.get_root().html.add_child(
        folium.Element(create_draggable_html_legend(legend_items, "Station Map"))
    )

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
                f"background-color: rgba(155, 89, 182, 0.9); color: white; "
                f"border: 2px solid #8e44ad; padding: 6px 12px; border-radius: 8px; "
                f"box-shadow: 0 2px 4px rgba(0,0,0,0.3); "
                f'cursor: grab; white-space: nowrap; display: inline-block;">{name}</div>'
            ),
        ).add_to(labels_group)

    res_group.add_to(m)
    comm_group.add_to(m)
    labels_group.add_to(m)
    folium.LayerControl(position="topleft", collapsed=False).add_to(m)

    m.save(os.path.join(output_dir, "complete_station_map.html"))


def create_complete_flow_map(
    od_matrix, stations_df, period_name, output_dir, min_trips=5
):
    """Complete flow map with all features"""
    center_lat = stations_df["lat"].mean()
    center_lng = stations_df["lng"].mean()

    m = folium.Map(location=[center_lat, center_lng], zoom_start=13, tiles=None)

    folium.TileLayer("cartodbpositron", name="Light", control=True, show=True).add_to(m)
    folium.TileLayer("openstreetmap", name="Standard", control=True, show=False).add_to(
        m
    )
    folium.TileLayer(
        "cartodbdark_matter", name="Dark", control=True, show=False
    ).add_to(m)

    legend_items = [
        {"type": "circle", "color": "#e74c3c", "label": "Residential"},
        {"type": "circle", "color": "#3498db", "label": "Commercial"},
        {"type": "line", "color": "#2ecc71", "width": 6, "label": "High Flow"},
        {"type": "line", "color": "#f39c12", "width": 4, "label": "Medium Flow"},
        {"type": "line", "color": "#6595b5", "width": 3, "label": "Low Flow"},
        {"type": "arrow", "color": "#2ecc71", "label": "Direction"},
    ]
    m.get_root().html.add_child(
        folium.Element(
            create_draggable_html_legend(legend_items, f"Flow Map: {period_name}")
        )
    )

    high_flow = folium.FeatureGroup(name="High Flow", show=True)
    med_flow = folium.FeatureGroup(name="Medium Flow", show=True)
    low_flow = folium.FeatureGroup(name="Low Flow", show=True)
    arrows = folium.FeatureGroup(name="Arrows", show=True)
    labels_group = folium.FeatureGroup(name="Neighborhoods", show=True)

    for _, station in stations_df.iterrows():
        color = "#e74c3c" if station["zone_type"] == 0 else "#3498db"
        folium.CircleMarker(
            location=[station["lat"], station["lng"]],
            radius=6,
            color="black",
            fill=True,
            fill_color=color,
            fill_opacity=0.8,
            weight=2,
        ).add_to(m)

    max_flow = od_matrix.max().max()
    flows = [
        od_matrix.iloc[i, j]
        for i in range(len(od_matrix))
        for j in range(len(od_matrix))
        if i != j and od_matrix.iloc[i, j] >= min_trips
    ]

    if flows:
        high_thresh = np.percentile(flows, 80)
        med_thresh = np.percentile(flows, 50)

        for i in range(len(od_matrix)):
            for j in range(len(od_matrix)):
                if i != j and od_matrix.iloc[i, j] >= min_trips:
                    flow = od_matrix.iloc[i, j]
                    start = stations_df.iloc[i]
                    end = stations_df.iloc[j]

                    if flow >= high_thresh:
                        color, weight, group = (
                            "#2ecc71",
                            max(4, min(8, flow / max_flow * 10)),
                            high_flow,
                        )
                        curve_height, arrow_size = 0.4, 12
                    elif flow >= med_thresh:
                        color, weight, group = (
                            "#f39c12",
                            max(2, min(5, flow / max_flow * 8)),
                            med_flow,
                        )
                        curve_height, arrow_size = 0.3, 10
                    else:
                        color, weight, group = (
                            "#6595b5",
                            max(2, min(4, flow / max_flow * 6)),
                            low_flow,
                        )
                        curve_height, arrow_size = 0.2, 8

                    path, midpoint, angle = create_curved_path_with_direction(
                        start["lat"], start["lng"], end["lat"], end["lng"], curve_height
                    )

                    folium.PolyLine(
                        locations=path,
                        color=color,
                        weight=weight,
                        opacity=0.7,
                        popup=f"{start['name']} → {end['name']}<br>Trips: {flow:.0f}",
                    ).add_to(group)

                    if midpoint:
                        folium.Marker(
                            location=midpoint,
                            icon=folium.DivIcon(
                                html=f'<div style="transform: rotate({angle}deg); '
                                f"font-size: {arrow_size}px; color: {color}; "
                                f'text-shadow: 1px 1px 2px rgba(0,0,0,0.8);">▶</div>'
                            ),
                        ).add_to(arrows)

    for name, (lat, lng) in NEIGHBORHOOD_LABELS.items():
        folium.Marker(
            location=[lat, lng],
            draggable=True,
            icon=folium.DivIcon(
                html=f'<div style="font-size: 13px; font-weight: bold; '
                f"background-color: rgba(155, 89, 182, 0.9); color: white; "
                f"border: 2px solid #8e44ad; padding: 5px 10px; border-radius: 7px; "
                f"box-shadow: 0 2px 4px rgba(0,0,0,0.4); "
                f'cursor: grab; white-space: nowrap; display: inline-block;">{name}</div>'
            ),
        ).add_to(labels_group)

    high_flow.add_to(m)
    med_flow.add_to(m)
    low_flow.add_to(m)
    arrows.add_to(m)
    labels_group.add_to(m)
    folium.LayerControl(position="topleft", collapsed=False).add_to(m)

    safe_name = period_name.lower().replace(" ", "_")
    m.save(os.path.join(output_dir, f"complete_flow_map_{safe_name}.html"))


def create_complete_heatmap(od_matrix, stations_df, period_name, output_dir):
    """Complete heatmap with draggable legend and dual export"""
    plt.figure(figsize=(22, 18))

    labels = [
        f"[{'C' if z==1 else 'R'}] {n[:25]}"
        for n, z in zip(stations_df["name"], stations_df["zone_type"])
    ]

    ax = plt.subplot(111)
    sns.heatmap(
        od_matrix,
        cmap="YlOrRd",
        square=True,
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={"label": "Trips", "shrink": 0.6},
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
    plt.xticks(rotation=45, ha="right", fontsize=14)
    plt.yticks(rotation=0, fontsize=14)

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markerfacecolor="#e74c3c",
            markersize=16,
            label="[R] Residential",
            markeredgecolor="black",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markerfacecolor="#3498db",
            markersize=16,
            label="[C] Commercial",
            markeredgecolor="black",
        ),
    ]

    make_matplotlib_legend_draggable(
        ax,
        handles=legend_elements,
        loc="upper left",
        fontsize=16,
        title="Zone Types",
        title_fontsize=18,
    )

    plt.tight_layout()

    safe_name = period_name.lower().replace(" ", "_")
    plt.savefig(
        os.path.join(output_dir, f"heatmap_{safe_name}.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.savefig(
        os.path.join(output_dir, f"heatmap_{safe_name}.svg"), bbox_inches="tight"
    )
    plt.close()


def create_flow_analysis(trips_df, stations_df, output_dir):
    """Flow direction analysis"""
    fig, axes = plt.subplots(2, 2, figsize=(22, 18))
    fig.suptitle("Flow Direction Analysis", fontsize=20, fontweight="bold")

    zone_map = {s["name"]: s["zone_label"].lower() for _, s in stations_df.iterrows()}

    for idx, (period_name, (start_h, end_h)) in enumerate(
        CONFIG["TIME_PERIODS"].items()
    ):
        ax = axes[idx // 2, idx % 2]

        if period_name == "all_day":
            period_trips = trips_df
        else:
            period_trips = trips_df[
                (trips_df["hour"] >= start_h) & (trips_df["hour"] < end_h)
            ]

        flow_counts = {"Res→Comm": 0, "Comm→Res": 0, "Res→Res": 0, "Comm→Comm": 0}

        for _, trip in period_trips.iterrows():
            orig = zone_map.get(trip["start_station_name"], "unknown")
            dest = zone_map.get(trip["end_station_name"], "unknown")

            if orig == "residential" and dest == "commercial":
                flow_counts["Res→Comm"] += 1
            elif orig == "commercial" and dest == "residential":
                flow_counts["Comm→Res"] += 1
            elif orig == "residential" and dest == "residential":
                flow_counts["Res→Res"] += 1
            elif orig == "commercial" and dest == "commercial":
                flow_counts["Comm→Comm"] += 1

        colors = ["#e74c3c", "#3498db", "#e67e22", "#9b59b6"]
        bars = ax.bar(
            flow_counts.keys(),
            flow_counts.values(),
            color=colors,
            alpha=0.8,
            edgecolor="black",
        )

        total = sum(flow_counts.values())
        for bar, value in zip(bars, flow_counts.values()):
            pct = (value / total * 100) if total > 0 else 0
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + total * 0.01,
                f"{pct:.1f}%\n({value:,})",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        ax.set_title(period_name, fontsize=16, fontweight="bold")
        ax.set_ylabel("Trips", fontsize=13)
        ax.tick_params(axis="x", rotation=45)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "flow_analysis.png"), dpi=300, bbox_inches="tight"
    )
    plt.savefig(os.path.join(output_dir, "flow_analysis.svg"), bbox_inches="tight")
    plt.close()


def create_topology_overview(stations_df, output_dir):
    """Network topology"""
    fig, ax = plt.subplots(figsize=(16, 12))

    for zone_type, marker, color, label in [
        (0, "o", "#e74c3c", "Residential"),
        (1, "s", "#3498db", "Commercial"),
    ]:
        subset = stations_df[stations_df["zone_type"] == zone_type]
        if len(subset) > 0:
            ax.scatter(
                subset["lng"],
                subset["lat"],
                c=color,
                marker=marker,
                s=120,
                alpha=0.8,
                edgecolors="black",
                linewidth=1.5,
                label=label,
            )

    for name, (lat, lng) in NEIGHBORHOOD_LABELS.items():
        ax.annotate(
            name,
            (lng, lat),
            fontsize=13,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.9),
        )

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#e74c3c",
            markersize=14,
            label="Residential",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markerfacecolor="#3498db",
            markersize=14,
            label="Commercial",
        ),
    ]
    make_matplotlib_legend_draggable(
        ax, handles=legend_elements, loc="lower right", fontsize=14
    )

    ax.set_title("Network Topology", fontsize=18, fontweight="bold")
    ax.set_xlabel("Longitude", fontsize=14)
    ax.set_ylabel("Latitude", fontsize=14)
    ax.grid(True, alpha=0.4)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "topology.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(output_dir, "topology.svg"), bbox_inches="tight")
    plt.close()


###########################################
# MAIN
###########################################


def main():
    """Main execution"""
    os.makedirs(CONFIG["OUTPUT_DIR"], exist_ok=True)

    print("=" * 80)
    print("SGCN WITH COMPLETE VISUALIZATION SUITE")
    print("=" * 80)

    # Load data
    train_trips = load_citibike_data(
        CONFIG["DATA_DIR"], CONFIG["TRAIN_PATTERN"], "2023"
    )
    test_trips = load_citibike_data(CONFIG["DATA_DIR"], CONFIG["TEST_PATTERN"], "2024")

    # Get stations
    stations_df = get_top_stations(train_trips, CONFIG["NUM_STATIONS"])
    stations_df.to_csv(os.path.join(CONFIG["OUTPUT_DIR"], "stations.csv"), index=False)

    # Create complete station map
    create_complete_folium_station_map(stations_df, CONFIG["OUTPUT_DIR"])

    # Build OD matrices
    train_od_data = build_od_matrices(train_trips, stations_df, CONFIG["TIME_PERIODS"])
    test_od_data = build_od_matrices(test_trips, stations_df, CONFIG["TIME_PERIODS"])

    # Train and evaluate
    results_summary = []

    for period_name in CONFIG["TIME_PERIODS"].keys():
        print(f"\n{'='*80}")
        print(f"PROCESSING: {period_name.upper()}")
        print(f"{'='*80}")

        train_od = train_od_data[period_name]["od_matrix"]
        test_od = test_od_data[period_name]["od_matrix"]

        # NORMALIZE OD matrices to [0, 1] for stable training
        max_flow = max(train_od.max(), test_od.max())
        if max_flow > 0:
            train_od_normalized = train_od / max_flow
            test_od_normalized = test_od / max_flow
        else:
            train_od_normalized = train_od
            test_od_normalized = test_od

        print(f"\nNormalization: max_flow={max_flow:.0f}")
        print(
            f"Train OD normalized range: [{train_od_normalized.min():.3f}, {train_od_normalized.max():.3f}]"
        )
        print(
            f"Test OD normalized range: [{test_od_normalized.min():.3f}, {test_od_normalized.max():.3f}]"
        )

        # Create features (using original OD for network statistics)
        train_node_feat = create_node_features(stations_df, train_od)
        test_node_feat = create_node_features(stations_df, test_od)

        # NORMALIZE NODE FEATURES using StandardScaler
        node_scaler = StandardScaler()
        train_node_feat_norm = node_scaler.fit_transform(train_node_feat)
        test_node_feat_norm = node_scaler.transform(test_node_feat)

        print(f"Node features normalized: mean≈0, std≈1")

        # Create edge features
        train_edge_feat = create_edge_features(stations_df, train_node_feat)
        test_edge_feat = create_edge_features(stations_df, test_node_feat)

        # NORMALIZE EDGE FEATURES
        num_stations = len(stations_df)
        train_edge_flat = train_edge_feat.reshape(-1, train_edge_feat.shape[-1])
        test_edge_flat = test_edge_feat.reshape(-1, test_edge_feat.shape[-1])

        edge_scaler = StandardScaler()
        train_edge_flat_norm = edge_scaler.fit_transform(train_edge_flat)
        test_edge_flat_norm = edge_scaler.transform(test_edge_flat)

        train_edge_feat_norm = train_edge_flat_norm.reshape(
            num_stations, num_stations, -1
        )
        test_edge_feat_norm = test_edge_flat_norm.reshape(
            num_stations, num_stations, -1
        )

        print(f"Edge features normalized: mean≈0, std≈1")

        adjacency = create_adjacency_matrix(stations_df, train_od)

        # Datasets - USE NORMALIZED FEATURES
        train_dataset = ODDataset(
            train_od_normalized,
            train_edge_feat_norm,
            train_od_data[period_name]["trip_details"],
        )
        test_dataset = ODDataset(
            test_od_normalized,
            test_edge_feat_norm,
            test_od_data[period_name]["trip_details"],
        )

        train_loader = DataLoader(
            train_dataset, batch_size=CONFIG["BATCH_SIZE"], shuffle=True
        )
        test_loader = DataLoader(
            test_dataset, batch_size=CONFIG["BATCH_SIZE"], shuffle=False
        )

        # Model
        device = torch.device(CONFIG["DEVICE"])
        model = SGCNModel(
            hidden_dim=CONFIG["EMBEDDING_DIM"],
            num_layers=CONFIG["NUM_LAYERS"],
            dropout=CONFIG["DROPOUT"],
        ).to(device)

        optimizer = optim.Adam(
            model.parameters(),
            lr=CONFIG["LEARNING_RATE"],
            weight_decay=CONFIG["L2_REG"],
        )
        criterion = nn.MSELoss()  # Changed from MAE to MSE for better gradients

        # Add learning rate scheduler for better convergence
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=15
        )

        # Training
        print(f"\nTraining SGCN...")
        best_loss = float("inf")
        patience_counter = 0
        train_temporal_feat = train_dataset.temporal_features.numpy()

        for epoch in range(CONFIG["NUM_EPOCHS"]):
            train_loss = train_sgcn(
                model,
                train_loader,
                train_node_feat_norm,
                train_edge_feat_norm,
                train_temporal_feat,
                adjacency,
                optimizer,
                criterion,
                device,
                epoch,
            )

            # Step the scheduler
            scheduler.step(train_loss)

            if (epoch + 1) % 10 == 0:
                print(
                    f"  Epoch {epoch+1}/{CONFIG['NUM_EPOCHS']}, Loss: {train_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}"
                )

            if train_loss < best_loss:
                best_loss = train_loss
                patience_counter = 0
                torch.save(
                    model.state_dict(),
                    os.path.join(CONFIG["OUTPUT_DIR"], f"sgcn_{period_name}.pth"),
                )
            else:
                patience_counter += 1

            if patience_counter >= CONFIG["PATIENCE"]:
                print(f"  Early stopping at epoch {epoch+1}")
                break

        # Load best
        model.load_state_dict(
            torch.load(os.path.join(CONFIG["OUTPUT_DIR"], f"sgcn_{period_name}.pth"))
        )

        # Evaluate
        test_temporal_feat = test_dataset.temporal_features.numpy()
        mae, rmse, predictions, actuals = evaluate_model(
            model,
            test_loader,
            test_node_feat_norm,
            test_edge_feat_norm,
            test_temporal_feat,
            adjacency,
            device,
        )

        # DENORMALIZE predictions and actuals back to trip counts
        predictions_denorm = predictions * max_flow
        actuals_denorm = actuals * max_flow

        # Calculate metrics on denormalized values
        mae_denorm = mean_absolute_error(actuals_denorm, predictions_denorm)
        rmse_denorm = np.sqrt(mean_squared_error(actuals_denorm, predictions_denorm))

        print(f"\nSGCN Results (normalized loss={mae:.4f}):")
        print(f"  MAE:  {mae_denorm:.2f} trips")
        print(f"  RMSE: {rmse_denorm:.2f} trips")

        # Baseline
        gravity_pred = train_gravity_model(train_od, stations_df).flatten()
        gravity_mae = mean_absolute_error(actuals_denorm, gravity_pred)
        gravity_rmse = np.sqrt(mean_squared_error(actuals_denorm, gravity_pred))
        print(f"Gravity Model:")
        print(f"  MAE:  {gravity_mae:.2f} trips")
        print(f"  RMSE: {gravity_rmse:.2f} trips")

        results_summary.append(
            {
                "Period": period_name,
                "SGCN_MAE": mae_denorm,
                "SGCN_RMSE": rmse_denorm,
                "Gravity_MAE": gravity_mae,
                "Gravity_RMSE": gravity_rmse,
            }
        )

        # Complete visualizations - USE DENORMALIZED PREDICTIONS
        print(f"\nGenerating complete visualizations...")
        pred_od = predictions_denorm.reshape(
            CONFIG["NUM_STATIONS"], CONFIG["NUM_STATIONS"]
        )

        # Complete flow map
        test_od_df = pd.DataFrame(test_od)
        create_complete_flow_map(
            test_od_df, stations_df, period_name, CONFIG["OUTPUT_DIR"]
        )

        # Complete heatmaps
        create_complete_heatmap(
            test_od, stations_df, f"{period_name}_actual", CONFIG["OUTPUT_DIR"]
        )
        create_complete_heatmap(
            pred_od, stations_df, f"{period_name}_predicted", CONFIG["OUTPUT_DIR"]
        )

    # Additional analyses
    create_flow_analysis(train_trips, stations_df, CONFIG["OUTPUT_DIR"])
    create_topology_overview(stations_df, CONFIG["OUTPUT_DIR"])

    # Save results
    results_df = pd.DataFrame(results_summary)
    results_df.to_csv(
        os.path.join(CONFIG["OUTPUT_DIR"], "results_summary.csv"), index=False
    )

    print("\n" + "=" * 80)
    print("COMPLETE!")
    print("=" * 80)
    print(results_df.to_string(index=False))
    print(f"\nResults in: {CONFIG['OUTPUT_DIR']}")


if __name__ == "__main__":
    main()
