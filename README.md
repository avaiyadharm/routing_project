# Dynamic Traffic-Optimized Routing Engine

A high-performance, machine learning-driven routing and fleet optimization engine designed to minimize multi-stop travel times by combining real-time contextual variables with physical road network constraints.

**Status**: ✅ **PHASE 1-4 COMPLETE** — Local MVP + Google Maps Live Production

---

## 🎯 Quick Start

### Option A — Google Maps Live Production (Recommended)

```bash
# 1. Set your Google Maps API key (one-time)
export GOOGLE_MAPS_API_KEY='your-api-key-here'

# 2. Activate environment & install dependencies
source .venv/bin/activate
pip install -r requirements.txt

# 3. Run the live optimizer (interactive CLI)
python route_optimizer_live.py
```

The script will prompt you for:
- **Source** & **Destination** as address strings (e.g. `Times Square, NY`)
- **Waypoints** separated by semicolons (e.g. `Brooklyn Bridge, NY; Central Park, NY`)
- **Departure time** (`HH:MM`), **day of week**, **weather**, and **event** flags

**Result:** Optimized multi-stop route via Google Distance Matrix + XGBoost ML overlay + OR-Tools solver.

> **Prerequisites:** Enable the **Geocoding API**, **Distance Matrix API**, and **Directions API** in your [Google Cloud Console](https://console.cloud.google.com/apis/library).

### Option B — Local Offline Routing (Legacy)

```bash
# 1. Setup (first time only)
source .venv/bin/activate
pip install -r requirements.txt
python src/graph_loader.py

# 2. Find optimal route
.venv/bin/python src/main.py --source 39.745,-75.546 --dest 39.758,-75.532 --hour 8

# 3. Run tests
.venv/bin/python -m pytest tests/ -v
```

**Result:** Route with 2.14 km distance in ~15.9 minutes (with Dijkstra + ML predictions)

---

## 🏗️ System Architecture

The core framework follows a decoupled, three-layer pipeline:

### Layer 1: Spatial Network (Phase 1 - graph_loader.py)
- **Input**: OpenStreetMap data (Delaware/Wilmington area)
- **Process**: Load and convert to NetworkX directed graph
- **Output**: 
  - `data/delaware_graph.pkl` (2278 nodes, 5220 edges)
  - `data/segment_features.csv` (edge metadata)
- **Key Function**: `load_osm_graph()` → Construct road network

### Layer 2: Predictive Traffic Model (Phase 2 - traffic_predictor.py)
- **Input**: Temporal features (hour, day, month, weather, events)
- **Model**: XGBoost Regressor (trained on NYC taxi data)
- **Process**: Predict travel time for each road segment
- **Output**: ML-adjusted travel time in seconds
- **Key Class**: `TrafficPredictor` → `predict_edge_time()`

### Layer 3: Prescriptive Routing (Phase 3 - router.py + main.py)
- **Algorithm**: Dijkstra's shortest path with dynamic edge weights
- **Input**: Source, destination, departure time, weather
- **Process**:
  1. Find nearest nodes to source/dest
  2. For each edge: query ML model for predicted travel time
  3. Execute Dijkstra using ML-predicted weights
  4. Return optimal path with waypoints
- **Output**: Route with coordinates, distance, ETA
- **Key Class**: `OptimalRouter` → `find_optimal_route()`

---

## 📊 Technical Specs

| Component | Technology | Details |
|-----------|-----------|---------|
| **Graph** | NetworkX 3.6 | MultiDiGraph (2278 nodes, 5220 edges) |
| **Map Data** | OSM via osmnx | Wilmington, Delaware area |
| **ML Model** | XGBoost 3.2 | Trained on 500k NYC taxi trips |
| **Pathfinding** | Dijkstra | O(E log V) complexity |
| **Features** | hour_of_day, day_of_week, month, is_raining, is_festival_zone | |

---

## 🧪 Validation

### All Tests Passing ✅
```
tests/test_graph_loader.py: 4/4 passed
tests/test_router.py: 8/8 passed
---
TOTAL: 12/12 PASSED
```

### Performance Metrics
- **Graph Load**: 8 seconds (Overpass API)
- **Route Query**: 0.7 seconds (Dijkstra on full graph)
- **ML Prediction**: 2ms per edge

### Example Output
```
Source:      39.7450, -75.5460
Destination: 39.7580, -75.5320
Distance:    2.14 km
Estimated:   15.9 minutes (956 seconds)
Segments:    26 turns
```

---

## 📁 Project Structure

```
routing_project/
├── src/
│   ├── __init__.py
│   ├── config.py                 # Constants and paths
│   ├── utils.py                  # Helper functions (haversine, nearest node, etc.)
│   ├── graph_loader.py           # Phase 1: Load OSM → build graph
│   ├── traffic_predictor.py      # Phase 2: ML predictions wrapper
│   ├── router.py                 # Phase 3: Dijkstra pathfinding
│   └── main.py                   # CLI entry point
├── tests/
│   ├── test_graph_loader.py     # Unit tests for graph loading
│   └── test_router.py           # Unit tests for routing & ML
├── data/
│   ├── delaware_graph.pkl        # Serialized NetworkX graph
│   ├── segment_features.csv      # Edge feature matrix
│   └── delaware-latest.osm.pbf   # Raw OSM file (reference)
├── train_traffic_model.py        # ML training pipeline (reference)
├── traffic_xgb_model.pkl         # Trained XGBoost model
├── train.csv                     # Training dataset (400k records)
├── requirements.txt              # Python dependencies
├── USAGE.md                      # Detailed usage guide
├── README.md                     # This file
└── Dockerfile                    # Docker deployment config
```

---

## 🚀 Usage Examples

### Example 1: Basic Route
```bash
python src/main.py --source 39.745,-75.546 --dest 39.758,-75.532
```

### Example 2: Morning Commute (8 AM, Monday)
```bash
python src/main.py --source 39.745,-75.546 --dest 39.758,-75.532 \
                   --hour 8 --day 0 --month 6
```

### Example 3: Evening Rush in Rain
```bash
python src/main.py --source 39.745,-75.546 --dest 39.758,-75.532 \
                   --hour 18 --day 4 --raining 1 --json
```

### Example 4: Python API
```python
from src.router import OptimalRouter

router = OptimalRouter()
result = router.find_optimal_route(
    start_lat=39.745, start_lon=-75.546,
    end_lat=39.758, end_lon=-75.532,
    departure_hour=9
)
print(f"ETA: {result['total_time_minutes']:.1f} minutes")
```

---

## 🔧 How It Works: The Algorithm

### Dijkstra's Algorithm with ML-Adjusted Weights

```
1. Get source/destination coordinates
2. Find nearest nodes in graph
3. Initialize: distance[start] = 0, all others = ∞
4. While unvisited nodes remain:
   a. Select unvisited node with smallest distance
   b. For each neighbor:
      - Query ML model: predicted_time = predict_traffic(hour, day, weather)
      - new_distance = current_distance + predicted_time
      - If new_distance < old_distance: update
   c. Mark node as visited
5. Return path from start to destination
```

### ML Traffic Model

The XGBoost model takes contextual features and outputs predicted travel time:

```
Input:  [hour=8, day_of_week=0, month=6, is_raining=0, is_festival=0]
Process: XGBoost tree ensemble
Output:  911.2 seconds (typical morning trip)

Input:  [hour=18, day_of_week=4, month=6, is_raining=1, is_festival=0]
Output:  862.5 seconds (evening, rainy)
```

Multiplier is applied to base edge time:
```
adjusted_time = base_time × (ml_predicted / baseline)
```

---

## 📈 Future Enhancements

1. **A* Search**: Add heuristic-guided pathfinding with great-circle distance
2. **Real-Time Data**: Integrate live traffic feeds (Google Maps API, INRIX)
3. **Multi-Stop VRP**: Extend to vehicle routing problems (keep existing OR-Tools implementation)
4. **Historical Patterns**: ML model with seasonal/historical data analysis
5. **Turn Restrictions**: Account for one-way streets, no-turn zones
6. **EV Charging**: Route optimization with charging station constraints

---

## 📚 Key References

### Algorithm
- **Dijkstra's Algorithm**: https://en.wikipedia.org/wiki/Dijkstra%27s_algorithm
- **Vehicle Routing Problem**: https://en.wikipedia.org/wiki/Vehicle_routing_problem
- **XGBoost**: https://arxiv.org/abs/1603.02754

### Libraries
- **NetworkX**: https://networkx.org/ (graph algorithms)
- **osmnx**: https://osmnx.readthedocs.io/ (OpenStreetMap data)
- **XGBoost**: https://xgboost.readthedocs.io/ (ML model)
- **Google OR-Tools**: https://developers.google.com/optimization (constraint programming)

### Data
- **NYC Taxi Dataset**: https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page
- **OpenStreetMap**: https://www.openstreetmap.org/

---

## 🛠️ Development & Testing

### Run All Tests
```bash
python -m pytest tests/ -v
```

### Individual Phases
```bash
# Phase 1: Load graph
python src/graph_loader.py

# Phase 2: Test ML model
python src/traffic_predictor.py

# Phase 3: Test routing
python src/router.py

# Phase 3: Test CLI
python src/main.py --help
```

---

## 📝 Mathematical Formulation

### Core Path Cost

Let the street network be represented as a directed graph $G = (V, E)$, where $V$ represents intersections and $E$ represents directed road segments.

For any edge $e \in E$, the travel time cost is:

$$\text{Cost}(e, t) = \text{BaseTime}(e) \times \frac{\text{ML\_Predict}(\mathbf{X}_t)}{\text{Baseline}}$$

Where:
- $\text{BaseTime}(e)$ = physical time based on distance and speed limit
- $\mathbf{X}_t$ = temporal/environmental features at time $t$
- $\text{ML\_Predict}(\cdot)$ = XGBoost model output

### Routing Objective

Find path $P = (e_1, e_2, \ldots, e_n)$ minimizing:

$$\min \sum_{e \in P} \text{Cost}(e, t)$$

Subject to connectivity constraints (no disconnected regions).

---

## 🤝 Contributing

For improvements:
1. Add features to `src/` modules
2. Update tests in `tests/`
3. Verify all tests pass: `pytest tests/ -v`
4. Update USAGE.md with new functionality

---

## 📞 Support

**Quick troubleshooting**:
- "Graph not found" → Run `python src/graph_loader.py`
- "ModuleNotFoundError" → Ensure `source .venv/bin/activate`
- "Overpass timeout" → Graph loader falls back to Wilmington city center
- Test failures → Check Python 3.13+ and all dependencies installed

---

## 📄 License

Educational project for traffic routing optimization.

---

**Last Updated**: June 4, 2026  
**Status**: ✅ Phase 1-3 Complete | All Tests Passing | Production Ready (MVP)