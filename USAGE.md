# Usage Guide - ML-Driven Traffic Routing Engine

This guide covers how to use the routing engine for finding optimal traffic-aware routes.

## Quick Start

### 1. Initialize the System (First Time Only)

Load and prepare the road network:

```bash
source .venv/bin/activate
python src/graph_loader.py
```

This creates:
- `data/delaware_graph.pkl` - Road network graph (2278 nodes, 5220 edges)
- `data/segment_features.csv` - Edge feature matrix

### 2. Find an Optimal Route

Basic usage (defaults: noon, Monday, June, no weather):

```bash
python src/main.py --source 39.745,-75.546 --dest 39.758,-75.532
```

### 3. Advanced: With Traffic Conditions

```bash
# Morning commute
python src/main.py --source 39.745,-75.546 --dest 39.758,-75.532 \
                   --hour 8 --day 0 --month 6

# Evening rush in rain
python src/main.py --source 39.745,-75.546 --dest 39.758,-75.532 \
                   --hour 18 --day 4 --month 6 --raining 1

# Weekend with festival
python src/main.py --source 39.745,-75.546 --dest 39.758,-75.532 \
                   --hour 14 --day 5 --festival 1
```

## API Reference

### Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--source` | str | required | Start coordinates (lat,lon) |
| `--dest` | str | required | End coordinates (lat,lon) |
| `--hour` | int | 12 | Hour of day (0-23) |
| `--day` | int | 0 | Day of week (0=Monday, 6=Sunday) |
| `--month` | int | 6 | Month (1-12) |
| `--raining` | int | 0 | Weather (0=dry, 1=rain) |
| `--festival` | int | 0 | Event (0=normal, 1=festival) |
| `--json` | flag | false | Output as JSON |

### Output Format

**Human-readable (default):**
```
Source:           39.7450, -75.5460
Destination:      39.7580, -75.5320
Distance:         2.14 km
Estimated time:   15.9 minutes (956s)
Number of segments: 26
Conditions:
  Hour: 08:00
  Day: Monday
  Raining: No
  Festival: No
```

**JSON output (with --json flag):**
```json
{
  "status": "success",
  "route": {
    "waypoints": [{"lat": 39.744, "lon": -75.546}, ...],
    "distance_km": 2.14,
    "estimated_time_minutes": 15.9,
    "estimated_time_seconds": 956,
    "segments": 26
  }
}
```

## Examples

### Example 1: Daily Commute Comparison

Compare morning vs. evening commute times:

```bash
# Morning
python src/main.py --source 39.745,-75.546 --dest 39.758,-75.532 --hour 8

# Evening
python src/main.py --source 39.745,-75.546 --dest 39.758,-75.532 --hour 18
```

### Example 2: Weather Impact Analysis

See how rain affects travel time:

```bash
# Dry conditions
python src/main.py --source 39.745,-75.546 --dest 39.758,-75.532 --hour 14

# Rainy conditions
python src/main.py --source 39.745,-75.546 --dest 39.758,-75.532 --hour 14 --raining 1
```

### Example 3: Machine Readable Output

Get JSON for integration with other systems:

```bash
python src/main.py --source 39.745,-75.546 --dest 39.758,-75.532 \
                   --hour 18 --raining 1 --json > route.json
```

## Python API

Use the routing engine programmatically:

```python
from src.router import OptimalRouter

# Initialize
router = OptimalRouter()

# Find route
result = router.find_optimal_route(
    start_lat=39.745,
    start_lon=-75.546,
    end_lat=39.758,
    end_lon=-75.532,
    departure_hour=9,
    departure_day=0,
    departure_month=6,
    is_raining=0,
    is_festival_zone=0
)

# Access results
print(f"Distance: {result['distance_km']:.2f} km")
print(f"Time: {result['total_time_minutes']:.1f} minutes")
print(f"Route: {result['path']}")  # List of (lat, lon) tuples
```

## Testing

Run all tests:

```bash
source .venv/bin/activate
python -m pytest tests/ -v
```

Individual test suites:
```bash
pytest tests/test_graph_loader.py -v   # Graph loading tests
pytest tests/test_router.py -v         # Routing and ML tests
```

## Architecture

### System Layers

1. **Phase 1 - Spatial Layer (graph_loader.py)**
   - Loads OpenStreetMap data
   - Constructs NetworkX graph
   - Computes base travel times

2. **Phase 2 - Predictive Layer (traffic_predictor.py)**
   - ML model (XGBoost)
   - Predicts traffic conditions
   - Adjusts edge weights

3. **Phase 3 - Prescriptive Layer (router.py)**
   - Dijkstra's algorithm
   - Dynamic pathfinding
   - Optimal route selection

## Performance

- **Graph loading**: ~8 seconds (Overpass API download)
- **Route finding**: ~0.7 seconds per query (Dijkstra on 2k+ node graph)
- **ML prediction**: ~2ms per edge

### Tested on:
- macOS 13.x
- Python 3.13
- 2278 road network nodes
- 5220 road segments

## Troubleshooting

### "Graph not found" Error

**Solution:** Run `python src/graph_loader.py` first

### Overpass API Timeout

**Solution:** Graph loader falls back to Wilmington city center. For full Delaware: edit `config.py` and adjust `SEARCH_RADIUS`

### Route Not Found

**Possible causes:**
- Start/end points too far apart
- Disconnected network regions
- Try different coordinates closer to Wilmington

## Contributing

Improvements welcome:
1. Add support for turn restrictions
2. Implement A* with heuristics
3. Add real-time traffic data integration
4. Support for multi-stop routing

## References

- **OSMnx**: Graph from OpenStreetMap
- **NetworkX**: Graph algorithms
- **XGBoost**: Machine learning predictions
- **Dijkstra's Algorithm**: https://en.wikipedia.org/wiki/Dijkstra%27s_algorithm
