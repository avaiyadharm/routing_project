#!/usr/bin/env python3
"""
=======================================================================
 LIVE PRODUCTION VEHICLE ROUTING OPTIMIZER
 Google Maps API + XGBoost ML Overlay + Google OR-Tools Solver
=======================================================================

 Replaces the legacy local OSRM-based routing pipeline with a live,
 production-grade system that uses:

   1. Google Geocoding API   — convert address strings → lat/lng
   2. Google Distance Matrix  — real-time N×N travel time matrix
      API                      with departure_time & traffic_model
   3. XGBoost ML Model        — predictive buffer overlay for
                                weather/event anomalies
   4. Google OR-Tools          — prescriptive TSP/VRP solver for
                                optimal stop sequencing

 Author:  Routing Engine Team
 Version: 2.0.0 (Google Maps Live)
 Date:    June 2026
=======================================================================
"""

import os
import sys
import pickle
import math
import logging
from datetime import datetime, timedelta
from typing import List, Tuple, Dict, Optional
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
#  Logging Configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("route_optimizer_live")

# ---------------------------------------------------------------------------
#  Constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent
MODEL_PATH = PROJECT_ROOT / "traffic_xgb_model.pkl"

# Feature layout the XGBoost model was trained on
ML_FEATURE_NAMES = [
    "hour_of_day",
    "day_of_week",
    "month",
    "is_raining",
    "is_festival_zone",
]

# Day-of-week mapping (Python weekday convention: 0=Monday … 6=Sunday)
DAY_NAME_TO_INT = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}

# Google Distance Matrix API limits per request
DISTANCE_MATRIX_MAX_ELEMENTS = 25  # origins × destinations ≤ 25


# ======================================================================
#  STAGE 1 — INTERACTIVE CLI INPUT COLLECTION
# ======================================================================

def collect_user_inputs() -> Dict:
    """
    Prompt the user for all required routing parameters via stdin.

    Returns:
        dict with keys:
            source        (str)  — source address string
            destination   (str)  — destination address string
            waypoints     (list) — list of intermediate stop strings
            target_hour   (int)  — hour component of departure time (0-23)
            target_minute (int)  — minute component (0-59)
            day_of_week   (int)  — 0=Monday … 6=Sunday
            day_name      (str)  — e.g. "Monday"
            is_raining    (int)  — 0 or 1
            is_festival   (int)  — 0 or 1
    """
    print("\n" + "=" * 65)
    print("  🚗  LIVE PRODUCTION ROUTE OPTIMIZER  —  Google Maps + ML + OR-Tools")
    print("=" * 65)
    print()

    # --- Source ---
    source = _prompt_non_empty("📍 Enter SOURCE location (e.g. 'Times Square, NY'): ")

    # --- Destination ---
    destination = _prompt_non_empty(
        "📍 Enter DESTINATION location (e.g. 'JFK Airport, NY'): "
    )

    # --- Waypoints ---
    print(
        "📍 Enter WAYPOINT locations separated by semicolons (;).\n"
        "   Example: Brooklyn Bridge, NY; Central Park, NY\n"
        "   Leave blank for direct routing (source → destination only)."
    )
    raw_waypoints = input("   Waypoints: ").strip()
    waypoints: List[str] = []
    if raw_waypoints:
        waypoints = [w.strip() for w in raw_waypoints.split(";") if w.strip()]

    # --- Target time ---
    target_hour, target_minute = _prompt_time(
        "🕒 Enter TARGET departure time (HH:MM, 24-hr format, e.g. 08:30): "
    )

    # --- Day of week ---
    day_of_week, day_name = _prompt_day(
        "📅 Enter TARGET day of the week (e.g. Monday, Saturday): "
    )

    # --- Weather flag ---
    is_raining = _prompt_yes_no(
        "🌧️  Is there active rain / severe weather? (yes/no): "
    )

    # --- Festival / event flag ---
    is_festival = _prompt_yes_no(
        "🎪 Is there an active event / festival near the route? (yes/no): "
    )

    inputs = {
        "source": source,
        "destination": destination,
        "waypoints": waypoints,
        "target_hour": target_hour,
        "target_minute": target_minute,
        "day_of_week": day_of_week,
        "day_name": day_name,
        "is_raining": is_raining,
        "is_festival": is_festival,
    }

    # Echo back
    print("\n" + "-" * 50)
    print("  ✅  INPUTS CONFIRMED")
    print("-" * 50)
    print(f"  Source       : {source}")
    print(f"  Destination  : {destination}")
    print(f"  Waypoints    : {waypoints if waypoints else '(none — direct route)'}")
    print(f"  Departure    : {target_hour:02d}:{target_minute:02d} on {day_name}")
    print(f"  Raining      : {'Yes' if is_raining else 'No'}")
    print(f"  Festival     : {'Yes' if is_festival else 'No'}")
    print("-" * 50 + "\n")

    return inputs


# ---- Input helpers ----

def _prompt_non_empty(prompt: str) -> str:
    """Prompt until a non-empty string is entered."""
    while True:
        value = input(prompt).strip()
        if value:
            return value
        print("   ⚠️  This field cannot be empty. Please try again.")


def _prompt_time(prompt: str) -> Tuple[int, int]:
    """Prompt for HH:MM and return (hour, minute)."""
    while True:
        raw = input(prompt).strip()
        try:
            parts = raw.split(":")
            if len(parts) != 2:
                raise ValueError
            hour, minute = int(parts[0]), int(parts[1])
            if not (0 <= hour <= 23 and 0 <= minute <= 59):
                raise ValueError
            return hour, minute
        except ValueError:
            print("   ⚠️  Invalid format. Please enter time as HH:MM (e.g. 08:30).")


def _prompt_day(prompt: str) -> Tuple[int, str]:
    """Prompt for day name and return (int_code, canonical_name)."""
    while True:
        raw = input(prompt).strip().lower()
        if raw in DAY_NAME_TO_INT:
            canonical = raw.capitalize()
            return DAY_NAME_TO_INT[raw], canonical
        print(
            "   ⚠️  Invalid day. Please enter one of: "
            + ", ".join(d.capitalize() for d in DAY_NAME_TO_INT)
        )


def _prompt_yes_no(prompt: str) -> int:
    """Prompt for yes/no and return 1 or 0."""
    while True:
        raw = input(prompt).strip().lower()
        if raw in ("yes", "y", "1"):
            return 1
        if raw in ("no", "n", "0"):
            return 0
        print("   ⚠️  Please enter 'yes' or 'no'.")


# ======================================================================
#  STAGE 2 — GOOGLE MAPS API INTEGRATION LAYER
# ======================================================================

def _get_gmaps_client():
    """
    Instantiate the Google Maps client using the API key from
    the environment variable GOOGLE_MAPS_API_KEY.

    Raises:
        SystemExit: if the key is not set.
    """
    try:
        import googlemaps
    except ImportError:
        logger.error(
            "❌ 'googlemaps' package not installed. "
            "Run:  pip install googlemaps"
        )
        sys.exit(1)

    api_key = os.environ.get("GOOGLE_MAPS_API_KEY")
    if not api_key:
        logger.error(
            "❌ GOOGLE_MAPS_API_KEY environment variable is not set.\n"
            "   Export it before running:\n"
            "     export GOOGLE_MAPS_API_KEY='your-key-here'"
        )
        sys.exit(1)

    return googlemaps.Client(key=api_key, timeout=30)


def geocode_address(
    gmaps, address: str
) -> Tuple[float, float, str]:
    """
    Geocode a human-readable address string into (lat, lng, formatted_name).

    Args:
        gmaps:   googlemaps.Client instance
        address: free-form location string (e.g. "Times Square, NY")

    Returns:
        (latitude, longitude, google_formatted_address)

    Raises:
        ValueError: if the address cannot be geocoded
    """
    try:
        results = gmaps.geocode(address)
    except Exception as exc:
        raise ValueError(
            f"Geocoding API error for '{address}': {exc}"
        ) from exc

    if not results:
        raise ValueError(
            f"❌ Could not geocode address: '{address}'. "
            "Please check the spelling or provide a more specific location."
        )

    loc = results[0]["geometry"]["location"]
    formatted = results[0].get("formatted_address", address)
    return loc["lat"], loc["lng"], formatted


def geocode_all_locations(
    gmaps, source: str, destination: str, waypoints: List[str]
) -> List[Dict]:
    """
    Geocode source, destination, and all waypoints.

    Returns:
        List of dicts: [
            {"label": str, "address": str, "lat": float, "lng": float},
            ...
        ]
        Index 0 is always the source (depot).
        Index -1 is always the destination.
        Intermediate indices are waypoints in input order.
    """
    all_addresses = [source] + waypoints + [destination]
    labels = (
        ["Source"]
        + [f"Waypoint {i+1}" for i in range(len(waypoints))]
        + ["Destination"]
    )

    locations = []
    print("\n🌐 STAGE 2a — Geocoding Addresses via Google Geocoding API")
    print("-" * 60)

    for label, addr in zip(labels, all_addresses):
        lat, lng, formatted = geocode_address(gmaps, addr)
        locations.append(
            {
                "label": label,
                "address": formatted,
                "lat": lat,
                "lng": lng,
            }
        )
        print(f"  ✅ {label:12s} → {formatted}")
        print(f"{'':16s}   ({lat:.6f}, {lng:.6f})")

    print("-" * 60)
    return locations


def _compute_next_departure(
    target_hour: int, target_minute: int, target_day_int: int
) -> datetime:
    """
    Compute the next future datetime matching the user's
    target day-of-week and time.  Google Distance Matrix API
    requires departure_time to be in the future.

    Returns:
        datetime (UTC-naive) of next matching departure.
    """
    now = datetime.utcnow()
    # Build candidate for this week
    days_ahead = target_day_int - now.weekday()
    if days_ahead < 0:
        days_ahead += 7

    candidate = now.replace(
        hour=target_hour, minute=target_minute, second=0, microsecond=0
    ) + timedelta(days=days_ahead)

    # If the candidate is in the past, push to next week
    if candidate <= now:
        candidate += timedelta(weeks=1)

    return candidate


def build_travel_time_matrix(
    gmaps,
    locations: List[Dict],
    departure_time: datetime,
    traffic_model: str = "best_guess",
) -> List[List[int]]:
    """
    Build an N×N travel-time matrix (seconds) via the Google
    Distance Matrix API with live traffic data.

    The API is called with departure_time and traffic_model to
    obtain duration_in_traffic values that incorporate real-time
    and predicted historical congestion.

    Args:
        gmaps:          googlemaps.Client instance
        locations:      list of location dicts (from geocode_all_locations)
        departure_time: future datetime for traffic estimation
        traffic_model:  'best_guess' | 'pessimistic' | 'optimistic'

    Returns:
        N×N matrix of travel durations in seconds (int).
    """
    n = len(locations)
    matrix = [[0] * n for _ in range(n)]
    coords = [(loc["lat"], loc["lng"]) for loc in locations]

    print(f"\n🗺️  STAGE 2b — Building {n}×{n} Travel Time Matrix via Distance Matrix API")
    print(f"   departure_time  = {departure_time.isoformat()}")
    print(f"   traffic_model   = {traffic_model}")
    print("-" * 60)

    # The Distance Matrix API constrains origins × destinations ≤ 25.
    # For small N (≤5), we can do one call.  For larger N, batch.
    for i in range(n):
        # Build one row at a time (1 origin × N destinations)
        origin = coords[i]
        destinations = coords

        try:
            result = gmaps.distance_matrix(
                origins=[origin],
                destinations=destinations,
                mode="driving",
                departure_time=departure_time,
                traffic_model=traffic_model,
            )
        except Exception as exc:
            logger.error(f"❌ Distance Matrix API error for row {i}: {exc}")
            raise

        row_elements = result["rows"][0]["elements"]
        for j, element in enumerate(row_elements):
            if element["status"] != "OK":
                logger.warning(
                    f"   ⚠️  No route [{i}→{j}] "
                    f"({locations[i]['label']} → {locations[j]['label']}): "
                    f"{element['status']}"
                )
                # Use a high penalty so OR-Tools avoids this arc
                matrix[i][j] = 999999
            else:
                # Prefer duration_in_traffic (includes live congestion);
                # fall back to plain duration if not available.
                if "duration_in_traffic" in element:
                    matrix[i][j] = element["duration_in_traffic"]["value"]
                else:
                    matrix[i][j] = element["duration"]["value"]

        # Ensure diagonal is 0
        matrix[i][i] = 0

    # Pretty-print the raw matrix
    _print_matrix("RAW Google Distance Matrix (seconds)", locations, matrix)

    return matrix


def _print_matrix(
    title: str, locations: List[Dict], matrix: List[List[int]]
) -> None:
    """Pretty-print a matrix with location labels."""
    n = len(matrix)
    # Build short labels
    short = [loc["label"][:8] for loc in locations]

    print(f"\n  📊 {title}")
    header = "         " + "".join(f"{s:>10s}" for s in short)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for i in range(n):
        row_str = "".join(f"{matrix[i][j]:>10d}" for j in range(n))
        print(f"  {short[i]:>7s} |{row_str}")
    print()


# ======================================================================
#  STAGE 3 — PREDICTIVE ML BUFFER OVERLAY  (XGBoost)
# ======================================================================

def load_ml_model(model_path: Path = MODEL_PATH):
    """
    Load the pre-trained XGBoost model from disk.

    Returns:
        The deserialized model, or None if the file does not exist
        (graceful degradation — the pipeline will use the raw
        Google matrix without ML adjustment).
    """
    if not model_path.exists():
        logger.warning(
            f"⚠️  ML model not found at {model_path}. "
            "Proceeding WITHOUT predictive adjustment."
        )
        return None

    logger.info(f"💾 Loading XGBoost model from {model_path} …")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    logger.info("✅ ML model loaded successfully.")
    return model


def apply_ml_overlay(
    raw_matrix: List[List[int]],
    model,
    hour: int,
    day_of_week: int,
    month: int,
    is_raining: int,
    is_festival: int,
    locations: List[Dict],
) -> List[List[int]]:
    """
    Apply the XGBoost predictive buffer to the raw travel-time matrix.

    Strategy:
        1. Predict a trip duration under the user's ACTUAL conditions
           (hour, day, weather, festival).
        2. Predict a trip duration under a NEUTRAL baseline
           (12:00, Wednesday, no rain, no festival).
        3. Compute scaling_factor = actual_prediction / baseline_prediction.
        4. Multiply each cell of the raw matrix by this factor.
        5. Floor-clamp: adjusted value ≥ 60% of raw value (safety net).

    When no anomaly conditions are active (no rain AND no festival),
    the ML overlay is skipped to avoid unnecessary perturbation of
    the already traffic-aware Google data.

    Args:
        raw_matrix:   N×N travel time matrix from Google (seconds)
        model:        trained XGBoost model (or None to skip)
        hour:         departure hour (0-23)
        day_of_week:  0=Monday … 6=Sunday
        month:        1-12
        is_raining:   0 or 1
        is_festival:  0 or 1
        locations:    location dicts for labelling

    Returns:
        N×N ML-adjusted matrix (seconds, int).
    """
    n = len(raw_matrix)

    print("\n🧠 STAGE 3 — Predictive ML Buffer Overlay (XGBoost)")
    print("-" * 60)

    # If no model or no anomaly conditions, pass through raw matrix
    if model is None:
        print("   ⏭️  No ML model loaded — using raw Google matrix as-is.")
        return [row[:] for row in raw_matrix]

    if is_raining == 0 and is_festival == 0:
        print(
            "   ⏭️  No anomaly conditions active (no rain, no festival).\n"
            "       Google's traffic_model data is already sufficient.\n"
            "       Passing raw matrix through unchanged."
        )
        return [row[:] for row in raw_matrix]

    # ---- Compute scaling factor ----
    # Actual conditions
    actual_features = np.array(
        [[hour, day_of_week, month, is_raining, is_festival]], dtype=float
    )
    actual_prediction = float(model.predict(actual_features)[0])

    # Neutral baseline: noon, Wednesday, clear, no festival
    baseline_features = np.array([[12, 2, month, 0, 0]], dtype=float)
    baseline_prediction = float(model.predict(baseline_features)[0])

    if baseline_prediction <= 0:
        logger.warning("   ⚠️  Baseline prediction ≤ 0 — skipping ML overlay.")
        return [row[:] for row in raw_matrix]

    scaling_factor = actual_prediction / baseline_prediction

    print(f"   ML Prediction (actual conditions) : {actual_prediction:,.0f} s")
    print(f"   ML Prediction (neutral baseline)  : {baseline_prediction:,.0f} s")
    print(f"   Derived Scaling Factor            : {scaling_factor:.4f}")
    print(f"   Features → {ML_FEATURE_NAMES}")
    print(f"   Values   → [{hour}, {day_of_week}, {month}, {is_raining}, {is_festival}]")

    # ---- Apply scaling to matrix ----
    adjusted = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                adjusted[i][j] = 0
                continue
            raw_val = raw_matrix[i][j]
            scaled = raw_val * scaling_factor
            # Floor-clamp: never reduce below 60% of Google's value
            floor = raw_val * 0.60
            adjusted[i][j] = max(int(round(scaled)), int(round(floor)))

    _print_matrix("ML-ADJUSTED Travel Time Matrix (seconds)", locations, adjusted)

    # Show delta summary
    total_raw = sum(raw_matrix[i][j] for i in range(n) for j in range(n) if i != j)
    total_adj = sum(adjusted[i][j] for i in range(n) for j in range(n) if i != j)
    delta_pct = ((total_adj - total_raw) / total_raw * 100) if total_raw else 0
    print(
        f"   Σ raw = {total_raw:,}s  →  Σ adjusted = {total_adj:,}s  "
        f"(Δ {delta_pct:+.1f}%)\n"
    )

    return adjusted


# ======================================================================
#  STAGE 4 — PRESCRIPTIVE SOLVER  (Google OR-Tools)
# ======================================================================

def solve_optimal_route(
    matrix: List[List[int]], locations: List[Dict]
) -> Optional[Dict]:
    """
    Solve the Traveling Salesman Problem (TSP) using Google OR-Tools.

    The vehicle starts and ends at the depot (index 0 = source).
    All intermediate nodes (waypoints + destination) must be visited
    exactly once.

    Args:
        matrix:    N×N travel-time matrix (seconds)
        locations: list of location dicts (index 0 = depot)

    Returns:
        dict with keys:
            sequence       — list of node indices in visit order
            sequence_names — list of location labels
            total_seconds  — total travel time for the route
            legs           — list of (from_label, to_label, seconds)
        or None if no solution found.
    """
    from ortools.constraint_solver import routing_enums_pb2, pywrapcp

    n = len(matrix)

    print("\n⚙️  STAGE 4 — Prescriptive Solver (Google OR-Tools TSP/VRP)")
    print("-" * 60)
    print(f"   Nodes       : {n}")
    print(f"   Depot       : 0 ({locations[0]['label']})")
    print(f"   Vehicles    : 1")

    # Create the routing index manager (single vehicle, depot=0)
    manager = pywrapcp.RoutingIndexManager(n, 1, 0)  # nodes, vehicles, depot
    routing = pywrapcp.RoutingModel(manager)

    # Register the transit callback
    def time_callback(from_index: int, to_index: int) -> int:
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return matrix[from_node][to_node]

    transit_cb_index = routing.RegisterTransitCallback(time_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_cb_index)

    # Search parameters
    search_params = pywrapcp.DefaultRoutingSearchParameters()
    search_params.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    search_params.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    search_params.time_limit.FromSeconds(5)

    print("   Strategy    : PATH_CHEAPEST_ARC + GUIDED_LOCAL_SEARCH (5 s limit)")
    print("   Solving …")

    solution = routing.SolveWithParameters(search_params)

    if not solution:
        logger.error("❌ OR-Tools could not find a feasible solution.")
        return None

    # Extract solution
    sequence = []
    index = routing.Start(0)
    while not routing.IsEnd(index):
        sequence.append(manager.IndexToNode(index))
        index = solution.Value(routing.NextVar(index))
    sequence.append(manager.IndexToNode(index))  # return to depot

    total_seconds = solution.ObjectiveValue()

    # Build legs
    legs = []
    for k in range(len(sequence) - 1):
        i, j = sequence[k], sequence[k + 1]
        legs.append(
            (locations[i]["label"], locations[j]["label"], matrix[i][j])
        )

    sequence_names = [locations[idx]["label"] for idx in sequence]

    result = {
        "sequence": sequence,
        "sequence_names": sequence_names,
        "total_seconds": total_seconds,
        "legs": legs,
    }

    print(f"   ✅ Solution found!")
    print(f"   Objective   : {total_seconds:,} seconds ({total_seconds / 60:.1f} min)")
    print("-" * 60)

    return result


# ======================================================================
#  STAGE 5 — FORMATTED RESULTS PRESENTATION
# ======================================================================

def present_results(
    locations: List[Dict],
    raw_matrix: List[List[int]],
    adjusted_matrix: List[List[int]],
    solution: Dict,
    user_inputs: Dict,
) -> None:
    """Print a comprehensive summary of the optimization."""

    print("\n" + "=" * 65)
    print("  🎉  OPTIMIZED ROUTE — FINAL RESULTS")
    print("=" * 65)

    # ---- Locations ----
    print("\n  📍 GEOCODED LOCATIONS:")
    for loc in locations:
        print(f"     {loc['label']:12s}  {loc['address']}")
        print(f"{'':17s}  ({loc['lat']:.6f}, {loc['lng']:.6f})")

    # ---- Optimal Sequence ----
    print("\n  🛣️  OPTIMAL VISIT SEQUENCE:")
    route_str = " ➔ ".join(solution["sequence_names"])
    print(f"     {route_str}")

    # ---- Per-Leg Breakdown ----
    print("\n  📋 LEG-BY-LEG BREAKDOWN:")
    print(f"     {'From':<15s} {'To':<15s} {'Duration':>12s}")
    print("     " + "-" * 44)
    for from_label, to_label, secs in solution["legs"]:
        mins = secs / 60
        print(f"     {from_label:<15s} {to_label:<15s} {secs:>7,d} s  ({mins:.1f} min)")

    # ---- Total Duration ----
    total_s = solution["total_seconds"]
    total_m = total_s / 60
    print(f"\n  ⏱️  TOTAL ESTIMATED DRIVING DURATION:")
    print(f"     {total_s:,} seconds  =  {total_m:.1f} minutes")

    # ---- Conditions Summary ----
    print(f"\n  🔧 CONDITIONS APPLIED:")
    print(f"     Departure : {user_inputs['target_hour']:02d}:{user_inputs['target_minute']:02d} on {user_inputs['day_name']}")
    print(f"     Raining   : {'Yes ☔' if user_inputs['is_raining'] else 'No ☀️'}")
    print(f"     Festival  : {'Yes 🎪' if user_inputs['is_festival'] else 'No'}")

    # ---- Matrix Comparison (compact) ----
    n = len(raw_matrix)
    raw_total = sum(raw_matrix[i][j] for i in range(n) for j in range(n) if i != j)
    adj_total = sum(adjusted_matrix[i][j] for i in range(n) for j in range(n) if i != j)
    delta_pct = ((adj_total - raw_total) / raw_total * 100) if raw_total else 0

    print(f"\n  📊 MATRIX SUMMARY:")
    print(f"     Raw Google Σ      : {raw_total:>10,} s")
    print(f"     ML-Adjusted Σ     : {adj_total:>10,} s  (Δ {delta_pct:+.1f}%)")

    print("\n" + "=" * 65)
    print("  ✅  Optimization complete.")
    print("=" * 65 + "\n")


# ======================================================================
#  MAIN PIPELINE
# ======================================================================

def main() -> None:
    """
    End-to-end pipeline:
      1. Collect user inputs
      2. Geocode addresses & build travel-time matrix via Google Maps
      3. Apply XGBoost ML buffer overlay
      4. Solve TSP/VRP via OR-Tools
      5. Present results
    """
    # ── Stage 1: Collect Inputs ──
    user_inputs = collect_user_inputs()

    # ── Stage 2: Google Maps API ──
    gmaps = _get_gmaps_client()

    # 2a. Geocode
    try:
        locations = geocode_all_locations(
            gmaps,
            user_inputs["source"],
            user_inputs["destination"],
            user_inputs["waypoints"],
        )
    except ValueError as exc:
        logger.error(f"❌ Geocoding failed: {exc}")
        sys.exit(1)

    # Compute future departure_time
    departure_dt = _compute_next_departure(
        user_inputs["target_hour"],
        user_inputs["target_minute"],
        user_inputs["day_of_week"],
    )

    # 2b. Distance Matrix
    try:
        raw_matrix = build_travel_time_matrix(
            gmaps, locations, departure_dt, traffic_model="best_guess"
        )
    except Exception as exc:
        logger.error(f"❌ Distance Matrix API failed: {exc}")
        sys.exit(1)

    # ── Stage 3: ML Overlay ──
    now = datetime.now()
    current_month = now.month

    model = load_ml_model()
    adjusted_matrix = apply_ml_overlay(
        raw_matrix=raw_matrix,
        model=model,
        hour=user_inputs["target_hour"],
        day_of_week=user_inputs["day_of_week"],
        month=current_month,
        is_raining=user_inputs["is_raining"],
        is_festival=user_inputs["is_festival"],
        locations=locations,
    )

    # ── Stage 4: OR-Tools Solver ──
    solution = solve_optimal_route(adjusted_matrix, locations)

    if solution is None:
        logger.error(
            "❌ Could not find a feasible route. "
            "Check that all locations are drivable."
        )
        sys.exit(1)

    # ── Stage 5: Present Results ──
    present_results(
        locations=locations,
        raw_matrix=raw_matrix,
        adjusted_matrix=adjusted_matrix,
        solution=solution,
        user_inputs=user_inputs,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⛔ Interrupted by user.")
        sys.exit(130)
