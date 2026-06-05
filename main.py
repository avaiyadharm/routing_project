#!/usr/bin/env python3
"""
==========================================================================
 VEHICLE ROUTING OPTIMIZER — FastAPI REST API BACKEND
 Google Maps API  ·  XGBoost ML Overlay  ·  Google OR-Tools Solver
==========================================================================

 Production-grade REST API that exposes the full routing optimization
 pipeline as HTTP endpoints with Pydantic validation, structured JSON
 responses, and enterprise-grade error handling.

 Endpoints:
   GET  /              → Health check
   POST /api/v1/optimize-route → Full optimization pipeline

 Run:
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload

 Swagger UI:
   http://localhost:8000/docs

 Author:  Routing Engine Team
 Version: 3.0.0 (FastAPI Production)
 Date:    June 2026
==========================================================================
"""

import os
import pickle
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
#  Logging Configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("routing_api")

# ---------------------------------------------------------------------------
#  Constants & Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent
MODEL_PATH = PROJECT_ROOT / "traffic_xgb_model.pkl"

ML_FEATURE_NAMES = [
    "hour_of_day",
    "day_of_week",
    "month",
    "is_raining",
    "is_festival_zone",
]

# Module-level reference to the ML model (loaded once at startup)
_ml_model = None


# ==========================================================================
#  PYDANTIC REQUEST / RESPONSE SCHEMAS
# ==========================================================================

class LocationQuery(BaseModel):
    """
    Incoming POST payload for route optimization.

    Locations can be provided as:
      - A human-readable address string  (e.g. ``"Times Square, NY"``)
      - A ``[lat, lon]`` coordinate pair  (e.g. ``[40.758, -73.985]``)

    All temporal and contextual fields are validated within their
    physical bounds so that downstream consumers never receive
    nonsensical inputs.
    """

    source: Union[str, List[float]] = Field(
        ...,
        description=(
            "Source / depot location. "
            "Either an address string or [lat, lon] coordinates."
        ),
        examples=["Times Square, NY", [40.758896, -73.985130]],
    )
    destination: Union[str, List[float]] = Field(
        ...,
        description=(
            "Final destination location. "
            "Either an address string or [lat, lon] coordinates."
        ),
        examples=["JFK Airport, NY", [40.641766, -73.780968]],
    )
    waypoints: List[Union[str, List[float]]] = Field(
        default_factory=list,
        description=(
            "Intermediate delivery stops. "
            "Each element is an address string or [lat, lon] pair."
        ),
        examples=[["Brooklyn Bridge, NY", "Central Park, NY"]],
    )
    departure_hour: int = Field(
        default=12,
        ge=0,
        le=23,
        description="Hour of departure (0–23).",
    )
    day_of_week: int = Field(
        default=0,
        ge=0,
        le=6,
        description="Day of week: 0 = Monday … 6 = Sunday.",
    )
    month: int = Field(
        default=6,
        ge=1,
        le=12,
        description="Month of the year (1–12).",
    )
    is_raining: int = Field(
        default=0,
        ge=0,
        le=1,
        description="Rain / severe weather flag (0 = no, 1 = yes).",
    )
    is_festival_zone: int = Field(
        default=0,
        ge=0,
        le=1,
        description="Active event / festival near route (0 = no, 1 = yes).",
    )

    @field_validator("source", "destination")
    @classmethod
    def validate_location(cls, v: Union[str, List[float]]) -> Union[str, List[float]]:
        """Ensure location is a non-empty string or a valid [lat, lon] pair."""
        if isinstance(v, str):
            if not v.strip():
                raise ValueError("Location string must not be empty.")
            return v.strip()
        if isinstance(v, list):
            if len(v) != 2:
                raise ValueError("Coordinate pair must contain exactly 2 elements: [lat, lon].")
            if not (-90 <= v[0] <= 90):
                raise ValueError(f"Latitude {v[0]} out of range [-90, 90].")
            if not (-180 <= v[1] <= 180):
                raise ValueError(f"Longitude {v[1]} out of range [-180, 180].")
            return v
        raise ValueError("Location must be a string or [lat, lon] list.")

    @field_validator("waypoints")
    @classmethod
    def validate_waypoints(
        cls, v: List[Union[str, List[float]]]
    ) -> List[Union[str, List[float]]]:
        """Validate each waypoint entry."""
        validated = []
        for idx, wp in enumerate(v):
            if isinstance(wp, str):
                if not wp.strip():
                    raise ValueError(f"Waypoint [{idx}] string must not be empty.")
                validated.append(wp.strip())
            elif isinstance(wp, list):
                if len(wp) != 2:
                    raise ValueError(
                        f"Waypoint [{idx}] coordinate pair must have exactly 2 elements."
                    )
                if not (-90 <= wp[0] <= 90):
                    raise ValueError(
                        f"Waypoint [{idx}] latitude {wp[0]} out of range."
                    )
                if not (-180 <= wp[1] <= 180):
                    raise ValueError(
                        f"Waypoint [{idx}] longitude {wp[1]} out of range."
                    )
                validated.append(wp)
            else:
                raise ValueError(
                    f"Waypoint [{idx}] must be a string or [lat, lon] list."
                )
        return validated


class LegDetail(BaseModel):
    """A single leg within the optimized route."""

    from_location: str = Field(..., description="Starting node label for this leg.")
    to_location: str = Field(..., description="Ending node label for this leg.")
    duration_seconds: int = Field(..., description="Travel time for this leg (seconds).")


class ContextApplied(BaseModel):
    """Echo of the weather/temporal/event context that was applied."""

    departure_hour: int
    day_of_week: int
    month: int
    is_raining: bool
    is_festival_zone: bool
    ml_scaling_factor: Optional[float] = Field(
        None,
        description=(
            "The ML-derived scaling factor applied to the travel matrix. "
            "None if ML overlay was skipped."
        ),
    )


class OptimizeRouteResponse(BaseModel):
    """Structured JSON response from the optimization endpoint."""

    status: str = Field(default="success", description="Operation status.")
    optimal_sequence_order: List[str] = Field(
        ...,
        description=(
            "Ordered list of location labels representing the optimal visit "
            "sequence (e.g. Source → Waypoint 2 → Waypoint 1 → Destination → Source)."
        ),
    )
    total_optimized_duration_seconds: int = Field(
        ..., description="Total travel time for the optimized route (seconds)."
    )
    total_distance_meters: int = Field(
        ..., description="Total physical distance covered (meters)."
    )
    legs: List[LegDetail] = Field(
        ..., description="Per-leg breakdown of the optimized route."
    )
    context_applied: ContextApplied = Field(
        ..., description="Echo of the contextual parameters processed."
    )
    geocoded_locations: List[dict] = Field(
        ..., description="All geocoded locations with labels and coordinates."
    )
    raw_matrix_seconds: List[List[int]] = Field(
        ..., description="The raw N×N travel-time matrix from Google (seconds)."
    )
    ml_adjusted_matrix_seconds: List[List[int]] = Field(
        ..., description="The ML-adjusted N×N travel-time matrix (seconds)."
    )


class HealthResponse(BaseModel):
    """Health check response schema."""

    status: str = Field(default="online", description="Server status.")
    service: str = Field(
        default="Vehicle Routing Optimizer API",
        description="Service identifier.",
    )
    version: str = Field(default="3.0.0", description="API version.")
    ml_model_loaded: bool = Field(
        ..., description="Whether the XGBoost traffic model is loaded."
    )
    timestamp: str = Field(..., description="Current server time (ISO 8601).")


# ==========================================================================
#  APPLICATION LIFESPAN  (load ML model once at startup)
# ==========================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load heavy resources on startup, release on shutdown."""
    global _ml_model

    logger.info("🚀 Starting Vehicle Routing Optimizer API v3.0.0 …")

    # Load XGBoost model
    if MODEL_PATH.exists():
        try:
            with open(MODEL_PATH, "rb") as f:
                _ml_model = pickle.load(f)
            logger.info(f"✅ XGBoost model loaded from {MODEL_PATH}")
        except Exception as exc:
            logger.warning(f"⚠️  Failed to load ML model: {exc} — continuing without it")
            _ml_model = None
    else:
        logger.warning(
            f"⚠️  ML model not found at {MODEL_PATH} — "
            "ML overlay will be skipped for all requests"
        )
        _ml_model = None

    yield  # ── Application is running ──

    logger.info("🛑 Shutting down Routing API …")
    _ml_model = None


# ==========================================================================
#  FASTAPI APPLICATION
# ==========================================================================

app = FastAPI(
    title="Vehicle Routing Optimizer API",
    description=(
        "Production-grade REST API for multi-stop vehicle route optimization. "
        "Combines Google Maps live traffic data, an XGBoost ML prediction overlay, "
        "and Google OR-Tools TSP/VRP solver to compute the fastest visit sequence."
    ),
    version="3.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)


# ==========================================================================
#  ENDPOINT: GET / — Health Check
# ==========================================================================

@app.get(
    "/",
    response_model=HealthResponse,
    tags=["System"],
    summary="Health check",
    description="Returns current server status and whether the ML model is loaded.",
)
async def health_check() -> HealthResponse:
    """
    **Health Check Endpoint**

    Returns a JSON object confirming the API server is online,
    including the ML model load status and current server timestamp.
    """
    return HealthResponse(
        status="online",
        service="Vehicle Routing Optimizer API",
        version="3.0.0",
        ml_model_loaded=_ml_model is not None,
        timestamp=datetime.utcnow().isoformat() + "Z",
    )


# ==========================================================================
#  ENDPOINT: POST /api/v1/optimize-route — Full Optimization Pipeline
# ==========================================================================

@app.post(
    "/api/v1/optimize-route",
    response_model=OptimizeRouteResponse,
    tags=["Routing"],
    summary="Optimize a multi-stop vehicle route",
    description=(
        "Accepts a `LocationQuery` payload, geocodes addresses via Google Maps, "
        "builds a live traffic-aware travel-time matrix, applies the XGBoost ML "
        "prediction overlay, and solves the optimal visit sequence via OR-Tools."
    ),
    responses={
        400: {"description": "Invalid address or geocoding failure."},
        500: {"description": "Server configuration error (e.g. missing API key)."},
        502: {"description": "Upstream Google Maps API failure."},
    },
)
async def optimize_route(payload: LocationQuery) -> OptimizeRouteResponse:
    """
    **Full Route Optimization Pipeline**

    Processing stages:
    1. Validate Google Maps API key presence
    2. Geocode all locations (source, destination, waypoints)
    3. Build N×N travel-time matrix via Distance Matrix API (with live traffic)
    4. Apply XGBoost ML buffer overlay for weather/event anomalies
    5. Solve TSP/VRP via Google OR-Tools
    6. Return structured JSON with optimized sequence, durations, and matrices

    Raises:
        HTTPException 400: Invalid or unresolvable address.
        HTTPException 500: Missing API key or ML model load error.
        HTTPException 502: Google Maps API network/timeout failure.
    """
    logger.info(
        f"📥 Incoming optimization request: "
        f"source={payload.source}, dest={payload.destination}, "
        f"waypoints={len(payload.waypoints)}, "
        f"hour={payload.departure_hour}, day={payload.day_of_week}"
    )

    # ── Step 1: Validate API key ──
    gmaps_client = _get_gmaps_client()

    # ── Step 2: Geocode all locations ──
    locations = _geocode_all_locations(gmaps_client, payload)

    # ── Step 3: Build travel-time matrix ──
    departure_dt = _compute_next_departure(
        payload.departure_hour, 0, payload.day_of_week
    )
    raw_matrix, distance_matrix_meters = _build_travel_time_and_distance_matrices(
        gmaps_client, locations, departure_dt
    )

    # ── Step 4: ML overlay ──
    adjusted_matrix, scaling_factor = _apply_ml_overlay(
        raw_matrix=raw_matrix,
        hour=payload.departure_hour,
        day_of_week=payload.day_of_week,
        month=payload.month,
        is_raining=payload.is_raining,
        is_festival=payload.is_festival_zone,
    )

    # ── Step 5: Solve with OR-Tools ──
    solution = _solve_tsp(adjusted_matrix, locations)

    # ── Step 6: Build response ──
    # Compute total distance from the optimized sequence
    total_distance_m = 0
    legs: List[LegDetail] = []
    for k in range(len(solution["sequence"]) - 1):
        i = solution["sequence"][k]
        j = solution["sequence"][k + 1]
        total_distance_m += distance_matrix_meters[i][j]
        legs.append(
            LegDetail(
                from_location=locations[i]["label"],
                to_location=locations[j]["label"],
                duration_seconds=adjusted_matrix[i][j],
            )
        )

    response = OptimizeRouteResponse(
        status="success",
        optimal_sequence_order=solution["sequence_names"],
        total_optimized_duration_seconds=solution["total_seconds"],
        total_distance_meters=total_distance_m,
        legs=legs,
        context_applied=ContextApplied(
            departure_hour=payload.departure_hour,
            day_of_week=payload.day_of_week,
            month=payload.month,
            is_raining=bool(payload.is_raining),
            is_festival_zone=bool(payload.is_festival_zone),
            ml_scaling_factor=scaling_factor,
        ),
        geocoded_locations=[
            {
                "label": loc["label"],
                "address": loc["address"],
                "lat": loc["lat"],
                "lng": loc["lng"],
            }
            for loc in locations
        ],
        raw_matrix_seconds=raw_matrix,
        ml_adjusted_matrix_seconds=adjusted_matrix,
    )

    logger.info(
        f"✅ Optimization complete: "
        f"{solution['total_seconds']}s, "
        f"{total_distance_m}m, "
        f"sequence={'→'.join(solution['sequence_names'])}"
    )

    return response


# ==========================================================================
#  INTERNAL PIPELINE FUNCTIONS
# ==========================================================================

def _get_gmaps_client():
    """
    Instantiate the Google Maps Python client.

    The API key is loaded from the ``GOOGLE_MAPS_API_KEY`` environment
    variable.  If the key is missing, an HTTP 500 is raised immediately
    so the caller receives a clear configuration-error response rather
    than an opaque downstream failure.

    Returns:
        googlemaps.Client instance.

    Raises:
        HTTPException 500: if the API key is not set.
    """
    try:
        import googlemaps
    except ImportError:
        logger.error("googlemaps package not installed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=(
                "Server misconfiguration: 'googlemaps' package is not installed. "
                "Run: pip install googlemaps"
            ),
        )

    api_key = os.environ.get("GOOGLE_MAPS_API_KEY")
    if not api_key:
        logger.error("GOOGLE_MAPS_API_KEY environment variable is not set")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=(
                "Server configuration error: GOOGLE_MAPS_API_KEY environment "
                "variable is not set. Please configure it before starting the API."
            ),
        )

    try:
        return googlemaps.Client(key=api_key, timeout=30)
    except Exception as exc:
        logger.error(f"Failed to initialize Google Maps client: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to initialize Google Maps client: {exc}",
        )


def _geocode_single(gmaps, location: Union[str, List[float]], label: str) -> dict:
    """
    Geocode a single location (address string or coordinate pair).

    If the location is already a ``[lat, lon]`` pair, a reverse-geocode
    is performed to obtain a human-readable formatted address.

    Args:
        gmaps:    googlemaps.Client instance.
        location: Address string or ``[lat, lon]`` list.
        label:    Descriptive label (e.g. "Source", "Waypoint 1").

    Returns:
        dict with keys: ``label``, ``address``, ``lat``, ``lng``.

    Raises:
        HTTPException 400: if the address cannot be resolved.
        HTTPException 502: if the Google API call fails.
    """
    try:
        if isinstance(location, list):
            # Coordinate pair — reverse geocode for a formatted address
            lat, lng = location[0], location[1]
            try:
                results = gmaps.reverse_geocode((lat, lng))
                address = (
                    results[0].get("formatted_address", f"{lat},{lng}")
                    if results
                    else f"{lat},{lng}"
                )
            except Exception:
                address = f"{lat},{lng}"
            return {"label": label, "address": address, "lat": lat, "lng": lng}

        # String address — forward geocode
        results = gmaps.geocode(location)
        if not results:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"Could not geocode {label} address: '{location}'. "
                    "Please verify the spelling or provide a more specific address."
                ),
            )
        geo = results[0]["geometry"]["location"]
        formatted = results[0].get("formatted_address", location)
        return {
            "label": label,
            "address": formatted,
            "lat": geo["lat"],
            "lng": geo["lng"],
        }

    except HTTPException:
        raise  # Re-raise our own 400s
    except Exception as exc:
        logger.error(f"Geocoding API failure for {label} ({location}): {exc}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=(
                f"Google Geocoding API error for {label} ('{location}'): {exc}. "
                "This may be a network timeout or API quota issue."
            ),
        )


def _geocode_all_locations(gmaps, payload: LocationQuery) -> List[dict]:
    """
    Geocode every location in the payload (source + waypoints + destination).

    The ordering convention for the returned list is:
      [0] = Source (depot)
      [1 … N-2] = Waypoints in input order
      [N-1] = Destination

    Args:
        gmaps:   googlemaps.Client instance.
        payload: Validated LocationQuery.

    Returns:
        List of location dicts.
    """
    locations = []

    # Source
    locations.append(
        _geocode_single(gmaps, payload.source, "Source")
    )

    # Waypoints
    for idx, wp in enumerate(payload.waypoints):
        locations.append(
            _geocode_single(gmaps, wp, f"Waypoint {idx + 1}")
        )

    # Destination
    locations.append(
        _geocode_single(gmaps, payload.destination, "Destination")
    )

    logger.info(
        f"📍 Geocoded {len(locations)} locations: "
        + ", ".join(f"{loc['label']}={loc['address']}" for loc in locations)
    )

    return locations


def _compute_next_departure(
    target_hour: int, target_minute: int, target_day_int: int
) -> datetime:
    """
    Compute the next future UTC datetime matching the requested
    day-of-week and time.

    Google's Distance Matrix API requires ``departure_time`` to be
    in the future to return ``duration_in_traffic`` values.

    Args:
        target_hour:    Hour (0–23).
        target_minute:  Minute (0–59).
        target_day_int: Day of week (0=Monday … 6=Sunday).

    Returns:
        A future ``datetime`` (UTC-naive).
    """
    now = datetime.utcnow()
    days_ahead = target_day_int - now.weekday()
    if days_ahead < 0:
        days_ahead += 7

    candidate = now.replace(
        hour=target_hour, minute=target_minute, second=0, microsecond=0
    ) + timedelta(days=days_ahead)

    if candidate <= now:
        candidate += timedelta(weeks=1)

    return candidate


def _build_travel_time_and_distance_matrices(
    gmaps,
    locations: List[dict],
    departure_time: datetime,
    traffic_model: str = "best_guess",
) -> tuple:
    """
    Build N×N travel-time and distance matrices via Google Distance Matrix API.

    The API is called with ``departure_time`` and ``traffic_model`` to
    incorporate real-time and predicted historical traffic congestion.

    Args:
        gmaps:          googlemaps.Client instance.
        locations:      Geocoded location list.
        departure_time: Future datetime for traffic estimation.
        traffic_model:  ``'best_guess'`` | ``'pessimistic'`` | ``'optimistic'``.

    Returns:
        Tuple of (time_matrix, distance_matrix) where each is an N×N
        list of lists. Time is in seconds, distance in meters.

    Raises:
        HTTPException 502: if the Distance Matrix API call fails.
    """
    n = len(locations)
    time_matrix = [[0] * n for _ in range(n)]
    dist_matrix = [[0] * n for _ in range(n)]
    coords = [(loc["lat"], loc["lng"]) for loc in locations]

    logger.info(
        f"🗺️  Building {n}×{n} matrix | "
        f"departure={departure_time.isoformat()} | "
        f"traffic_model={traffic_model}"
    )

    for i in range(n):
        try:
            result = gmaps.distance_matrix(
                origins=[coords[i]],
                destinations=coords,
                mode="driving",
                departure_time=departure_time,
                traffic_model=traffic_model,
            )
        except Exception as exc:
            logger.error(f"Distance Matrix API error for row {i}: {exc}")
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=(
                    f"Google Distance Matrix API failure for origin "
                    f"'{locations[i]['label']}': {exc}. "
                    "Check network connectivity and API quota."
                ),
            )

        elements = result["rows"][0]["elements"]
        for j, elem in enumerate(elements):
            if elem["status"] != "OK":
                logger.warning(
                    f"No route [{i}→{j}] "
                    f"({locations[i]['label']} → {locations[j]['label']}): "
                    f"{elem['status']}"
                )
                time_matrix[i][j] = 999999
                dist_matrix[i][j] = 999999
            else:
                if "duration_in_traffic" in elem:
                    time_matrix[i][j] = elem["duration_in_traffic"]["value"]
                else:
                    time_matrix[i][j] = elem["duration"]["value"]
                dist_matrix[i][j] = elem["distance"]["value"]

        # Diagonal = 0
        time_matrix[i][i] = 0
        dist_matrix[i][i] = 0

    logger.info(f"✅ Travel-time matrix built ({n}×{n})")
    return time_matrix, dist_matrix


def _apply_ml_overlay(
    raw_matrix: List[List[int]],
    hour: int,
    day_of_week: int,
    month: int,
    is_raining: int,
    is_festival: int,
) -> tuple:
    """
    Apply the XGBoost predictive buffer overlay to the raw matrix.

    **Strategy:**

    1. Predict trip duration under the caller's *actual* conditions.
    2. Predict trip duration under a *neutral* baseline
       (noon, Wednesday, no rain, no festival).
    3. Compute ``scaling_factor = actual / baseline``.
    4. Scale each off-diagonal cell by this factor.
    5. **Floor guardrail**: ``max(predicted_time, base_time)`` ensures
       no cell falls below the Google raw value.

    If no anomaly conditions are present (``is_raining == 0`` and
    ``is_festival == 0``), the raw matrix is passed through unchanged.

    Args:
        raw_matrix:   N×N travel-time matrix from Google (seconds).
        hour:         Departure hour (0–23).
        day_of_week:  0=Monday … 6=Sunday.
        month:        1–12.
        is_raining:   0 or 1.
        is_festival:  0 or 1.

    Returns:
        Tuple of (adjusted_matrix, scaling_factor).
        ``scaling_factor`` is ``None`` if overlay was skipped.
    """
    n = len(raw_matrix)

    if _ml_model is None:
        logger.info("ML model not loaded — skipping overlay")
        return [row[:] for row in raw_matrix], None

    if is_raining == 0 and is_festival == 0:
        logger.info("No anomaly conditions — skipping ML overlay")
        return [row[:] for row in raw_matrix], None

    # ── Compute scaling factor ──
    actual_features = np.array(
        [[hour, day_of_week, month, is_raining, is_festival]], dtype=float
    )
    actual_prediction = float(_ml_model.predict(actual_features)[0])

    baseline_features = np.array([[12, 2, month, 0, 0]], dtype=float)
    baseline_prediction = float(_ml_model.predict(baseline_features)[0])

    if baseline_prediction <= 0:
        logger.warning("Baseline prediction ≤ 0 — skipping ML overlay")
        return [row[:] for row in raw_matrix], None

    scaling_factor = actual_prediction / baseline_prediction

    logger.info(
        f"🧠 ML overlay: actual={actual_prediction:.0f}s, "
        f"baseline={baseline_prediction:.0f}s, "
        f"scale={scaling_factor:.4f}"
    )

    # ── Apply with floor guardrail ──
    adjusted = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                adjusted[i][j] = 0
                continue
            base_time = raw_matrix[i][j]
            predicted_time = int(round(base_time * scaling_factor))
            # Floor guardrail: adjusted value never falls below the raw base
            adjusted[i][j] = max(predicted_time, base_time)

    return adjusted, round(scaling_factor, 6)


def _solve_tsp(
    matrix: List[List[int]], locations: List[dict]
) -> dict:
    """
    Solve the Traveling Salesman Problem (TSP) / Vehicle Routing
    Problem (VRP) using Google OR-Tools.

    The vehicle starts and ends at the depot (node 0 = source).
    All intermediate nodes must be visited exactly once.

    Args:
        matrix:    N×N travel-time matrix (seconds).
        locations: Geocoded location list.

    Returns:
        dict with keys:
            ``sequence``       — list of node indices in visit order
            ``sequence_names`` — list of location labels
            ``total_seconds``  — total route duration

    Raises:
        HTTPException 500: if OR-Tools cannot find a feasible solution.
    """
    from ortools.constraint_solver import routing_enums_pb2, pywrapcp

    n = len(matrix)

    manager = pywrapcp.RoutingIndexManager(n, 1, 0)
    routing = pywrapcp.RoutingModel(manager)

    def time_callback(from_index: int, to_index: int) -> int:
        """Return travel time between two nodes."""
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return matrix[from_node][to_node]

    transit_cb_index = routing.RegisterTransitCallback(time_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_cb_index)

    search_params = pywrapcp.DefaultRoutingSearchParameters()
    search_params.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    search_params.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    search_params.time_limit.FromSeconds(5)

    logger.info(f"⚙️  Solving TSP ({n} nodes, depot={locations[0]['label']}) …")

    solution = routing.SolveWithParameters(search_params)

    if not solution:
        logger.error("OR-Tools could not find a feasible solution")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=(
                "The route optimizer could not find a feasible solution. "
                "Ensure all locations are reachable by driving."
            ),
        )

    # ── Extract solution ──
    sequence = []
    index = routing.Start(0)
    while not routing.IsEnd(index):
        sequence.append(manager.IndexToNode(index))
        index = solution.Value(routing.NextVar(index))
    sequence.append(manager.IndexToNode(index))  # return to depot

    total_seconds = solution.ObjectiveValue()
    sequence_names = [locations[idx]["label"] for idx in sequence]

    logger.info(
        f"✅ Solution: {total_seconds}s | "
        f"{'→'.join(sequence_names)}"
    )

    return {
        "sequence": sequence,
        "sequence_names": sequence_names,
        "total_seconds": total_seconds,
    }


# ==========================================================================
#  DEVELOPMENT SERVER ENTRY POINT
# ==========================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
