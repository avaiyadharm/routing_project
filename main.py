#!/usr/bin/env python3
"""
==========================================================================
 VEHICLE ROUTING OPTIMIZER — FastAPI REST API BACKEND
 Dual Backend: Google Maps (paid) or OpenStreetMap (free)
 XGBoost ML Overlay  ·  Google OR-Tools Solver
==========================================================================

 Production-grade REST API that exposes the full routing optimization
 pipeline as HTTP endpoints with Pydantic validation, deeply structured
 JSON responses, and enterprise-grade error handling.

 BACKENDS:
   - "osm"    (default) — FREE. Uses Nominatim geocoding + OSRM routing.
                          No API key required.
   - "google" — Paid.   Uses Google Maps Geocoding + Distance Matrix APIs.
                Requires GOOGLE_MAPS_API_KEY env var.

   Set via: GEO_BACKEND=osm|google  (defaults to "osm" if not set)

 Endpoints:
   GET  /                        → Health check
   POST /api/v1/optimize-route   → Full optimization pipeline

 Run:
   python main.py                           # free OSM backend
   GEO_BACKEND=google python main.py        # Google Maps backend

 Swagger UI:
   http://localhost:8000/docs

 What's new in v4.0.0:
   - route_legs array: per-leg base/ML/delay durations + coordinates
   - CO2 sustainability layer: per-leg and total emissions (kg)
   - Human-readable navigation steps array
   - ML diagnostics: feature echo, congestion tier, fallback flag
   - Fully restructured Pydantic response hierarchy

 Author:  Routing Engine Team
 Version: 4.0.0
 Date:    June 2026
==========================================================================
"""

import os
import pickle
import logging
import time as _time
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import requests as http_requests
from fastapi import FastAPI, HTTPException, status
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
MODEL_PATH   = PROJECT_ROOT / "traffic_xgb_model.pkl"

# Feature columns fed to the v1 XGBoost model — order must match training
ML_FEATURE_NAMES: List[str] = [
    "hour_of_day",
    "day_of_week",
    "month",
    "is_raining",
    "is_festival_zone",
]

# Backend selection: "osm" (free) or "google" (paid)
GEO_BACKEND = os.environ.get("GEO_BACKEND", "osm").lower()

# OSRM public demo server (free, no key required)
OSRM_BASE_URL = "https://router.project-osrm.org"

# Nominatim user-agent (required by OSM usage policy)
NOMINATIM_USER_AGENT = "vehicle-routing-optimizer/4.0.0"

# ---------------------------------------------------------------------------
#  Environmental constants
# ---------------------------------------------------------------------------
# Standard fleet emission coefficient: ~0.22 kg CO2 per kilometre for a
# medium commercial delivery van (ICCT / European Environment Agency baseline).
CO2_KG_PER_KM: float = 0.22

# Module-level reference to the ML model (loaded once at startup)
_ml_model = None


# ==========================================================================
#  PYDANTIC REQUEST SCHEMA
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
                raise ValueError(
                    "Coordinate pair must contain exactly 2 elements: [lat, lon]."
                )
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


# ==========================================================================
#  PYDANTIC RESPONSE SCHEMAS — deeply structured hierarchy
# ==========================================================================

class Coordinates(BaseModel):
    """A geographic coordinate pair."""

    lat: float = Field(..., description="Latitude in decimal degrees.")
    lng: float = Field(..., description="Longitude in decimal degrees.")


class RouteLeg(BaseModel):
    """
    Full telemetry for a single movement between two consecutive stops
    in the optimised route sequence.
    """

    leg_index: int = Field(
        ...,
        description="Zero-based index of this leg within the overall route.",
    )
    start_label: str = Field(
        ...,
        description="Human-readable label of the origin node (e.g. 'Source', 'Waypoint 2').",
    )
    end_label: str = Field(
        ...,
        description="Human-readable label of the destination node.",
    )
    start_address: str = Field(
        ...,
        description="Full geocoded address string of the leg's origin.",
    )
    end_address: str = Field(
        ...,
        description="Full geocoded address string of the leg's destination.",
    )
    start_coordinates: Coordinates = Field(
        ...,
        description="Lat/lng of the leg's origin.",
    )
    end_coordinates: Coordinates = Field(
        ...,
        description="Lat/lng of the leg's destination.",
    )
    distance_meters: int = Field(
        ...,
        description="Physical road distance for this leg (metres).",
    )
    base_duration_seconds: int = Field(
        ...,
        description=(
            "Open-road baseline travel time for this leg as returned by the "
            "geospatial routing engine (OSRM or Google Distance Matrix), "
            "before any ML adjustment."
        ),
    )
    ml_predicted_duration_seconds: int = Field(
        ...,
        description=(
            "Final dynamic travel time after the XGBoost model applies "
            "temporal, weather, and contextual weights. "
            "Always ≥ base_duration_seconds (floor guardrail enforced)."
        ),
    )
    traffic_delay_seconds: int = Field(
        ...,
        description=(
            "Explicit congestion-induced delay for this leg "
            "(ml_predicted_duration_seconds − base_duration_seconds). "
            "Zero when no ML overlay was applied."
        ),
    )
    estimated_co2_kg: float = Field(
        ...,
        description=(
            f"Estimated CO₂ emissions for this leg in kilograms. "
            f"Calculated as distance_km × {CO2_KG_PER_KM} kg/km "
            "(standard commercial delivery van baseline)."
        ),
    )
    navigation_instruction: str = Field(
        ...,
        description=(
            "Human-readable navigation directive for this leg "
            "(e.g. 'Depart from Source toward Waypoint 1')."
        ),
    )


class MLInputFeatures(BaseModel):
    """
    Echo of the exact feature vector passed into the XGBoost model matrix.
    Enables reproducibility audits and client-side debugging.
    """

    hour_of_day:     int   = Field(..., description="Hour of departure (0–23).")
    day_of_week:     int   = Field(..., description="Day of week (0=Mon … 6=Sun).")
    month:           int   = Field(..., description="Calendar month (1–12).")
    is_raining:      int   = Field(..., description="Rain flag (0 or 1).")
    is_festival_zone: int  = Field(..., description="Festival zone flag (0 or 1).")


class MLDiagnostics(BaseModel):
    """
    Comprehensive machine-learning context diagnostics sub-object.
    Exposes the model's decision inputs, congestion classification,
    computed scaling factor, and backend fallback status.
    """

    input_features_evaluated: MLInputFeatures = Field(
        ...,
        description="Exact feature values passed to the XGBoost model.",
    )
    scaling_factor: Optional[float] = Field(
        None,
        description=(
            "Global travel-time multiplier derived from the ML model "
            "(actual_prediction / baseline_prediction). "
            "None when the ML overlay was skipped."
        ),
    )
    actual_prediction_seconds: Optional[float] = Field(
        None,
        description="Raw XGBoost prediction for actual conditions (seconds).",
    )
    baseline_prediction_seconds: Optional[float] = Field(
        None,
        description="Raw XGBoost prediction for neutral baseline conditions (seconds).",
    )
    congestion_tier: str = Field(
        ...,
        description=(
            "Categorical congestion classification derived from the "
            "computed scaling factor:\n"
            "  • 'Clear'              — scale < 1.05\n"
            "  • 'Light Congestion'   — 1.05 ≤ scale < 1.20\n"
            "  • 'Moderate Congestion' — 1.20 ≤ scale < 1.50\n"
            "  • 'Heavy Congestion'   — 1.50 ≤ scale < 2.00\n"
            "  • 'Severe Gridlock'    — scale ≥ 2.00\n"
            "  • 'N/A (ML Skipped)'  — overlay not applied."
        ),
    )
    overlay_applied: bool = Field(
        ...,
        description=(
            "True when the ML scaling factor was computed and applied "
            "to the travel-time matrix; False when the overlay was skipped "
            "(no adverse conditions or model not loaded)."
        ),
    )
    system_fallback_active: bool = Field(
        ...,
        description=(
            "True when the system is operating on the free OpenStreetMap "
            "backend (Nominatim + OSRM) — i.e. GEO_BACKEND != 'google' or "
            "GOOGLE_MAPS_API_KEY is absent. "
            "False when the live Google Cloud environment is connected."
        ),
    )
    ml_model_loaded: bool = Field(
        ...,
        description="True when traffic_xgb_model.pkl was successfully loaded at startup.",
    )


class SustainabilityTelemetry(BaseModel):
    """
    Route-level environmental impact summary.
    Individual leg emissions are reported inside each RouteLeg object.
    """

    total_distance_km: float = Field(
        ...,
        description="Total optimised route distance in kilometres.",
    )
    total_co2_kg: float = Field(
        ...,
        description=(
            "Total estimated CO₂ emissions for the full route (kg). "
            f"Formula: total_distance_km × {CO2_KG_PER_KM} kg/km."
        ),
    )
    emission_coefficient_kg_per_km: float = Field(
        default=CO2_KG_PER_KM,
        description="Emission factor used (kg CO₂ per km). Standard commercial van baseline.",
    )
    co2_offset_trees_equivalent: float = Field(
        ...,
        description=(
            "Equivalent number of mature trees needed to absorb the total CO₂ "
            "over one year (~21 kg CO₂ absorbed per tree per year)."
        ),
    )


class ContextApplied(BaseModel):
    """Echo of the weather / temporal / event context applied in this run."""

    departure_hour:    int
    day_of_week:       int
    month:             int
    is_raining:        bool
    is_festival_zone:  bool
    ml_scaling_factor: Optional[float] = Field(
        None,
        description=(
            "ML-derived scaling factor applied to the travel matrix. "
            "None if ML overlay was skipped."
        ),
    )


class GeocodedLocation(BaseModel):
    """A geocoded location entry returned for client reference."""

    label:       str   = Field(..., description="Node label (e.g. 'Source', 'Waypoint 1').")
    address:     str   = Field(..., description="Full resolved address string.")
    lat:         float = Field(..., description="Latitude.")
    lng:         float = Field(..., description="Longitude.")


class OptimizeRouteResponse(BaseModel):
    """
    Structured JSON response from ``POST /api/v1/optimize-route``.

    Hierarchy:
      root
      ├── status / geo_backend / version
      ├── optimal_sequence_order          (ordered list of node labels)
      ├── total_optimized_duration_seconds
      ├── total_distance_meters
      ├── route_legs[]                    ← per-leg detail (NEW in v4)
      ├── navigation_steps[]              ← human-readable steps (NEW in v4)
      ├── sustainability                  ← CO₂ telemetry (NEW in v4)
      ├── ml_diagnostics                  ← ML context dump (NEW in v4)
      ├── context_applied
      ├── geocoded_locations[]
      ├── raw_matrix_seconds[][]
      └── ml_adjusted_matrix_seconds[][]
    """

    status:  str = Field(default="success", description="Operation status.")
    version: str = Field(default="4.0.0",   description="API version.")
    geo_backend: str = Field(
        ...,
        description="Active geospatial backend: 'osm' (free) or 'google' (paid).",
    )

    optimal_sequence_order: List[str] = Field(
        ...,
        description=(
            "Ordered list of location labels representing the optimal visit "
            "sequence (e.g. Source → Waypoint 2 → Waypoint 1 → Destination → Source)."
        ),
    )
    total_optimized_duration_seconds: int = Field(
        ...,
        description="Total travel time for the optimised route (seconds, ML-adjusted).",
    )
    total_distance_meters: int = Field(
        ...,
        description="Total physical road distance covered (metres).",
    )

    # ── NEW: Rich per-leg breakdown ────────────────────────────────────────
    route_legs: List[RouteLeg] = Field(
        ...,
        description=(
            "Granular per-leg breakdown for every movement in the optimised sequence. "
            "Each entry contains base time, ML-adjusted time, delay, CO₂, "
            "coordinates, and a navigation instruction."
        ),
    )

    # ── NEW: Navigation step array ─────────────────────────────────────────
    navigation_steps: List[str] = Field(
        ...,
        description=(
            "Ordered list of human-readable navigation directives "
            "a driver can follow sequentially."
        ),
    )

    # ── NEW: Sustainability telemetry ──────────────────────────────────────
    sustainability: SustainabilityTelemetry = Field(
        ...,
        description="Route-level CO₂ emissions and environmental impact summary.",
    )

    # ── NEW: ML diagnostics ────────────────────────────────────────────────
    ml_diagnostics: MLDiagnostics = Field(
        ...,
        description=(
            "Full machine-learning context diagnostics: input features, "
            "congestion tier, scaling factor, and system status flags."
        ),
    )

    context_applied: ContextApplied = Field(
        ...,
        description="Echo of the weather / temporal / event context processed.",
    )
    geocoded_locations: List[GeocodedLocation] = Field(
        ...,
        description="All geocoded locations with labels, addresses, and coordinates.",
    )
    raw_matrix_seconds: List[List[int]] = Field(
        ...,
        description="Raw N×N travel-time matrix from the routing engine (seconds).",
    )
    ml_adjusted_matrix_seconds: List[List[int]] = Field(
        ...,
        description="ML-adjusted N×N travel-time matrix (seconds).",
    )


class HealthResponse(BaseModel):
    """Health check response schema."""

    status:          str  = Field(default="online",  description="Server status.")
    service:         str  = Field(
        default="Vehicle Routing Optimizer API",
        description="Service identifier.",
    )
    version:         str  = Field(default="4.0.0",   description="API version.")
    geo_backend:     str  = Field(..., description="Active backend: 'osm' or 'google'.")
    ml_model_loaded: bool = Field(..., description="XGBoost model loaded at startup.")
    timestamp:       str  = Field(..., description="Current server time (ISO 8601).")


# ==========================================================================
#  APPLICATION LIFESPAN  (load ML model once at startup)
# ==========================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load heavy resources on startup; release on shutdown."""
    global _ml_model

    logger.info("🚀 Starting Vehicle Routing Optimizer API v4.0.0 …")
    logger.info(f"🌐 Geo backend: {GEO_BACKEND.upper()}")

    if GEO_BACKEND == "google":
        api_key = os.environ.get("GOOGLE_MAPS_API_KEY")
        if api_key:
            logger.info("🔑 GOOGLE_MAPS_API_KEY detected")
        else:
            logger.warning(
                "⚠️  GEO_BACKEND=google but GOOGLE_MAPS_API_KEY is not set. "
                "API calls will fail."
            )
    else:
        logger.info("🆓 Using FREE OpenStreetMap backend (Nominatim + OSRM)")

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
        "Production-grade REST API for multi-stop vehicle route optimization.\n\n"
        "**Dual backend support:**\n"
        "- **OSM (free)** — OpenStreetMap Nominatim geocoding + OSRM routing. No API key needed.\n"
        "- **Google (paid)** — Google Maps Geocoding + Distance Matrix with live traffic.\n\n"
        "Set `GEO_BACKEND=osm` (default) or `GEO_BACKEND=google` to choose.\n\n"
        "Both backends feed into the XGBoost ML prediction overlay and "
        "Google OR-Tools TSP/VRP solver.\n\n"
        "**v4.0 additions:** per-leg telemetry, CO₂ sustainability layer, "
        "navigation steps, ML diagnostics."
    ),
    version="4.0.0",
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
    description="Returns server status, active backend, and ML model load status.",
)
async def health_check() -> HealthResponse:
    """
    **Health Check**

    Returns a JSON object confirming the API is online, including the
    active geo backend and XGBoost model load status.
    """
    return HealthResponse(
        status="online",
        service="Vehicle Routing Optimizer API",
        version="4.0.0",
        geo_backend=GEO_BACKEND,
        ml_model_loaded=_ml_model is not None,
        timestamp=datetime.utcnow().isoformat() + "Z",
    )


# ==========================================================================
#  ENDPOINT: POST /api/v1/optimize-route
# ==========================================================================

@app.post(
    "/api/v1/optimize-route",
    response_model=OptimizeRouteResponse,
    tags=["Routing"],
    summary="Optimize a multi-stop vehicle route",
    description=(
        "Accepts a `LocationQuery` payload, geocodes all addresses, "
        "builds a travel-time matrix, applies the XGBoost ML prediction "
        "overlay, and solves the optimal visit sequence via OR-Tools.\n\n"
        "**Response includes:**\n"
        "- `route_legs`: per-leg base/ML/delay durations, coordinates, CO₂, navigation step\n"
        "- `navigation_steps`: ordered human-readable driving directives\n"
        "- `sustainability`: total CO₂ emissions and tree-equivalent offset\n"
        "- `ml_diagnostics`: feature echo, congestion tier, system flags"
    ),
    responses={
        400: {"description": "Invalid address or geocoding failure."},
        500: {"description": "Server configuration error or solver failure."},
        502: {"description": "Upstream geocoding / routing API failure."},
    },
)
async def optimize_route(payload: LocationQuery) -> OptimizeRouteResponse:
    """
    **Full Route Optimization Pipeline (v4.0)**

    Processing stages:
    1. Geocode all locations (via Nominatim or Google Maps)
    2. Build N×N travel-time + distance matrices (via OSRM or Google)
    3. Apply XGBoost ML travel-time scaling overlay
    4. Solve optimal visit sequence via Google OR-Tools TSP
    5. Assemble per-leg telemetry, CO₂ metrics, navigation steps, ML diagnostics
    6. Return fully structured JSON response

    Raises:
        HTTPException 400: Invalid or unresolvable address.
        HTTPException 500: Missing API key, solver failure, or config error.
        HTTPException 502: Upstream API network / timeout failure.
    """
    logger.info(
        f"📥 Incoming optimization request [{GEO_BACKEND.upper()}]: "
        f"source={payload.source!r}, dest={payload.destination!r}, "
        f"waypoints={len(payload.waypoints)}, "
        f"hour={payload.departure_hour}, day={payload.day_of_week}, "
        f"rain={payload.is_raining}, festival={payload.is_festival_zone}"
    )

    # ── Step 1: Geocode all locations ──────────────────────────────────────
    locations = _geocode_all_locations(payload)

    # ── Step 2: Build travel-time + distance matrices ──────────────────────
    departure_dt = _compute_next_departure(
        payload.departure_hour, 0, payload.day_of_week
    )
    raw_matrix, distance_matrix_meters = _build_matrices(locations, departure_dt)

    # ── Step 3: ML overlay ─────────────────────────────────────────────────
    (
        adjusted_matrix,
        scaling_factor,
        actual_pred,
        baseline_pred,
    ) = _apply_ml_overlay(
        raw_matrix=raw_matrix,
        hour=payload.departure_hour,
        day_of_week=payload.day_of_week,
        month=payload.month,
        is_raining=payload.is_raining,
        is_festival=payload.is_festival_zone,
    )

    # ── Step 4: Solve with OR-Tools ────────────────────────────────────────
    solution = _solve_tsp(adjusted_matrix, locations)

    # ── Step 5: Assemble per-leg telemetry ─────────────────────────────────
    route_legs: List[RouteLeg]  = []
    navigation_steps: List[str] = []
    total_distance_m            = 0

    sequence = solution["sequence"]
    for k in range(len(sequence) - 1):
        i = sequence[k]
        j = sequence[k + 1]

        base_sec  = raw_matrix[i][j]
        ml_sec    = adjusted_matrix[i][j]
        # Floor guardrail — ML time never falls below raw baseline
        ml_sec    = max(ml_sec, base_sec)
        delay_sec = ml_sec - base_sec

        leg_dist_m  = distance_matrix_meters[i][j]
        leg_dist_km = leg_dist_m / 1000.0
        leg_co2_kg  = round(leg_dist_km * CO2_KG_PER_KM, 4)

        total_distance_m += leg_dist_m

        loc_i = locations[i]
        loc_j = locations[j]

        # Navigation instruction
        is_last_leg = (k == len(sequence) - 2)
        if k == 0:
            nav_step = (
                f"🚦 Depart from {loc_i['label']} ({loc_i['address'][:60]}) "
                f"toward {loc_j['label']}."
            )
        elif is_last_leg and loc_j["label"] == locations[sequence[0]]["label"]:
            nav_step = f"🏁 Return to depot at {loc_j['label']} ({loc_j['address'][:60]})."
        elif is_last_leg:
            nav_step = f"🏁 Arrive at final destination: {loc_j['label']} ({loc_j['address'][:60]})."
        else:
            nav_step = (
                f"➡️  Continue from {loc_i['label']} to {loc_j['label']} "
                f"({loc_j['address'][:60]})."
            )

        route_legs.append(
            RouteLeg(
                leg_index=k,
                start_label=loc_i["label"],
                end_label=loc_j["label"],
                start_address=loc_i["address"],
                end_address=loc_j["address"],
                start_coordinates=Coordinates(lat=loc_i["lat"], lng=loc_i["lng"]),
                end_coordinates=Coordinates(lat=loc_j["lat"], lng=loc_j["lng"]),
                distance_meters=leg_dist_m,
                base_duration_seconds=base_sec,
                ml_predicted_duration_seconds=ml_sec,
                traffic_delay_seconds=delay_sec,
                estimated_co2_kg=leg_co2_kg,
                navigation_instruction=nav_step,
            )
        )
        navigation_steps.append(nav_step)

    # ── Step 6: Sustainability totals ──────────────────────────────────────
    total_dist_km  = total_distance_m / 1000.0
    total_co2_kg   = round(total_dist_km * CO2_KG_PER_KM, 4)
    # One mature tree absorbs ~21 kg CO2/year (FAO estimate)
    tree_equivalent = round(total_co2_kg / 21.0, 2)

    sustainability = SustainabilityTelemetry(
        total_distance_km=round(total_dist_km, 3),
        total_co2_kg=total_co2_kg,
        emission_coefficient_kg_per_km=CO2_KG_PER_KM,
        co2_offset_trees_equivalent=tree_equivalent,
    )

    # ── Step 7: ML diagnostics ─────────────────────────────────────────────
    congestion_tier = _classify_congestion(scaling_factor)
    overlay_applied = scaling_factor is not None
    system_fallback = not (
        GEO_BACKEND == "google" and bool(os.environ.get("GOOGLE_MAPS_API_KEY"))
    )

    ml_diagnostics = MLDiagnostics(
        input_features_evaluated=MLInputFeatures(
            hour_of_day=payload.departure_hour,
            day_of_week=payload.day_of_week,
            month=payload.month,
            is_raining=payload.is_raining,
            is_festival_zone=payload.is_festival_zone,
        ),
        scaling_factor=scaling_factor,
        actual_prediction_seconds=round(actual_pred, 2) if actual_pred is not None else None,
        baseline_prediction_seconds=round(baseline_pred, 2) if baseline_pred is not None else None,
        congestion_tier=congestion_tier,
        overlay_applied=overlay_applied,
        system_fallback_active=system_fallback,
        ml_model_loaded=_ml_model is not None,
    )

    # ── Step 8: Assemble final response ────────────────────────────────────
    response = OptimizeRouteResponse(
        status="success",
        version="4.0.0",
        geo_backend=GEO_BACKEND,
        optimal_sequence_order=solution["sequence_names"],
        total_optimized_duration_seconds=solution["total_seconds"],
        total_distance_meters=total_distance_m,
        route_legs=route_legs,
        navigation_steps=navigation_steps,
        sustainability=sustainability,
        ml_diagnostics=ml_diagnostics,
        context_applied=ContextApplied(
            departure_hour=payload.departure_hour,
            day_of_week=payload.day_of_week,
            month=payload.month,
            is_raining=bool(payload.is_raining),
            is_festival_zone=bool(payload.is_festival_zone),
            ml_scaling_factor=scaling_factor,
        ),
        geocoded_locations=[
            GeocodedLocation(
                label=loc["label"],
                address=loc["address"],
                lat=loc["lat"],
                lng=loc["lng"],
            )
            for loc in locations
        ],
        raw_matrix_seconds=raw_matrix,
        ml_adjusted_matrix_seconds=adjusted_matrix,
    )

    logger.info(
        f"✅ Optimization complete [{GEO_BACKEND.upper()}]: "
        f"{solution['total_seconds']}s total, "
        f"{total_distance_m}m, "
        f"CO₂={total_co2_kg}kg, "
        f"tier='{congestion_tier}', "
        f"sequence={'→'.join(solution['sequence_names'])}"
    )

    return response


# ==========================================================================
#  BACKEND ROUTER — dispatches to Google or OSM functions
# ==========================================================================

def _geocode_all_locations(payload: LocationQuery) -> List[dict]:
    """
    Geocode every location in the payload, dispatching to the
    active backend (Google Maps or OpenStreetMap Nominatim).

    Args:
        payload: Validated LocationQuery instance.

    Returns:
        List of location dicts with keys: label, address, lat, lng.
    """
    if GEO_BACKEND == "google":
        gmaps = _get_gmaps_client()
        return _google_geocode_all(gmaps, payload)
    else:
        return _osm_geocode_all(payload)


def _build_matrices(
    locations: List[dict], departure_time: datetime
) -> tuple:
    """
    Build N×N travel-time and distance matrices, dispatching to the
    active backend (Google Distance Matrix or OSRM Table).

    Args:
        locations:      Geocoded location list.
        departure_time: Future datetime used for traffic estimation (Google).

    Returns:
        Tuple of (time_matrix_seconds, distance_matrix_meters).
    """
    if GEO_BACKEND == "google":
        gmaps = _get_gmaps_client()
        return _google_build_matrices(gmaps, locations, departure_time)
    else:
        return _osrm_build_matrices(locations)


# ==========================================================================
#  BACKEND A: FREE — OpenStreetMap (Nominatim + OSRM)
# ==========================================================================

def _osm_geocode_single(
    location: Union[str, List[float]], label: str
) -> dict:
    """
    Geocode a single location via OpenStreetMap Nominatim (free).

    If the location is already a [lat, lon] pair, a reverse-geocode
    is performed to obtain a human-readable address.

    Args:
        location: Address string or [lat, lon] list.
        label:    Descriptive label (e.g. "Source", "Waypoint 1").

    Returns:
        dict with keys: label, address, lat, lng.

    Raises:
        HTTPException 400: if the address cannot be resolved.
        HTTPException 502: if Nominatim API call fails.
    """
    try:
        if isinstance(location, list):
            lat, lng = location[0], location[1]
            # Reverse geocode — best-effort; fall back to raw coords
            try:
                resp = http_requests.get(
                    "https://nominatim.openstreetmap.org/reverse",
                    params={
                        "lat": lat,
                        "lon": lng,
                        "format": "json",
                        "addressdetails": 1,
                    },
                    headers={"User-Agent": NOMINATIM_USER_AGENT},
                    timeout=10,
                )
                resp.raise_for_status()
                data    = resp.json()
                address = data.get("display_name", f"{lat},{lng}")
            except Exception:
                address = f"{lat},{lng}"
            return {"label": label, "address": address, "lat": lat, "lng": lng}

        # Forward geocode
        resp = http_requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={
                "q": location,
                "format": "json",
                "limit": 1,
                "addressdetails": 1,
            },
            headers={"User-Agent": NOMINATIM_USER_AGENT},
            timeout=10,
        )
        resp.raise_for_status()
        results = resp.json()

        if not results:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"Could not geocode {label} address: '{location}'. "
                    "Please verify the spelling or provide a more specific address."
                ),
            )

        hit = results[0]
        return {
            "label":   label,
            "address": hit.get("display_name", location),
            "lat":     float(hit["lat"]),
            "lng":     float(hit["lon"]),
        }

    except HTTPException:
        raise
    except http_requests.exceptions.Timeout:
        logger.error(f"Nominatim timeout for {label} ({location})")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=(
                f"Nominatim geocoding timeout for {label} ('{location}'). "
                "The OpenStreetMap server may be busy. Please try again."
            ),
        )
    except Exception as exc:
        logger.error(f"Nominatim geocoding failure for {label} ({location}): {exc}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Nominatim geocoding error for {label} ('{location}'): {exc}.",
        )


def _osm_geocode_all(payload: LocationQuery) -> List[dict]:
    """
    Geocode all locations in the payload via Nominatim.

    Nominatim usage policy requires ≤ 1 request/second,
    so a 1.1-second delay is inserted between each call.

    Args:
        payload: Validated LocationQuery instance.

    Returns:
        List of geocoded location dicts.
    """
    locations = []
    all_items = (
        [("Source", payload.source)]
        + [(f"Waypoint {i+1}", wp) for i, wp in enumerate(payload.waypoints)]
        + [("Destination", payload.destination)]
    )

    for idx, (label, loc) in enumerate(all_items):
        if idx > 0:
            _time.sleep(1.1)  # Respect Nominatim's 1 req/s rate limit
        locations.append(_osm_geocode_single(loc, label))

    logger.info(
        f"📍 Geocoded {len(locations)} locations [Nominatim]: "
        + ", ".join(f"{loc['label']}={loc['address'][:40]}" for loc in locations)
    )
    return locations


def _osrm_build_matrices(locations: List[dict]) -> tuple:
    """
    Build N×N travel-time and distance matrices via the OSRM
    public Table API (free, no API key required).

    OSRM's /table/v1/driving endpoint returns duration and distance
    matrices for all coordinate pairs in a single request.

    Args:
        locations: Geocoded location list.

    Returns:
        Tuple of (time_matrix_seconds, distance_matrix_meters).

    Raises:
        HTTPException 502: if the OSRM API call fails.
    """
    n = len(locations)

    # OSRM expects coordinates as lon,lat (reversed from lat,lng)
    coords_str = ";".join(f"{loc['lng']},{loc['lat']}" for loc in locations)
    url        = f"{OSRM_BASE_URL}/table/v1/driving/{coords_str}"
    params     = {"annotations": "duration,distance"}

    logger.info(f"🗺️  Building {n}×{n} matrix via OSRM Table API [FREE]")

    try:
        resp = http_requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except http_requests.exceptions.Timeout:
        logger.error("OSRM Table API timeout")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=(
                "OSRM routing API timeout. The public OSRM server may be busy. "
                "Please try again in a moment."
            ),
        )
    except Exception as exc:
        logger.error(f"OSRM Table API error: {exc}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"OSRM routing API error: {exc}",
        )

    if data.get("code") != "Ok":
        error_msg = data.get("message", "Unknown OSRM error")
        logger.error(f"OSRM returned error: {error_msg}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=(
                f"OSRM could not compute routes: {error_msg}. "
                "Ensure all locations are on routable roads."
            ),
        )

    # Parse duration matrix
    time_matrix: List[List[int]] = []
    for row in data["durations"]:
        time_matrix.append(
            [int(round(val)) if val is not None else 999_999 for val in row]
        )

    # Parse distance matrix
    dist_matrix: List[List[int]] = []
    for row in data["distances"]:
        dist_matrix.append(
            [int(round(val)) if val is not None else 999_999 for val in row]
        )

    # Zero the diagonals
    for i in range(n):
        time_matrix[i][i] = 0
        dist_matrix[i][i] = 0

    logger.info(f"✅ OSRM travel-time matrix built ({n}×{n})")
    return time_matrix, dist_matrix


# ==========================================================================
#  BACKEND B: PAID — Google Maps APIs
# ==========================================================================

def _get_gmaps_client():
    """
    Instantiate the Google Maps Python client.

    Reads the API key from the ``GOOGLE_MAPS_API_KEY`` environment
    variable. Raises HTTP 500 if the key is missing or the package
    is not installed.

    Returns:
        googlemaps.Client instance.

    Raises:
        HTTPException 500: API key absent or package missing.
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
                "variable is not set. Either set it, or switch to the free "
                "OSM backend with: GEO_BACKEND=osm"
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


def _google_geocode_single(
    gmaps, location: Union[str, List[float]], label: str
) -> dict:
    """
    Geocode a single location via Google Maps Geocoding API.

    Args:
        gmaps:    googlemaps.Client instance.
        location: Address string or [lat, lon] list.
        label:    Descriptive label.

    Returns:
        dict with keys: label, address, lat, lng.

    Raises:
        HTTPException 400: if the address cannot be resolved.
        HTTPException 502: if the Google API call fails.
    """
    try:
        if isinstance(location, list):
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

        results = gmaps.geocode(location)
        if not results:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"Could not geocode {label} address: '{location}'. "
                    "Please verify the spelling or provide a more specific address."
                ),
            )
        geo       = results[0]["geometry"]["location"]
        formatted = results[0].get("formatted_address", location)
        return {
            "label":   label,
            "address": formatted,
            "lat":     geo["lat"],
            "lng":     geo["lng"],
        }

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Google Geocoding failure for {label} ({location}): {exc}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=(
                f"Google Geocoding API error for {label} ('{location}'): {exc}. "
                "This may be a network timeout or API quota issue."
            ),
        )


def _google_geocode_all(gmaps, payload: LocationQuery) -> List[dict]:
    """
    Geocode all locations in the payload via Google Maps.

    Args:
        gmaps:   googlemaps.Client instance.
        payload: Validated LocationQuery.

    Returns:
        List of geocoded location dicts.
    """
    locations = []
    all_items = (
        [("Source", payload.source)]
        + [(f"Waypoint {i+1}", wp) for i, wp in enumerate(payload.waypoints)]
        + [("Destination", payload.destination)]
    )

    for label, loc in all_items:
        locations.append(_google_geocode_single(gmaps, loc, label))

    logger.info(
        f"📍 Geocoded {len(locations)} locations [Google]: "
        + ", ".join(f"{loc['label']}={loc['address'][:40]}" for loc in locations)
    )
    return locations


def _google_build_matrices(
    gmaps,
    locations: List[dict],
    departure_time: datetime,
    traffic_model: str = "best_guess",
) -> tuple:
    """
    Build N×N travel-time and distance matrices via Google Distance Matrix API.

    Args:
        gmaps:          googlemaps.Client instance.
        locations:      Geocoded location list.
        departure_time: Future datetime for traffic estimation.
        traffic_model:  'best_guess' | 'pessimistic' | 'optimistic'.

    Returns:
        Tuple of (time_matrix_seconds, distance_matrix_meters).

    Raises:
        HTTPException 502: if the Distance Matrix API call fails.
    """
    n           = len(locations)
    time_matrix = [[0] * n for _ in range(n)]
    dist_matrix = [[0] * n for _ in range(n)]
    coords      = [(loc["lat"], loc["lng"]) for loc in locations]

    logger.info(
        f"🗺️  Building {n}×{n} matrix [Google Distance Matrix] | "
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
            logger.error(f"Google Distance Matrix API error for row {i}: {exc}")
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
                time_matrix[i][j] = 999_999
                dist_matrix[i][j] = 999_999
            else:
                if "duration_in_traffic" in elem:
                    time_matrix[i][j] = elem["duration_in_traffic"]["value"]
                else:
                    time_matrix[i][j] = elem["duration"]["value"]
                dist_matrix[i][j] = elem["distance"]["value"]

        time_matrix[i][i] = 0
        dist_matrix[i][i] = 0

    logger.info(f"✅ Google travel-time matrix built ({n}×{n})")
    return time_matrix, dist_matrix


# ==========================================================================
#  SHARED INTERNAL FUNCTIONS
# ==========================================================================

def _compute_next_departure(
    target_hour: int, target_minute: int, target_day_int: int
) -> datetime:
    """
    Compute the next future UTC datetime matching the requested
    day-of-week and hour/minute.

    Google's Distance Matrix API requires departure_time to be
    in the future to return ``duration_in_traffic`` values.

    Args:
        target_hour:    Hour (0–23).
        target_minute:  Minute (0–59).
        target_day_int: Day of week (0=Monday … 6=Sunday).

    Returns:
        A future datetime (UTC-naive).
    """
    now        = datetime.utcnow()
    days_ahead = target_day_int - now.weekday()
    if days_ahead < 0:
        days_ahead += 7

    candidate = now.replace(
        hour=target_hour, minute=target_minute, second=0, microsecond=0
    ) + timedelta(days=days_ahead)

    if candidate <= now:
        candidate += timedelta(weeks=1)

    return candidate


def _apply_ml_overlay(
    raw_matrix: List[List[int]],
    hour:        int,
    day_of_week: int,
    month:       int,
    is_raining:  int,
    is_festival: int,
) -> tuple:
    """
    Apply the XGBoost predictive buffer overlay to the raw travel-time matrix.

    Strategy:
    1. Predict trip duration under the caller's actual conditions.
    2. Predict trip duration under a neutral baseline
       (noon, Wednesday, month unchanged, no rain, no festival).
    3. Compute ``scaling_factor = actual / baseline``.
    4. Scale each off-diagonal cell by this factor.
    5. Floor guardrail: ``max(predicted_time, base_time)`` ensures
       no cell falls below the raw baseline value.

    The overlay is skipped (matrix returned unchanged) when:
    - The ML model is not loaded, OR
    - No adverse conditions are present (no rain AND no festival).

    Args:
        raw_matrix:   N×N travel-time matrix (seconds).
        hour:         Departure hour (0–23).
        day_of_week:  0=Monday … 6=Sunday.
        month:        1–12.
        is_raining:   0 or 1.
        is_festival:  0 or 1.

    Returns:
        Tuple of (adjusted_matrix, scaling_factor, actual_pred, baseline_pred).
        scaling_factor / predictions are None when overlay is skipped.
    """
    n = len(raw_matrix)

    if _ml_model is None:
        logger.info("ML model not loaded — skipping overlay")
        return [row[:] for row in raw_matrix], None, None, None

    if is_raining == 0 and is_festival == 0:
        logger.info("No adverse conditions — skipping ML overlay")
        return [row[:] for row in raw_matrix], None, None, None

    # ── Predict under actual and baseline conditions ───────────────────────
    actual_features   = np.array(
        [[hour, day_of_week, month, is_raining, is_festival]], dtype=float
    )
    baseline_features = np.array([[12, 2, month, 0, 0]], dtype=float)

    actual_pred   = float(_ml_model.predict(actual_features)[0])
    baseline_pred = float(_ml_model.predict(baseline_features)[0])

    if baseline_pred <= 0:
        logger.warning("Baseline prediction ≤ 0 — skipping ML overlay")
        return [row[:] for row in raw_matrix], None, None, None

    scaling_factor = actual_pred / baseline_pred

    logger.info(
        f"🧠 ML overlay: actual={actual_pred:.0f}s, "
        f"baseline={baseline_pred:.0f}s, "
        f"scale={scaling_factor:.4f}"
    )

    # ── Apply with per-cell floor guardrail ───────────────────────────────
    adjusted: List[List[int]] = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                adjusted[i][j] = 0
                continue
            base_time      = raw_matrix[i][j]
            predicted_time = int(round(base_time * scaling_factor))
            # Floor guardrail: adjusted cell never falls below the raw baseline
            adjusted[i][j] = max(predicted_time, base_time)

    return adjusted, round(scaling_factor, 6), actual_pred, baseline_pred


def _classify_congestion(scaling_factor: Optional[float]) -> str:
    """
    Map a numeric ML scaling factor to a human-readable congestion tier.

    Tier thresholds:
      Clear              < 1.05
      Light Congestion   1.05 – 1.20
      Moderate Congestion 1.20 – 1.50
      Heavy Congestion   1.50 – 2.00
      Severe Gridlock    ≥ 2.00

    Args:
        scaling_factor: ML-derived multiplier or None (overlay skipped).

    Returns:
        Categorical string label.
    """
    if scaling_factor is None:
        return "N/A (ML Skipped)"
    if scaling_factor < 1.05:
        return "Clear"
    if scaling_factor < 1.20:
        return "Light Congestion"
    if scaling_factor < 1.50:
        return "Moderate Congestion"
    if scaling_factor < 2.00:
        return "Heavy Congestion"
    return "Severe Gridlock"


def _solve_tsp(
    matrix: List[List[int]], locations: List[dict]
) -> dict:
    """
    Solve the Traveling Salesman Problem (TSP) via Google OR-Tools.

    The vehicle starts and ends at the depot (node 0 = source).
    All intermediate nodes must be visited exactly once.

    Args:
        matrix:    N×N travel-time matrix (seconds, ML-adjusted).
        locations: Geocoded location list.

    Returns:
        dict with keys:
            sequence       — list of node indices in visit order
            sequence_names — list of location labels
            total_seconds  — total route duration (seconds)

    Raises:
        HTTPException 500: if OR-Tools cannot find a feasible solution.
    """
    from ortools.constraint_solver import routing_enums_pb2, pywrapcp

    n = len(matrix)

    manager = pywrapcp.RoutingIndexManager(n, 1, 0)
    routing = pywrapcp.RoutingModel(manager)

    def time_callback(from_index: int, to_index: int) -> int:
        """Return travel time between two OR-Tools node indices."""
        return matrix[manager.IndexToNode(from_index)][manager.IndexToNode(to_index)]

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

    # ── Extract solution sequence ──────────────────────────────────────────
    sequence: List[int] = []
    index = routing.Start(0)
    while not routing.IsEnd(index):
        sequence.append(manager.IndexToNode(index))
        index = solution.Value(routing.NextVar(index))
    sequence.append(manager.IndexToNode(index))  # return to depot

    total_seconds  = solution.ObjectiveValue()
    sequence_names = [locations[idx]["label"] for idx in sequence]

    logger.info(
        f"✅ TSP solution: {total_seconds}s | {'→'.join(sequence_names)}"
    )

    return {
        "sequence":       sequence,
        "sequence_names": sequence_names,
        "total_seconds":  total_seconds,
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
