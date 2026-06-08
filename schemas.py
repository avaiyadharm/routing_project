#!/usr/bin/env python3
"""
==========================================================================
 VEHICLE ROUTING OPTIMIZER — Expanded Pydantic v2 Request / Response Schemas
 Version: 4.0.0
 Date:    June 2026
==========================================================================

 Covers four new constraint categories added over the baseline v3 schema:

   1. Vehicle & Capacity Constraints  (CVRP support)
   2. Advanced Temporal & Driver Constraints  (shift windows, break rules)
   3. Real-Time & Environmental Features  (XGBoost overlay expansion)
   4. Enterprise / Business Logic  (priority scheduling, dynamic rerouting)

 Usage:
   from schemas import LocationQuery, OptimizeRouteResponse
==========================================================================
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator


# ==========================================================================
#  ENUMERATIONS
# ==========================================================================

class VehicleType(str, Enum):
    """
    Supported vehicle classes.

    Used to:
    - Apply correct speed / load profiles in OR-Tools.
    - Filter city-zone eligibility (e.g. EVs allowed in low-emission zones).
    - Estimate CO₂ / energy consumption for ESG reporting.
    """
    EV_VAN       = "EV_van"
    DIESEL_TRUCK = "diesel_truck"
    PETROL_VAN   = "petrol_van"
    TWO_WHEELER  = "two_wheeler"
    CARGO_BIKE   = "cargo_bike"


class TrafficDensityLevel(str, Enum):
    """
    Qualitative traffic congestion levels.

    Accepts either this enum OR a raw float in [0.0, 1.0].
    Maps to a canonical float before XGBoost inference:
        LOW      → 0.15
        MEDIUM   → 0.40
        HEAVY    → 0.70
        GRIDLOCK → 0.95
    """
    LOW      = "Low"
    MEDIUM   = "Medium"
    HEAVY    = "Heavy"
    GRIDLOCK = "Gridlock"


class RoadTypePreference(str, Enum):
    """
    High-level road-type constraint passed to the OR-Tools arc filter.

    - AVOID_TOLLS      — penalise toll arcs (soft constraint via cost multiplier).
    - HIGHWAYS_ONLY    — restrict arcs to motorway/trunk road classes.
    - SHORTEST_DISTANCE — switch the OR-Tools objective from time to distance.
    """
    AVOID_TOLLS       = "avoid_tolls"
    HIGHWAYS_ONLY     = "highways_only"
    SHORTEST_DISTANCE = "shortest_distance"


# ==========================================================================
#  SUB-MODELS
# ==========================================================================

class WaypointServiceWindow(BaseModel):
    """
    Time-window and service-time specification for a single waypoint.

    All times are expressed as **minutes from the scheduled departure**.
    OR-Tools maps these directly to its AddTimeWindow() and
    NodeDimensionTransit() calls.

    Example:
        Waypoint 0 must be visited between +30 min and +90 min after
        departure, and takes 15 min to unload:
        >>> WaypointServiceWindow(waypoint_index=0,
        ...                       earliest_arrival_min=30,
        ...                       latest_arrival_min=90,
        ...                       service_time_min=15)
    """

    waypoint_index: int = Field(
        ...,
        ge=0,
        description=(
            "0-based index into the `waypoints` array. "
            "Must be < len(waypoints)."
        ),
    )
    earliest_arrival_min: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "Earliest permissible arrival at this waypoint, "
            "in minutes from departure. Default 0 (no lower bound)."
        ),
    )
    latest_arrival_min: float = Field(
        ...,
        gt=0.0,
        description=(
            "Hard deadline for arrival at this waypoint, "
            "in minutes from departure. "
            "OR-Tools treats this as a hard time-window upper bound."
        ),
    )
    service_time_min: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "Fixed dwell time at this node (unloading, signature collection, "
            "etc.), in minutes. Added as a constant transit penalty in the "
            "OR-Tools time dimension."
        ),
    )

    @field_validator("latest_arrival_min")
    @classmethod
    def latest_must_exceed_earliest(cls, v: float, info) -> float:
        earliest = info.data.get("earliest_arrival_min", 0.0)
        if v <= earliest:
            raise ValueError(
                f"latest_arrival_min ({v}) must be strictly greater than "
                f"earliest_arrival_min ({earliest})."
            )
        return v


# ==========================================================================
#  MAIN REQUEST BODY
# ==========================================================================

class LocationQuery(BaseModel):
    """
    Incoming POST payload for ``POST /api/v1/optimize-route``.

    ## Field Groups

    ### Core Routing (unchanged from v3)
    `source`, `destination`, `waypoints`

    ### Baseline Temporal / Weather Context (unchanged from v3)
    `departure_hour`, `day_of_week`, `month`, `is_raining`, `is_festival_zone`

    ### 1 · Vehicle & Capacity Constraints
    `vehicle_type`, `vehicle_capacity_weight`, `vehicle_capacity_volume`,
    `current_load_weight`, `current_load_volume`

    ### 2 · Advanced Temporal & Driver Constraints
    `driver_shift_start`, `driver_shift_end`, `max_driving_hours`,
    `waypoint_service_windows`

    ### 3 · Real-Time & Environmental Features (XGBoost Overlay)
    `traffic_density_index`, `road_closure_count`, `wind_speed`, `visibility`,
    `road_type_preference`

    ### 4 · Enterprise / Business Logic
    `priority_weights`, `allow_dynamic_rerouting`
    """

    # ── ① CORE ROUTING ────────────────────────────────────────────────────

    source: Union[str, List[float]] = Field(
        ...,
        description=(
            "Depot / origin location. "
            "Address string (e.g. ``\"Times Square, NY\"``) "
            "or ``[lat, lon]`` coordinate pair."
        ),
        examples=["Connaught Place, New Delhi", [28.6315, 77.2167]],
    )
    destination: Union[str, List[float]] = Field(
        ...,
        description=(
            "Final destination. "
            "Address string or ``[lat, lon]`` coordinate pair."
        ),
        examples=["Indira Gandhi International Airport, Delhi", [28.5562, 77.1000]],
    )
    waypoints: List[Union[str, List[float]]] = Field(
        default_factory=list,
        description=(
            "Intermediate delivery stops (order is optimised by OR-Tools). "
            "Each element is an address string or ``[lat, lon]`` pair. "
            "Maximum 24 waypoints recommended for solver performance."
        ),
        examples=[["Lajpat Nagar, Delhi", "Nehru Place, Delhi"]],
    )

    # ── ② BASELINE TEMPORAL / WEATHER (v3 fields, unchanged) ─────────────

    departure_hour: int = Field(
        default=12,
        ge=0,
        le=23,
        description="Hour of departure in 24-hour format (0–23).",
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
        description="Calendar month of the trip (1 = January … 12 = December).",
    )
    is_raining: int = Field(
        default=0,
        ge=0,
        le=1,
        description="Binary rain / severe weather flag (0 = clear, 1 = raining).",
    )
    is_festival_zone: int = Field(
        default=0,
        ge=0,
        le=1,
        description="Binary event / festival proximity flag (0 = no, 1 = yes).",
    )

    # ── ③ VEHICLE & CAPACITY CONSTRAINTS (CVRP) ──────────────────────────

    vehicle_type: VehicleType = Field(
        default=VehicleType.DIESEL_TRUCK,
        description=(
            "Vehicle class. Determines speed profile, emission zone eligibility, "
            "and per-km cost coefficients used by the OR-Tools objective function. "
            "``EV_van`` and ``two_wheeler`` unlock low-emission zone access. "
            "``diesel_truck`` applies weight-restricted road filters."
        ),
    )
    vehicle_capacity_weight: Optional[float] = Field(
        default=None,
        gt=0.0,
        description=(
            "Maximum payload weight the vehicle can carry, in **kilograms**. "
            "Enables the OR-Tools capacity dimension (AddDimensionWithVehicleCapacity). "
            "If omitted, capacity constraints are not enforced."
        ),
        examples=[1500.0],
    )
    vehicle_capacity_volume: Optional[float] = Field(
        default=None,
        gt=0.0,
        description=(
            "Total cargo volume available, in **cubic metres (m³)**. "
            "Used alongside `vehicle_capacity_weight` to enforce dual-capacity CVRP. "
            "If omitted, volume constraints are not enforced."
        ),
        examples=[8.5],
    )
    current_load_weight: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "Pre-existing payload weight already on board at the depot (kg). "
            "Initialises the OR-Tools weight dimension counter so that the solver "
            "accounts for the starting utilisation before any pickup/delivery. "
            "Must be ≤ `vehicle_capacity_weight` when both are provided."
        ),
        examples=[200.0],
    )
    current_load_volume: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "Pre-existing cargo volume already on board at the depot (m³). "
            "Same role as `current_load_weight` for the volume dimension. "
            "Must be ≤ `vehicle_capacity_volume` when both are provided."
        ),
        examples=[1.2],
    )

    # ── ④ TEMPORAL & DRIVER CONSTRAINTS ──────────────────────────────────

    driver_shift_start: Optional[str] = Field(
        default=None,
        pattern=r"^([01]\d|2[0-3]):[0-5]\d$",
        description=(
            "Driver shift start time in **HH:MM** (24-hour) format. "
            "OR-Tools uses this as the global time-window lower bound (node 0). "
            "If omitted, no shift-start constraint is applied."
        ),
        examples=["08:00"],
    )
    driver_shift_end: Optional[str] = Field(
        default=None,
        pattern=r"^([01]\d|2[0-3]):[0-5]\d$",
        description=(
            "Driver shift end time in **HH:MM** (24-hour) format. "
            "OR-Tools uses this as the global time-window upper bound "
            "(hard constraint — the route must complete before this time). "
            "If omitted, no shift-end constraint is applied."
        ),
        examples=["17:30"],
    )
    max_driving_hours: Optional[float] = Field(
        default=None,
        gt=0.0,
        le=11.0,
        description=(
            "Maximum continuous driving time before a mandatory break, in **hours**. "
            "Regulatory ceiling for HGV/LGV drivers under EU Regulation (EC) 561/2006 "
            "is 4.5 h; overall daily limit is 9–10 h. "
            "OR-Tools implements this via a cumulative break dimension "
            "(AddBreakIntervalsToVehicle). "
            "Omit if break-scheduling is not required."
        ),
        examples=[4.5],
    )
    waypoint_service_windows: List[WaypointServiceWindow] = Field(
        default_factory=list,
        description=(
            "Per-waypoint time windows and dwell times. "
            "Each entry maps a waypoint index to its arrival window "
            "and fixed service duration. "
            "OR-Tools AddTimeWindow() is called for each entry in this list. "
            "Waypoints without an entry are treated as unconstrained. "
            "Indices must match positions in the `waypoints` array."
        ),
    )

    # ── ⑤ REAL-TIME & ENVIRONMENTAL FEATURES (XGBoost Overlay) ───────────

    traffic_density_index: Union[float, TrafficDensityLevel] = Field(
        default=0.0,
        description=(
            "Current congestion level on the route corridor. "
            "Accepts either:\n"
            "  - A **float** in [0.0, 1.0]: 0.0 = free flow, 1.0 = gridlock.\n"
            "  - A **string enum**: ``Low`` | ``Medium`` | ``Heavy`` | ``Gridlock``\n"
            "    (mapped internally to 0.15 / 0.40 / 0.70 / 0.95).\n"
            "Ingested as feature ``traffic_density_index`` in the XGBoost feature matrix. "
            "A high value increases the predicted travel-time scaling factor."
        ),
        examples=[0.65, "Heavy"],
    )
    road_closure_count: int = Field(
        default=0,
        ge=0,
        le=50,
        description=(
            "Number of confirmed active road closures / disruptions on major links "
            "within the route corridor. "
            "Ingested as feature ``road_closure_count`` in XGBoost. "
            "Additionally used by the OR-Tools arc filter to disqualify closed segments "
            "if individual closure coordinates are provided via future ``closed_segments`` field."
        ),
        examples=[2],
    )
    wind_speed: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=200.0,
        description=(
            "Current wind speed in **km/h**. "
            "High values (> 60 km/h) impair two-wheelers and increase fuel consumption "
            "for high-sided vehicles. "
            "Ingested as feature ``wind_speed_kmh`` in the XGBoost feature matrix "
            "to refine the weather-adjusted travel-time prediction. "
            "Omit if unavailable."
        ),
        examples=[18.5],
    )
    visibility: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=10000.0,
        description=(
            "Atmospheric visibility in **metres**. "
            "Maps to existing ``visibility_meters`` feature in the v2 XGBoost model. "
            "Values < 1000 m trigger fog-speed reductions in the overlay. "
            "Omit if unavailable."
        ),
        examples=[5000.0],
    )
    road_type_preference: RoadTypePreference = Field(
        default=RoadTypePreference.SHORTEST_DISTANCE,
        description=(
            "Strategic road-type preference applied as arc-level constraints in OR-Tools:\n"
            "  - ``avoid_tolls``       — toll arcs receive a large cost penalty.\n"
            "  - ``highways_only``     — non-trunk arcs are forbidden (hard filter).\n"
            "  - ``shortest_distance`` — objective function minimises distance, not time."
        ),
    )

    # ── ⑥ ENTERPRISE / BUSINESS LOGIC ────────────────────────────────────

    priority_weights: List[int] = Field(
        default_factory=list,
        description=(
            "Priority score for each waypoint (same length as `waypoints` or empty). "
            "Higher value = higher business priority (e.g. premium SLA customer). "
            "OR-Tools uses these as **penalty values** on optional nodes: "
            "if the route cannot serve a node within its time window, "
            "the solver drops the lowest-priority nodes first to remain feasible. "
            "Values must be positive integers. "
            "If empty, all waypoints are treated as mandatory (no disjunctions)."
        ),
        examples=[[10, 50, 30, 80]],
    )
    allow_dynamic_rerouting: bool = Field(
        default=False,
        description=(
            "If ``true``, the API response will include a ``rerouting_token`` "
            "that clients can POST back mid-route to trigger a live re-optimisation "
            "with updated traffic/closure data. "
            "If ``false`` (default), the solution is treated as static."
        ),
    )

    # ── CROSS-FIELD VALIDATORS ────────────────────────────────────────────

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
        raise ValueError("Location must be a non-empty string or [lat, lon] list.")

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
                    raise ValueError(f"Waypoint [{idx}] latitude {wp[0]} out of range.")
                if not (-180 <= wp[1] <= 180):
                    raise ValueError(f"Waypoint [{idx}] longitude {wp[1]} out of range.")
                validated.append(wp)
            else:
                raise ValueError(f"Waypoint [{idx}] must be a string or [lat, lon] list.")
        return validated

    @field_validator("traffic_density_index")
    @classmethod
    def validate_traffic_density(
        cls, v: Union[float, TrafficDensityLevel]
    ) -> Union[float, TrafficDensityLevel]:
        """If a raw float, ensure it is in [0.0, 1.0]."""
        if isinstance(v, float) and not (0.0 <= v <= 1.0):
            raise ValueError(
                f"traffic_density_index as a float must be in [0.0, 1.0], got {v}."
            )
        return v

    @model_validator(mode="after")
    def cross_field_checks(self) -> "LocationQuery":
        """Cross-field consistency checks across the full payload."""
        n_wp = len(self.waypoints)

        # 1. priority_weights length must match waypoints (when provided)
        if self.priority_weights and len(self.priority_weights) != n_wp:
            raise ValueError(
                f"priority_weights has {len(self.priority_weights)} entries but "
                f"waypoints has {n_wp}. They must have equal length."
            )

        # 2. All priority values must be positive
        if any(p <= 0 for p in self.priority_weights):
            raise ValueError("All priority_weights values must be positive integers (> 0).")

        # 3. Waypoint service window indices must be within range
        for sw in self.waypoint_service_windows:
            if sw.waypoint_index >= n_wp:
                raise ValueError(
                    f"waypoint_service_windows entry has waypoint_index={sw.waypoint_index} "
                    f"but there are only {n_wp} waypoints (max index {n_wp - 1})."
                )

        # 4. Current load must not exceed vehicle capacity
        if (
            self.vehicle_capacity_weight is not None
            and self.current_load_weight > self.vehicle_capacity_weight
        ):
            raise ValueError(
                f"current_load_weight ({self.current_load_weight} kg) exceeds "
                f"vehicle_capacity_weight ({self.vehicle_capacity_weight} kg)."
            )
        if (
            self.vehicle_capacity_volume is not None
            and self.current_load_volume > self.vehicle_capacity_volume
        ):
            raise ValueError(
                f"current_load_volume ({self.current_load_volume} m³) exceeds "
                f"vehicle_capacity_volume ({self.vehicle_capacity_volume} m³)."
            )

        # 5. Shift window sanity check
        if self.driver_shift_start and self.driver_shift_end:
            sh, sm = map(int, self.driver_shift_start.split(":"))
            eh, em = map(int, self.driver_shift_end.split(":"))
            shift_start_min = sh * 60 + sm
            shift_end_min   = eh * 60 + em
            if shift_end_min <= shift_start_min:
                raise ValueError(
                    f"driver_shift_end ({self.driver_shift_end}) must be after "
                    f"driver_shift_start ({self.driver_shift_start})."
                )

        return self


# ==========================================================================
#  HELPER: Resolve traffic_density_index to canonical float
# ==========================================================================

_DENSITY_ENUM_MAP: dict[TrafficDensityLevel, float] = {
    TrafficDensityLevel.LOW:      0.15,
    TrafficDensityLevel.MEDIUM:   0.40,
    TrafficDensityLevel.HEAVY:    0.70,
    TrafficDensityLevel.GRIDLOCK: 0.95,
}


def resolve_traffic_density(
    value: Union[float, TrafficDensityLevel]
) -> float:
    """
    Convert the polymorphic `traffic_density_index` field to a canonical float.

    Args:
        value: Either a float in [0.0, 1.0] or a TrafficDensityLevel enum.

    Returns:
        A float in [0.0, 1.0].
    """
    if isinstance(value, TrafficDensityLevel):
        return _DENSITY_ENUM_MAP[value]
    return float(value)


# ==========================================================================
#  HELPER: Build XGBoost feature vector from LocationQuery
# ==========================================================================

def build_xgboost_feature_vector(payload: "LocationQuery") -> dict:
    """
    Extract and normalise all XGBoost-relevant fields from the validated payload
    into a named feature dictionary.

    This dict is passed directly to TrafficPredictor.predict_with_features()
    or used to build a numpy feature array for batch prediction.

    Returns:
        dict mapping feature name → float value.
    """
    tdi = resolve_traffic_density(payload.traffic_density_index)

    return {
        # ── Baseline temporal features (v1 model) ──
        "hour_of_day":                payload.departure_hour,
        "day_of_week":                payload.day_of_week,
        "month":                      payload.month,
        "is_raining":                 float(payload.is_raining),
        "is_festival_zone":           float(payload.is_festival_zone),

        # ── New real-time / environmental features (v2 model) ──
        "traffic_density_index":      tdi,
        "road_closure_count":         float(payload.road_closure_count),
        "wind_speed_kmh":             float(payload.wind_speed or 0.0),
        "visibility_meters":          float(payload.visibility or 10_000.0),

        # ── Vehicle-type one-hot encoding ──
        "vehicle_is_ev":              float(payload.vehicle_type == VehicleType.EV_VAN),
        "vehicle_is_two_wheeler":     float(payload.vehicle_type == VehicleType.TWO_WHEELER),
        "vehicle_is_diesel_truck":    float(payload.vehicle_type == VehicleType.DIESEL_TRUCK),

        # ── Capacity utilisation ratio (0.0 → empty, 1.0 → full) ──
        "weight_utilisation_ratio": (
            payload.current_load_weight / payload.vehicle_capacity_weight
            if payload.vehicle_capacity_weight else 0.0
        ),
        "volume_utilisation_ratio": (
            payload.current_load_volume / payload.vehicle_capacity_volume
            if payload.vehicle_capacity_volume else 0.0
        ),
    }


# ==========================================================================
#  UPDATED RESPONSE SCHEMAS
# ==========================================================================

class CapacityStatus(BaseModel):
    """Current capacity utilisation after route planning."""

    weight_remaining_kg:    Optional[float] = None
    volume_remaining_m3:    Optional[float] = None
    weight_utilisation_pct: Optional[float] = None
    volume_utilisation_pct: Optional[float] = None


class ContextApplied(BaseModel):
    """Echo of all contextual parameters applied during this optimisation run."""

    # Temporal
    departure_hour:   int
    day_of_week:      int
    month:            int
    # Weather (v3)
    is_raining:       bool
    is_festival_zone: bool
    # New environmental
    traffic_density_index:  Optional[float] = None
    road_closure_count:     Optional[int]   = None
    wind_speed_kmh:         Optional[float] = None
    visibility_meters:      Optional[float] = None
    # Vehicle
    vehicle_type:           Optional[str]   = None
    road_type_preference:   Optional[str]   = None
    # ML
    ml_scaling_factor:      Optional[float] = Field(
        None,
        description="XGBoost scaling factor applied to the travel matrix. None if skipped.",
    )


class LegDetail(BaseModel):
    """A single inter-node leg within the optimised route."""

    from_location:      str = Field(..., description="Origin node label.")
    to_location:        str = Field(..., description="Destination node label.")
    duration_seconds:   int = Field(..., description="Travel time for this leg (seconds).")
    distance_meters:    Optional[int] = Field(None, description="Physical distance (metres).")


class OptimizeRouteResponse(BaseModel):
    """Structured JSON response from POST /api/v1/optimize-route."""

    status:                         str = Field(default="success")
    geo_backend:                    str
    optimal_sequence_order:         List[str]
    total_optimized_duration_seconds: int
    total_distance_meters:          int
    legs:                           List[LegDetail]
    context_applied:                ContextApplied
    capacity_status:                Optional[CapacityStatus] = None
    geocoded_locations:             List[dict]
    raw_matrix_seconds:             List[List[int]]
    ml_adjusted_matrix_seconds:     List[List[int]]
    dropped_waypoints:              List[str] = Field(
        default_factory=list,
        description=(
            "Labels of waypoints the solver dropped due to priority/time-window infeasibility. "
            "Only populated when priority_weights are provided."
        ),
    )
    rerouting_token:                Optional[str] = Field(
        None,
        description="Opaque token for mid-route re-optimisation. Populated only when allow_dynamic_rerouting=true.",
    )


class HealthResponse(BaseModel):
    """Health check response schema."""

    status:           str  = Field(default="online")
    service:          str  = Field(default="Vehicle Routing Optimizer API")
    version:          str  = Field(default="4.0.0")
    geo_backend:      str
    ml_model_loaded:  bool
    timestamp:        str
