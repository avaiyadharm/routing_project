"""
Feature Engineering Pipeline for ML-Driven Traffic Cost Function
Computes 50+ contextual features for dynamic edge weight calculation in vehicle routing.

Categories:
1. Temporal: holidays, school calendar, payday cycles, time-of-day encoding
2. Meteorological: precipitation, visibility, road surface states (Phase 2)
3. Events: proximity to stadiums/concerts/festivals (Phase 2)
4. Infrastructure: lane closures, dynamic speed limits, railway crossings (Phase 3)
5. Topological: road classification, signalization density, turn penalties (Phase 3)
6. Vehicle-specific: dimensions, performance characteristics (Phase 3)
"""

from datetime import datetime, timedelta
import math
from typing import Dict, Tuple, List, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Central hub for computing contextual features for traffic prediction."""

    def __init__(self, region: str = 'delaware'):
        """Initialize feature engineer with regional context.

        Args:
            region: Geographic region for holidays/school calendars ('delaware', 'us', etc.)
        """
        self.region = region
        self._initialize_calendar_data()

    def _initialize_calendar_data(self):
        """Initialize holiday and school calendar data for the region."""
        # US National holidays (2026)
        self.holidays_2026 = [
            (1, 1),   # New Year
            (1, 19),  # MLK Jr. Day
            (2, 16),  # Presidents Day
            (5, 25),  # Memorial Day
            (7, 4),   # Independence Day
            (9, 7),   # Labor Day
            (10, 12), # Columbus Day
            (11, 26), # Thanksgiving
            (11, 27), # Day After Thanksgiving
            (12, 25), # Christmas
        ]

        # Delaware-specific school calendar (2025-2026 academic year)
        # School runs from mid-August 2025 through mid-June 2026
        # For dates in 2026, semester is Aug-Jun of the school year
        self.school_calendar = {
            'semester_start_month': 8,   # August
            'semester_end_month': 6,     # June
            'breaks': [
                ((11, 24), (11, 28)),   # Thanksgiving break
                ((12, 22), (1, 5)),     # Winter break
                ((3, 16), (3, 20)),     # Spring break
            ]
        }

        # Payday cycles (US standard)
        self.paydays = [1, 15, -1]  # 1st, 15th, last day of month

    # ==================== PHASE 1: TEMPORAL FEATURES ====================

    def get_holiday_state(self, date: datetime, region: str = 'us') -> int:
        """Classify day as regular, holiday, bridge day, or holiday-adjacent.

        Returns:
            0 = regular weekday
            1 = eve-of-holiday surge window (6-48 hrs pre-holiday)
            2 = holiday or holiday-adjacent
            3 = bridge day (day between weekend and holiday)
            4 = post-holiday rebound (24-72 hrs post-holiday)
        """
        month, day = date.month, date.day
        current_date = (month, day)

        # Check if today is a holiday
        if current_date in self.holidays_2026:
            return 2

        # Check eve-of-holiday window (up to 2 days before)
        for holiday in self.holidays_2026:
            holiday_date = datetime(2026, holiday[0], holiday[1])
            days_to_holiday = (holiday_date - date).days
            if 0 < days_to_holiday <= 2:
                return 1

        # Check post-holiday rebound (up to 3 days after)
        for holiday in self.holidays_2026:
            holiday_date = datetime(2026, holiday[0], holiday[1])
            days_from_holiday = (date - holiday_date).days
            if 0 < days_from_holiday <= 3:
                return 4

        # Check bridge days (days between weekends and holidays)
        weekday = date.weekday()
        next_date = date + timedelta(days=1)
        prev_date = date - timedelta(days=1)

        if weekday == 4:  # Friday
            next_month, next_day = next_date.month, next_date.day
            if (next_month, next_day) in self.holidays_2026:
                return 3
        elif weekday == 0:  # Monday
            prev_month, prev_day = prev_date.month, prev_date.day
            if (prev_month, prev_day) in self.holidays_2026:
                return 3

        return 0  # Regular day

    def get_school_calendar_state(self, date: datetime) -> Tuple[int, float]:
        """Determine if school is in session and return phase encoding.

        Returns:
            (school_season_state: 0=break, 1=semester,
             school_phase: progress through academic year [-1, 1])
        """
        month, day = date.month, date.day
        current_date = (month, day)

        # Check if in explicit break period first
        for break_start, break_end in self.school_calendar['breaks']:
            if self._date_in_range(current_date, break_start, break_end):
                return 0, -1.0  # In break, return immediately

        # For 2026, school calendar runs Aug (prev year) through June (current year)
        # So: Dec-June 2026 is in semester, July-Nov is break
        in_semester = 8 <= month <= 12 or 1 <= month <= 6

        school_season = 1 if in_semester else 0

        # Compute school year phase based on position in academic year (Aug-June)
        if in_semester:
            # Map to 0-180 day progress (Aug=0, June=180)
            if month >= 8:  # Aug-Dec
                days_into_semester = self._to_day_of_year(month, day) - self._to_day_of_year(8, 1)
            else:  # Jan-June
                # Count from Aug of previous year
                days_into_semester = (self._to_day_of_year(month, day) +
                                      (365 - self._to_day_of_year(8, 1)))
            school_year_phase = (days_into_semester / 180.0) * 2 - 1  # [-1, 1]
        else:
            school_year_phase = -1.0  # Off-season

        return school_season, school_year_phase

    def get_school_hours_active(self, hour: int, school_season: int) -> bool:
        """Check if current hour is during peak school activity hours.

        School hours impact:
        - 7:30-9:00 AM: drop-off rush
        - 2:30-4:00 PM: pick-up rush
        - Other school hours (9 AM - 2:30 PM): parental errands
        """
        if school_season == 0:  # School is on break
            return False

        return (7 <= hour < 15)  # Broad school-influenced hours

    def get_payday_cycle(self, date: datetime) -> Tuple[float, float]:
        """Compute days to next payday and cyclical encoding.

        Paydays typically occur on 1st and 15th of month (salaried workers).
        Returns:
            (days_to_payday: signed distance,
             payday_cycle_sin: sinusoidal encoding)
        """
        month, day = date.month, date.day

        # Find nearest payday
        payday_dates = [1, 15]

        # Check 15th of current month
        days_to_15 = 15 - day if day < 15 else 15 - day + 30

        # Check 1st of next month
        days_to_next_1st = 1 - day + 30 if day != 1 else 30

        # Find minimum
        candidates = []
        for pd in payday_dates:
            if pd == day:
                days_to = 0
            elif pd > day:
                days_to = pd - day
            else:
                days_to = (30 - day) + pd
            candidates.append(days_to)

        min_days_to_payday = min(candidates)

        # Apply decay: peak multiplier at payday, decay over ~5 days
        payday_cycle_sin = math.sin(2 * math.pi * (day % 30) / 30)

        return float(min_days_to_payday), payday_cycle_sin

    def encode_cyclical_features(self, hour: int, day_of_year: int) -> Dict[str, float]:
        """Encode circular/cyclical temporal features using sinusoidal encoding.

        Sinusoidal encoding preserves circularity (e.g., 23:59 → 00:00 is nearby).

        Args:
            hour: Hour of day [0-23]
            day_of_year: Day of calendar year [1-365]

        Returns:
            Dictionary with sin/cos pairs for each cyclical feature
        """
        # Hour encoding (24-hour cycle)
        hour_sin = math.sin(2 * math.pi * hour / 24)
        hour_cos = math.cos(2 * math.pi * hour / 24)

        # Day-of-year encoding (365-day cycle)
        doy_sin = math.sin(2 * math.pi * day_of_year / 365)
        doy_cos = math.cos(2 * math.pi * day_of_year / 365)

        return {
            'hour_sin': hour_sin,
            'hour_cos': hour_cos,
            'day_of_year_sin': doy_sin,
            'day_of_year_cos': doy_cos,
        }

    # ==================== PHASE 2: METEOROLOGICAL FEATURES ====================

    def get_precipitation_multiplier(self, intensity_mmhr: float) -> float:
        """Compute traffic cost multiplier based on precipitation intensity.

        Non-linear relationship: drizzle has minimal effect, heavy rain cascades.

        Args:
            intensity_mmhr: Precipitation intensity in mm/hour

        Returns:
            Cost multiplier [1.0, 2.5+]
        """
        if intensity_mmhr < 0.5:
            return 1.0  # Dry
        elif intensity_mmhr < 2:
            return 1.0 + 0.05  # Light drizzle
        elif intensity_mmhr < 10:
            # Linear interpolation 2-10 mm/hr
            return 1.05 + 0.20 * (intensity_mmhr - 2) / 8
        elif intensity_mmhr < 20:
            # Steeper increase 10-20 mm/hr
            return 1.25 + 0.25 * (intensity_mmhr - 10) / 10
        else:
            # Severe: log saturation for extreme rainfall
            return 1.50 + 0.45 * math.log(intensity_mmhr / 20)

    def get_visibility_multiplier(self, visibility_m: float) -> float:
        """Compute cost multiplier based on visibility degradation.

        Args:
            visibility_m: Visibility distance in meters (100-50000)

        Returns:
            Cost multiplier [1.0, 1.6]
        """
        if visibility_m >= 10000:
            return 1.0
        elif visibility_m >= 5000:
            return 1.08 + 0.04 * (10000 - visibility_m) / 5000
        elif visibility_m >= 1000:
            return 1.12 + 0.23 * (5000 - visibility_m) / 4000
        else:
            # Severe fog: low visibility hazard
            return 1.35 + 0.25 * (1000 - visibility_m) / 1000

    def compute_glare_risk(self, road_bearing_degrees: float,
                          sun_azimuth_degrees: float, hour: int) -> float:
        """Compute glare risk on east-west oriented roads during sun rise/set.

        Args:
            road_bearing_degrees: Road bearing [0-360]
            sun_azimuth_degrees: Solar azimuth angle [0-360]
            hour: Current hour [0-23]

        Returns:
            Glare risk indicator [0, 1]
        """
        # Sunrise glare: 6-8 AM, eastern roads
        # Sunset glare: 4-6 PM, western roads

        if not ((6 <= hour <= 8) or (16 <= hour <= 18)):
            return 0.0

        # Compute angle difference (shortest path on circle)
        angle_diff = abs(road_bearing_degrees - sun_azimuth_degrees)
        if angle_diff > 180:
            angle_diff = 360 - angle_diff

        # Maximum glare when perpendicular (90°), minimum at parallel (0° or 180°)
        # Use sine since glare peaks at 90° perpendicularity
        glare = max(0, math.sin(math.radians(angle_diff)))

        return glare

    def compute_road_surface_state(self, rain_accumulation_mmhr_3h: float,
                                  elevation_grade_percent: float = 0,
                                  urban_density: float = 0.5) -> int:
        """Compute road surface state ordinal based on moisture and drainage.

        Args:
            rain_accumulation_mmhr_3h: Cumulative rain over past 3 hours
            elevation_grade_percent: Road grade percentage (slope)
            urban_density: Urban density factor [0, 1] affecting drainage

        Returns:
            Surface state: 0=dry, 1=damp, 2=wet_standing, 3=hydroplaning_risk
        """
        if rain_accumulation_mmhr_3h < 1:
            return 0  # Dry

        # Drainage efficiency: steeper grades drain faster, rural areas drain slower
        drainage_efficiency = (abs(elevation_grade_percent) / 5.0) + (1 - urban_density) * 0.5
        surface_wetness = rain_accumulation_mmhr_3h / (drainage_efficiency + 0.3)

        if surface_wetness < 3:
            return 1  # Damp
        elif surface_wetness < 12:
            return 2  # Wet standing water
        else:
            return 3  # Hydroplaning risk

    # ==================== PHASE 3: EVENT FEATURES ====================

    def get_event_proximity_decay(self, edge_distance_km: float,
                                  event_radius_km: float) -> float:
        """Compute event impact using Gaussian spatial decay.

        Args:
            edge_distance_km: Distance from edge to event epicenter
            event_radius_km: Effective radius of event impact zone

        Returns:
            Impact multiplier [1.0, 1.5]
        """
        if edge_distance_km > event_radius_km:
            return 1.0  # Outside impact zone

        # Gaussian decay within radius
        normalized_distance = edge_distance_km / event_radius_km
        gaussian = math.exp(-3 * normalized_distance ** 2)

        return 1.0 + 0.5 * gaussian  # [1.0, 1.5]

    def get_event_phase(self, current_timestamp: datetime,
                       event_start: datetime,
                       event_end: datetime) -> int:
        """Determine event phase relative to current time.

        Event phases have distinct traffic patterns:
        - Pre-event: anticipatory traffic
        - Crush window: peak ingress
        - Post-event: mass egress

        Returns:
            Phase: 0=pre_buffer, 1=early_arrivals, 2=crush, 3=egress, 4=normalized
        """
        minutes_to_event = (event_start - current_timestamp).total_seconds() / 60
        minutes_since_start = (current_timestamp - event_start).total_seconds() / 60

        if minutes_to_event < -120:
            return 4  # >2 hours before: normalized
        elif minutes_to_event > 0:
            if minutes_to_event > 30:
                return 0  # Early buffer
            else:
                return 1  # Early arrivals (last 30 min before start)
        elif minutes_since_start < 90:
            return 3  # Post-event egress (up to 90 min after)
        else:
            return 4  # Normalized


    # ==================== PHASE 4: INFRASTRUCTURE AND LIVE URBAN DYNAMICS ====================

    def get_disruption_cost_multiplier(self, disruptions: List[Dict]) -> float:
        """Compute multiplicative cost adjustments based on active disruptions.
        
        Args:
            disruptions: List of disruption dicts with 'type' and specific properties.
                Types: 'lane_closure', 'construction', 'parking_friction'
        
        Returns:
            Cost multiplier >= 1.0
        """
        total_cost = 1.0
        for disruption in disruptions:
            dtype = disruption.get('type')
            if dtype == 'lane_closure':
                closed = disruption.get('closed_lanes', 0)
                total = disruption.get('total_lanes', 1)
                cost = 1.0 + 0.5 * (closed / total) if total > 0 else 1.0
            elif dtype == 'construction':
                intensity = disruption.get('intensity_0_to_1', 0.5)
                cost = 1.0 + 0.3 + 0.5 * intensity
            elif dtype == 'parking_friction':
                occupancy = disruption.get('occupancy_ratio', 0.5)
                cost = 1.0 + 0.15 + 0.1 * occupancy
            else:
                cost = 1.0
            
            total_cost *= cost  # Multiplicative composition
        
        return total_cost

    def get_vsl_and_pricing_cost(self, vsl_mph: Optional[float], base_speed_mph: float, 
                                 is_pricing_zone: bool, is_peak_hour: bool) -> float:
        """Adjust cost for dynamic speed limits and congestion pricing zones.
        
        Args:
            vsl_mph: Variable speed limit (if active, else None)
            base_speed_mph: Base speed limit of the edge
            is_pricing_zone: True if edge is in active congestion pricing zone
            is_peak_hour: True if current time is peak hour for pricing
            
        Returns:
            Cost multiplier
        """
        # Variable speed limit multiplier
        if vsl_mph and vsl_mph > 0 and base_speed_mph > 0:
            speed_multiplier = base_speed_mph / vsl_mph
        else:
            speed_multiplier = 1.0
            
        # Congestion pricing penalty
        if is_pricing_zone and is_peak_hour:
            pricing_multiplier = 1.3  # 30% added friction
        else:
            pricing_multiplier = 1.0
            
        return speed_multiplier * pricing_multiplier

    def get_railway_crossing_cost(self, has_crossing: bool, is_train_approaching: bool, 
                                  has_drawbridge: bool, drawbridge_opening: bool) -> float:
        """Implements penalties for active railway crossings and drawbridge schedules.
        
        Args:
            has_crossing: True if edge has railway crossing
            is_train_approaching: True if train is approaching or crossing
            has_drawbridge: True if edge is on a drawbridge
            drawbridge_opening: True if drawbridge is currently opening/open
            
        Returns:
            Cost multiplier
        """
        multiplier = 1.0
        if has_crossing and is_train_approaching:
            multiplier *= 2.5  # Significant delay expected
            
        if has_drawbridge and drawbridge_opening:
            multiplier *= 4.0  # Huge delay expected
            
        return multiplier

    # ==================== PHASE 5: MICRO-SPATIAL AND TOPOLOGICAL ATTRIBUTES ====================

    def get_road_classification_cost(self, classification: str, num_lanes: int) -> float:
        """Returns a capacity multiplier based on road hierarchies and lane counts.
        
        Args:
            classification: OSM road classification (motorway, trunk, primary, etc.)
            num_lanes: Number of lanes
            
        Returns:
            Cost multiplier
        """
        default_lanes = {
            'motorway': 3,
            'trunk': 2,
            'primary': 2,
            'secondary': 2,
            'tertiary': 1,
            'residential': 1,
            'service': 1,
            'living_street': 1
        }
        
        typical_lanes = default_lanes.get(classification, 1)
        
        # More lanes -> lower multiplier (faster)
        lane_multiplier = 1.0 - 0.1 * ((num_lanes - typical_lanes) / typical_lanes)
        
        # Bound the multiplier to reasonable limits
        return max(0.6, min(1.5, lane_multiplier))

    def get_signalization_cost(self, signals_per_km: float) -> float:
        """Multiplier representing traffic light density delays.
        
        Args:
            signals_per_km: Number of traffic signals per kilometer
            
        Returns:
            Cost multiplier >= 1.0
        """
        if signals_per_km < 2:
            multiplier = 1.0
        elif signals_per_km < 4:
            multiplier = 1.0 + 0.125 * (signals_per_km - 2) / 2
        elif signals_per_km < 6:
            multiplier = 1.125 + 0.2 * (signals_per_km - 4) / 2
        else:
            multiplier = 1.325 + 0.25 * min((signals_per_km - 6) / 2, 1.0)
            
        return multiplier

    def get_turn_penalty(self, turn_angle: float, is_protected: bool = False, 
                         street_width_m: float = 10.0, has_signal: bool = False) -> float:
        """Cost penalty for turns, applying severe friction to unprotected left turns.
        
        Args:
            turn_angle: Angle of the turn in degrees [-180, +180]
            is_protected: True if turn is protected by signal/lane
            street_width_m: Width of the street turning into
            has_signal: True if intersection has signal
            
        Returns:
            Cost multiplier >= 1.0
        """
        # Normalize angle to [-180, 180]
        while turn_angle <= -180: turn_angle += 360
        while turn_angle > 180: turn_angle -= 360
            
        if abs(turn_angle) < 10:
            base_multiplier = 1.0  # straight
        elif 10 <= turn_angle < 90:
            base_multiplier = 1.08  # right_turn
        elif -90 < turn_angle <= -10:
            base_multiplier = 1.15 if is_protected else 1.40  # left_turn
        else:
            base_multiplier = 2.0  # u_turn
            
        # Radius penalty for narrow streets
        radius_penalty = 1.0 + 0.2 / max(street_width_m / 10, 1.0)
        
        # Signal penalty
        if has_signal:
            base_multiplier *= 1.2
            
        return base_multiplier * radius_penalty

    # ==================== PHASE 6: FLEET AND VEHICLE-SPECIFIC CONSTRAINTS ====================

    def get_vehicle_dimension_compatibility(self, vehicle_height: float, vehicle_weight: float, 
                                            edge_max_height: Optional[float], 
                                            edge_max_weight: Optional[float]) -> Tuple[bool, float]:
        """Checks hard constraints and soft penalties for clearance margins.
        
        Args:
            vehicle_height: Height of vehicle in meters
            vehicle_weight: Weight of vehicle in kg
            edge_max_height: Max height limit of edge
            edge_max_weight: Max weight limit of edge
            
        Returns:
            (is_traversable, penalty_multiplier)
        """
        if edge_max_height and vehicle_height > edge_max_height:
            return False, float('inf')
            
        if edge_max_weight and vehicle_weight > edge_max_weight:
            return False, float('inf')
            
        penalty = 1.0
        if edge_max_height:
            clearance_margin = edge_max_height - vehicle_height
            if clearance_margin < 1.0:
                penalty = 1.0 + (1.0 - clearance_margin) * 0.5
                
        return True, penalty

    def get_vehicle_type_performance(self, vehicle_type: str, grade_percent: float, 
                                     vehicle_turn_radius: float, min_turn_radius: float, 
                                     road_type: str, accel_efficiency: float, 
                                     signal_density: float) -> float:
        """Modifies edge costs based on vehicle gradient tolerances and agility.
        
        Args:
            vehicle_type: 'scooter', 'e_bike', 'delivery_van', 'heavy_truck'
            grade_percent: Grade of road %
            vehicle_turn_radius: Min turn radius of vehicle
            min_turn_radius: Min turn radius allowed on road
            road_type: Classification of road
            accel_efficiency: [0.2, 1.0] factor
            signal_density: Signals per km
            
        Returns:
            Cost multiplier >= 1.0
        """
        # Grade penalty
        if vehicle_type == 'scooter':
            grade_multiplier = 1.0
        elif vehicle_type == 'e_bike':
            grade_multiplier = 1.0 + 0.05 * max(grade_percent, 0)
        elif vehicle_type == 'delivery_van':
            grade_multiplier = 1.0 + 0.15 * max(grade_percent, 0)
        elif vehicle_type == 'heavy_truck':
            grade_multiplier = 1.0 + 0.45 * max(grade_percent, 0)
        else:
            grade_multiplier = 1.0 + 0.1 * max(grade_percent, 0)
            
        # Turn radius compatibility
        if vehicle_turn_radius > min_turn_radius:
            if road_type in ['residential', 'service', 'living_street']:
                if vehicle_turn_radius > min_turn_radius * 1.5:
                    return float('inf')  # Untraversable
                else:
                    traverse_penalty = 1.0 + 0.3 * (vehicle_turn_radius / min_turn_radius - 1)
            else:
                traverse_penalty = 1.0
        else:
            traverse_penalty = 1.0
            
        # Acceleration penalty
        accel_penalty = 1.0 + (1 - accel_efficiency) * 0.2 * (signal_density / 5)
        
        return grade_multiplier * traverse_penalty * accel_penalty

    # ==================== UTILITY METHODS ====================

    def build_temporal_feature_dict(self, timestamp: datetime) -> Dict[str, float]:
        """Construct all temporal features for a given timestamp.

        Returns:
            Dictionary with all temporal features ready for ML model
        """
        hour = timestamp.hour
        day_of_week = timestamp.weekday()
        month = timestamp.month
        day = timestamp.day
        day_of_year = self._to_day_of_year(month, day)

        # Temporal features
        temporal_features = {
            'hour_of_day': float(hour),
            'day_of_week': float(day_of_week),
            'month': float(month),
            'day_of_month': float(day),
        }

        # Holiday state
        holiday_state = self.get_holiday_state(timestamp)
        temporal_features['holiday_state_ordinal'] = float(holiday_state)

        # School calendar
        school_season, school_phase = self.get_school_calendar_state(timestamp)
        temporal_features['school_season_state'] = float(school_season)
        temporal_features['school_phase'] = school_phase
        temporal_features['school_hours_active'] = float(int(
            self.get_school_hours_active(hour, school_season)
        ))

        # Payday cycle
        days_to_payday, payday_sin = self.get_payday_cycle(timestamp)
        temporal_features['days_to_payday'] = days_to_payday
        temporal_features['payday_cycle_sin'] = payday_sin

        # Sinusoidal encodings
        cyclical = self.encode_cyclical_features(hour, day_of_year)
        temporal_features.update(cyclical)

        return temporal_features

    @staticmethod
    def _to_day_of_year(month: int, day: int) -> int:
        """Convert month/day to day-of-year [1-365]."""
        days_per_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        return sum(days_per_month[:month-1]) + day

    @staticmethod
    def _date_in_range(current_date: Tuple[int, int],
                      start_date: Tuple[int, int],
                      end_date: Tuple[int, int]) -> bool:
        """Check if current_date is between start_date and end_date."""
        current_doy = FeatureEngineer._to_day_of_year(current_date[0], current_date[1])
        start_doy = FeatureEngineer._to_day_of_year(start_date[0], start_date[1])
        end_doy = FeatureEngineer._to_day_of_year(end_date[0], end_date[1])

        return start_doy <= current_doy <= end_doy
