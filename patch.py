import os

feature_file = "/Users/admin/Desktop/routing_project/src/feature_engineering.py"

phases_code = """
    # ==================== PHASE 4: INFRASTRUCTURE AND LIVE URBAN DYNAMICS ====================

    def get_disruption_cost_multiplier(self, disruptions: List[Dict]) -> float:
        \"\"\"Compute multiplicative cost adjustments based on active disruptions.
        
        Args:
            disruptions: List of disruption dicts with 'type' and specific properties.
                Types: 'lane_closure', 'construction', 'parking_friction'
        
        Returns:
            Cost multiplier >= 1.0
        \"\"\"
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
        \"\"\"Adjust cost for dynamic speed limits and congestion pricing zones.
        
        Args:
            vsl_mph: Variable speed limit (if active, else None)
            base_speed_mph: Base speed limit of the edge
            is_pricing_zone: True if edge is in active congestion pricing zone
            is_peak_hour: True if current time is peak hour for pricing
            
        Returns:
            Cost multiplier
        \"\"\"
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
        \"\"\"Implements penalties for active railway crossings and drawbridge schedules.
        
        Args:
            has_crossing: True if edge has railway crossing
            is_train_approaching: True if train is approaching or crossing
            has_drawbridge: True if edge is on a drawbridge
            drawbridge_opening: True if drawbridge is currently opening/open
            
        Returns:
            Cost multiplier
        \"\"\"
        multiplier = 1.0
        if has_crossing and is_train_approaching:
            multiplier *= 2.5  # Significant delay expected
            
        if has_drawbridge and drawbridge_opening:
            multiplier *= 4.0  # Huge delay expected
            
        return multiplier

    # ==================== PHASE 5: MICRO-SPATIAL AND TOPOLOGICAL ATTRIBUTES ====================

    def get_road_classification_cost(self, classification: str, num_lanes: int) -> float:
        \"\"\"Returns a capacity multiplier based on road hierarchies and lane counts.
        
        Args:
            classification: OSM road classification (motorway, trunk, primary, etc.)
            num_lanes: Number of lanes
            
        Returns:
            Cost multiplier
        \"\"\"
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
        \"\"\"Multiplier representing traffic light density delays.
        
        Args:
            signals_per_km: Number of traffic signals per kilometer
            
        Returns:
            Cost multiplier >= 1.0
        \"\"\"
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
        \"\"\"Cost penalty for turns, applying severe friction to unprotected left turns.
        
        Args:
            turn_angle: Angle of the turn in degrees [-180, +180]
            is_protected: True if turn is protected by signal/lane
            street_width_m: Width of the street turning into
            has_signal: True if intersection has signal
            
        Returns:
            Cost multiplier >= 1.0
        \"\"\"
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
        \"\"\"Checks hard constraints and soft penalties for clearance margins.
        
        Args:
            vehicle_height: Height of vehicle in meters
            vehicle_weight: Weight of vehicle in kg
            edge_max_height: Max height limit of edge
            edge_max_weight: Max weight limit of edge
            
        Returns:
            (is_traversable, penalty_multiplier)
        \"\"\"
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
        \"\"\"Modifies edge costs based on vehicle gradient tolerances and agility.
        
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
        \"\"\"
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
"""

with open(feature_file, "r") as f:
    lines = f.readlines()

# find where utility methods starts
util_idx = -1
for i, line in enumerate(lines):
    if "==================== UTILITY METHODS ====================" in line:
        util_idx = i
        break

if util_idx != -1:
    lines.insert(util_idx, phases_code + "\n")
    with open(feature_file, "w") as f:
        f.writelines(lines)
    print("Successfully patched feature_engineering.py")
else:
    print("Could not find UTILITY METHODS section")
