import requests
import json
import pickle
import numpy as np
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

# =====================================================================
# STEP 1: DEFINE COORDINATES (Delaware Map Safe-Zone)
# =====================================================================
# Node 0 is your Depot/Hub. Nodes 1, 2, and 3 are delivery locations.
locations = [
    (39.15, -75.52),  # Depot
    (39.20, -75.60),  # Location 1
    (39.18, -75.55),  # Location 2
    (39.12, -75.58)   # Location 3
]

# =====================================================================
# STEP 2: LOAD BASE DATA FROM LOCAL OSRM SERVER (Port 5001)
# =====================================================================
def get_local_osrm_matrix(coords):
    formatted_coords = ";".join([f"{lon},{lat}" for lat, lon in coords])
    url = f"http://localhost:5001/table/v1/driving/{formatted_coords}?annotations=duration"
    response = requests.get(url)
    return [[int(seconds) for seconds in row] for row in response.json()['durations']]

# =====================================================================
# STEP 3: INTEGRATE THE TRAINED XGBOOST MODEL
# =====================================================================
print("💾 Loading trained XGBoost Traffic Model...")
with open("traffic_xgb_model.pkl", "rb") as f:
    trained_ml_model = pickle.load(f)

def apply_real_ml_predictions(base_matrix, hour, day, month, raining, festival):
    """
    Takes base travel times from OSRM and uses your trained XGBoost model
    to adapt the values dynamically based on environment conditions.
    """
    adjusted_matrix = []
    
    for row_idx, row in enumerate(base_matrix):
        new_row = []
        for col_idx, base_time in enumerate(row):
            if row_idx == col_idx:
                new_row.append(0)
                continue
            
            # Match the exact feature sequence the model learned:
            # ['hour_of_day', 'day_of_week', 'month', 'is_raining', 'is_festival_zone']
            input_features = np.array([[hour, day, month, raining, festival]], dtype=float)
            
            # Predict trip duration multiplier/value from your trained ML model
            predicted_time = trained_ml_model.predict(input_features)[0]
            
            # Safety mechanism: ensure dynamic time never drops below physical layout limit
            final_time = max(int(predicted_time), int(base_time))
            new_row.append(final_time)
            
        adjusted_matrix.append(new_row)
    return adjusted_matrix

# Fetch baseline data from OSRM
base_time_matrix = get_local_osrm_matrix(locations)

# =====================================================================
# CONFIGURATION: SET YOUR TRAFFIC CONTEXT HERE
# =====================================================================
# Let's simulate a bad traffic scenario: 6:00 PM (18), Friday (4), June (6), Raining, Festival Active
CURRENT_HOUR = 18          
DAY_OF_WEEK = 4            
MONTH = 6                  
IS_RAINING = 1             # 1 = Yes, 0 = No
IS_FESTIVAL_ZONE = 1       # 1 = Yes, 0 = No

print(f"\n🧠 Running ML Predictions for Context: Hour={CURRENT_HOUR}, Raining={IS_RAINING}, Festival={IS_FESTIVAL_ZONE}...")
ml_predicted_matrix = apply_real_ml_predictions(
    base_time_matrix, CURRENT_HOUR, DAY_OF_WEEK, MONTH, IS_RAINING, IS_FESTIVAL_ZONE
)

print("\n🚨 Traffic-Adjusted Matrix passed to OR-Tools:")
for row in ml_predicted_matrix:
    print(row)

# =====================================================================
# STEP 4: SOLVE THE OPTIMAL ROUTE SEQUENCING WITH GOOGLE OR-TOOLS
# =====================================================================
def solve_routing_problem(matrix):
    manager = pywrapcp.RoutingIndexManager(len(matrix), 1, 0)
    routing = pywrapcp.RoutingModel(manager)

    def time_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return matrix[from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(time_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )

    solution = routing.SolveWithParameters(search_parameters)

    if solution:
        print("\n=============================================")
        print("🎉 TRAFFIC-OPTIMIZED SEQUENCE FOUND 🎉")
        print("=============================================")
        index = routing.Start(0)
        route_sequence = []
        while not routing.IsEnd(index):
            route_sequence.append(manager.IndexToNode(index))
            index = solution.Value(routing.NextVar(index))
        route_sequence.append(manager.IndexToNode(index))
        
        print("👉 Best Order to Visit:", " ➔ ".join(map(str, route_sequence)))
        print(f"⏱️ Total Predicted Trip Duration: {solution.ObjectiveValue()} seconds")
        print("=============================================")

solve_routing_problem(ml_predicted_matrix)