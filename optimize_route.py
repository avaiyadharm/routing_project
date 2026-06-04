import requests
import json
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

# =====================================================================
# STEP 1: DEFINE REAL COORDINATES (Within your downloaded Delaware Map)
# =====================================================================
# Node 0 is our Hub/Depot. Nodes 1, 2, and 3 are customer drop-offs.
locations = [
    (39.15, -75.52),  # Depot (Start & End location)
    (39.20, -75.60),  # Location 1
    (39.18, -75.55),  # Location 2
    (39.12, -75.58)   # Location 3
]

# =====================================================================
# STEP 2: GET DYNAMIC MATRIX FROM YOUR LOCAL OSRM SERVER (Port 5001)
# =====================================================================
def get_local_osrm_matrix(coords):
    # OSRM expects: "long,lat;long,lat;long,lat"
    formatted_coords = ";".join([f"{lon},{lat}" for lat, lon in coords])
    
    # Notice we are hitting port 5001 here!
    url = f"http://localhost:5001/table/v1/driving/{formatted_coords}?annotations=duration"
    
    response = requests.get(url)
    data = response.json()
    
    # OSRM returns floats in seconds. We convert to integers for OR-Tools processing.
    time_matrix = [[int(seconds) for seconds in row] for row in data['durations']]
    return time_matrix

print("Connecting to local Docker OSRM on Port 5001...")
travel_time_matrix = get_local_osrm_matrix(locations)

print("\nGenerated Real Travel Time Matrix (in seconds):")
for row in travel_time_matrix:
    print(row)

# =====================================================================
# STEP 3: CONSTRAIN & OPTIMIZE VISITING ORDER WITH GOOGLE OR-TOOLS
# =====================================================================
def solve_routing_problem(matrix):
    # Setup index manager: (Total Locations, Number of Vehicles, Depot Index)
    manager = pywrapcp.RoutingIndexManager(len(matrix), 1, 0)
    routing = pywrapcp.RoutingModel(manager)

    # Define cost function callback between nodes
    def time_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return matrix[from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(time_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Use default search parameters
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )

    # Compute optimization
    solution = routing.SolveWithParameters(search_parameters)

    # Output details
    if solution:
        print("\n=============================================")
        print("🎉 OPTIMAL SOLUTION TRAFFIC SEQUENCE FOUND 🎉")
        print("=============================================")
        index = routing.Start(0)
        route_sequence = []
        while not routing.IsEnd(index):
            route_sequence.append(manager.IndexToNode(index))
            index = solution.Value(routing.NextVar(index))
        route_sequence.append(manager.IndexToNode(index))
        
        print("👉 Best Order to Visit:", " ➔ ".join(map(str, route_sequence)))
        print(f"⏱️ Total Optimized Multi-Stop Trip Time: {solution.ObjectiveValue()} seconds")
        print("=============================================")

solve_routing_problem(travel_time_matrix)