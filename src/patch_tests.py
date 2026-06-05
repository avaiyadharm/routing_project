import os

test_file = "/Users/admin/Desktop/routing_project/tests/test_feature_engineering.py"

phases_test_code = """
class TestInfrastructureFeatures:
    @pytest.fixture
    def engineer(self):
        return FeatureEngineer(region='delaware')

    def test_disruption_cost_multiplier(self, engineer):
        disruptions = [{'type': 'lane_closure', 'closed_lanes': 1, 'total_lanes': 2}]
        cost = engineer.get_disruption_cost_multiplier(disruptions)
        assert cost == 1.25

    def test_vsl_and_pricing_cost(self, engineer):
        cost = engineer.get_vsl_and_pricing_cost(30.0, 60.0, True, True)
        assert cost == 2.6  # (60/30) * 1.3

    def test_railway_crossing_cost(self, engineer):
        cost = engineer.get_railway_crossing_cost(True, True, True, True)
        assert cost == 10.0  # 2.5 * 4.0


class TestTopologicalFeatures:
    @pytest.fixture
    def engineer(self):
        return FeatureEngineer(region='delaware')

    def test_road_classification_cost(self, engineer):
        cost = engineer.get_road_classification_cost('motorway', 3)
        assert cost == 1.0

    def test_signalization_cost(self, engineer):
        cost = engineer.get_signalization_cost(5.0)
        assert cost == 1.225

    def test_turn_penalty(self, engineer):
        cost = engineer.get_turn_penalty(90, False, 10.0, False)
        # turn=90 -> U-turn (base 2.0). However in logic: -90 < angle <= -10 is left, angle >= 90 is U-turn.
        # radius_penalty for 10m = 1.2
        # total = 2.0 * 1.2 = 2.4
        assert cost == 2.4


class TestVehicleFeatures:
    @pytest.fixture
    def engineer(self):
        return FeatureEngineer(region='delaware')

    def test_dimension_compatibility(self, engineer):
        trav, pen = engineer.get_vehicle_dimension_compatibility(4.0, 10000, 4.5, 15000)
        assert trav is True
        assert pen == 1.25  # margin = 0.5, penalty = 1.0 + 0.5 * 0.5

    def test_type_performance(self, engineer):
        cost = engineer.get_vehicle_type_performance('heavy_truck', 10.0, 15.0, 10.0, 'residential', 0.5, 5.0)
        # grade_mult = 1.0 + 0.45 * 10 = 5.5
        # traverse_pen = inf since > 1.5 * 10
        assert cost == float('inf')
"""

with open(test_file, "r") as f:
    lines = f.readlines()

# append right before the if __name__ == '__main__': block
idx = -1
for i, line in enumerate(lines):
    if "if __name__ == '__main__':" in line:
        idx = i
        break

if idx != -1:
    lines.insert(idx, phases_test_code + "\n")
    with open(test_file, "w") as f:
        f.writelines(lines)
    print("Successfully patched test_feature_engineering.py")
else:
    print("Could not find main block in tests")
