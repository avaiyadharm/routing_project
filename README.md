# Dynamic Traffic-Optimized Routing Engine

A high-performance, machine learning-driven routing and fleet optimization engine designed to minimize multi-stop travel times by combining real-time contextual variables with physical road network constraints.

This system moves away from traditional, static routing (which relies strictly on distance or speed limits) by incorporating a hybrid architecture: **OSRM** for spatial network geometry, an **XGBoost Regressor** trained on historical trip profiles to predict stochastic traffic variations, and **Google OR-Tools** to solve NP-hard Vehicle Routing Problems (VRP).

---

## 🛠️ System Architecture

The core framework follows a decoupled, microservices-inspired processing pipeline:

1. **Spatial Layer (OSRM):** Consumes OpenStreetMap (OSM) data and utilizes Multi-Level Dijkstra (MLD) graph partitioning to handle physical street infrastructure (one-ways, legal turns, speeds).
2. **Predictive Layer (XGBoost):** Processes contextual environmental constraints to generate a dynamic, traffic-adjusted Travel Time Matrix.
3. **Prescriptive Layer (Google OR-Tools):** Executes constraint programming and guided local search heuristics to evaluate billions of route permutations, outputting the absolute fastest sequence.

---

## 📐 Mathematical Formulation

### Core Path Cost
Let the street network be represented as a directed graph $G = (V, E)$, where $V$ represents intersections and $E$ represents directed road segments. For any edge $e \in E$, its routing cost is calculated as a time-varying function:

$$\text{Cost}(e) = f\big(\text{OSRM\_Base\_Time}(e), \mathbf{X}_{\text{temporal}}, \mathbf{X}_{\text{environmental}}, \mathbf{X}_{\text{anomalous}}\big)$$

Where:
* $\mathbf{X}_{\text{temporal}} = \{\text{hour\_of\_day}, \text{day\_of\_week}, \text{month}\}$
* $\mathbf{X}_{\text{environmental}} = \{\text{is\_raining}\}$
* $\mathbf{X}_{\text{anomalous}} = \{\text{is\_festival\_zone}\}$

### Sequence Optimization Objective
Given a Source depot $S$ and a series of customer delivery vectors, Google OR-Tools computes the decision path sequence matrix that satisfies:

$$\min \sum_{e \in P} \text{Cost}(e)$$

---

## 📂 Repository File Structure

```text
routing_project/
├── .venv/                        # Isolated Python Virtual Environment
├── .gitignore                    # Prevents local heavy tracking data from pushing to GitHub
├── delaware-latest.osm.pbf       # Raw binary spatial OpenStreetMap snapshot
├── delaware-latest.osrm* # Compiled multi-level OSRM map graph binaries
├── train.csv                     # Historical benchmark training dataset (~400k records)
├── train_traffic_model.py        # Feature engineering and XGBoost training pipeline script
├── traffic_xgb_model.pkl         # Serialized ML model containing trained tree weights
├── optimize_route.py             # Main runtime combining OSRM, ML predictions, and OR-Tools
├── requirements.txt              # Project package footprints
└── Dockerfile                    # Multi-stage production deployment container configurations