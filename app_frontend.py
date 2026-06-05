import streamlit as st
import requests
import folium
from streamlit_folium import st_folium
import pandas as pd
from datetime import datetime

# ---------------------------------------------------------------------------
#  CONFIG & CONSTANTS
# ---------------------------------------------------------------------------
API_URL = "http://127.0.0.1:8000/api/v1/optimize-route"

st.set_page_config(
    page_title="Vehicle Routing Optimizer",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for metric cards and timeline
st.markdown("""
<style>
div[data-testid="metric-container"] {
    background-color: #1e1e1e;
    border: 1px solid #333;
    padding: 5% 5% 5% 10%;
    border-radius: 10px;
    box-shadow: 2px 2px 10px rgba(0,0,0,0.5);
    color: white;
}
.timeline {
    display: flex;
    align-items: center;
    flex-wrap: wrap;
    gap: 15px;
    padding: 20px;
    background-color: #2b2b2b;
    border-radius: 10px;
    margin-top: 20px;
    margin-bottom: 20px;
    border: 1px solid #444;
}
.timeline-item {
    display: flex;
    align-items: center;
    background-color: #3d3d3d;
    padding: 10px 15px;
    border-radius: 8px;
    font-weight: bold;
    color: #e0e0e0;
}
.timeline-arrow {
    color: #4CAF50;
    font-size: 24px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
#  HELPER FUNCTIONS
# ---------------------------------------------------------------------------
def format_duration(seconds: int) -> str:
    """Convert seconds to a readable Hours/Minutes format."""
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    if hours > 0:
        return f"{hours}h {minutes}m"
    return f"{minutes}m"

def format_distance(meters: int) -> str:
    """Convert meters to readable km and miles."""
    km = meters / 1000
    miles = km * 0.621371
    return f"{km:.1f} km ({miles:.1f} mi)"

# ---------------------------------------------------------------------------
#  SIDEBAR: USER INPUT INTERFACE
# ---------------------------------------------------------------------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2830/2830305.png", width=100)
    st.title("Route Optimizer")
    st.markdown("Configure your delivery route and temporal constraints below.")
    
    st.subheader("📍 Locations")
    source = st.text_input("Source / Origin Location", value="Times Square, NY", help="Starting depot")
    destination = st.text_input("Destination Location", value="JFK Airport, NY", help="Final drop-off")
    
    waypoints_text = st.text_area(
        "Intermediate Stops / Waypoints", 
        value="Brooklyn Bridge, NY;\nCentral Park, NY",
        help="Separate multiple stops with a semicolon (;)",
        height=100
    )
    
    st.subheader("🕒 Time & Date Context")
    
    current_hour = datetime.now().hour
    current_month = datetime.now().month
    current_weekday = datetime.now().weekday()
    
    departure_hour = st.slider("Departure Hour", min_value=0, max_value=23, value=current_hour, format="%d:00")
    
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    day_name = st.selectbox("Day of the Week", options=days, index=current_weekday)
    day_of_week = days.index(day_name)
    
    months = ["January", "February", "March", "April", "May", "June", 
              "July", "August", "September", "October", "November", "December"]
    month_name = st.selectbox("Month", options=months, index=current_month - 1)
    month = months.index(month_name) + 1
    
    st.subheader("🌩️ Environmental Context")
    is_raining = st.toggle("Live Weather: Is it Raining?", value=False)
    is_festival_zone = st.toggle("Urban Shock: Active Festival/Event?", value=False)
    
    submit_btn = st.button("🚀 Calculate Optimized Route", use_container_width=True, type="primary")

# ---------------------------------------------------------------------------
#  MAIN DASHBOARD
# ---------------------------------------------------------------------------
st.title("🚚 Dynamic Traffic-Optimized Routing Dashboard")
st.markdown("This dashboard interfaces with a live FastAPI backend powered by **Google OR-Tools** and an **XGBoost ML Traffic Model**.")

if submit_btn:
    # 1. Parse Waypoints
    raw_waypoints = [w.strip() for w in waypoints_text.replace("\n", "").split(";") if w.strip()]
    
    # 2. Build Payload
    payload = {
        "source": source,
        "destination": destination,
        "waypoints": raw_waypoints,
        "departure_hour": departure_hour,
        "day_of_week": day_of_week,
        "month": month,
        "is_raining": 1 if is_raining else 0,
        "is_festival_zone": 1 if is_festival_zone else 0
    }
    
    # 3. Call API
    with st.spinner("Fetching live traffic matrices and solving TSP optimization..."):
        try:
            response = requests.post(API_URL, json=payload, timeout=45)
            
            if response.status_code == 200:
                data = response.json()
                st.success("✅ Optimization Complete!")
                
                # --- METRICS PANEL ---
                st.subheader("📊 Core Performance Indicators")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        label="Total Optimized Duration", 
                        value=format_duration(data["total_optimized_duration_seconds"]),
                        delta="Fastest Route Found",
                        delta_color="normal"
                    )
                with col2:
                    st.metric(
                        label="Total Cumulative Distance", 
                        value=format_distance(data["total_distance_meters"])
                    )
                with col3:
                    scale = data["context_applied"].get("ml_scaling_factor")
                    st.metric(
                        label="ML Traffic Scaling Factor", 
                        value=f"{scale:.3f}x" if scale else "Skipped / N/A",
                        help="Applied by XGBoost overlay based on weather & time."
                    )
                
                # --- TIMELINE PANEL ---
                st.subheader("🛣️ Optimal Stop-by-Stop Execution Order")
                sequence = data["optimal_sequence_order"]
                
                timeline_html = "<div class='timeline'>"
                for idx, step in enumerate(sequence):
                    # Assign icons
                    if step == "Source":
                        icon = "🏢"
                        label = f"{icon} {step} (Depot)"
                    elif step == "Destination":
                        icon = "🏁"
                        label = f"{icon} {step}"
                    else:
                        icon = "📍"
                        label = f"{icon} {step}"
                        
                    timeline_html += f"<div class='timeline-item'>{label}</div>"
                    
                    if idx < len(sequence) - 1:
                        # Find the duration of this specific leg
                        leg_duration = "N/A"
                        for leg in data["legs"]:
                            if leg["from_location"] == step and leg["to_location"] == sequence[idx+1]:
                                leg_duration = format_duration(leg["duration_seconds"])
                                break
                        
                        timeline_html += f"<div class='timeline-arrow'>➔ <span style='font-size: 14px; color: #888;'>{leg_duration}</span> ➔</div>"
                
                timeline_html += "</div>"
                st.markdown(timeline_html, unsafe_allow_html=True)
                
                # --- MAP VISUALIZATION ---
                st.subheader("🗺️ Physical Route Map")
                
                geocoded = {loc["label"]: loc for loc in data["geocoded_locations"]}
                
                # Calculate map center
                avg_lat = sum(loc["lat"] for loc in geocoded.values()) / len(geocoded)
                avg_lon = sum(loc["lng"] for loc in geocoded.values()) / len(geocoded)
                
                m = folium.Map(location=[avg_lat, avg_lon], zoom_start=11, tiles="CartoDB positron")
                
                # Draw path lines and markers
                path_coords = []
                for idx, step_label in enumerate(sequence):
                    loc = geocoded[step_label]
                    coord = (loc["lat"], loc["lng"])
                    path_coords.append(coord)
                    
                    # Style markers
                    if step_label == "Source":
                        color = "green"
                        icon = "play"
                    elif step_label == "Destination":
                        color = "red"
                        icon = "stop"
                    else:
                        color = "blue"
                        icon = "info-sign"
                    
                    folium.Marker(
                        location=coord,
                        popup=f"<b>{step_label}</b><br>{loc['address']}",
                        tooltip=step_label,
                        icon=folium.Icon(color=color, icon=icon)
                    ).add_to(m)
                
                # Add route line
                folium.PolyLine(
                    locations=path_coords,
                    color="#4CAF50",
                    weight=4,
                    opacity=0.8,
                    dash_array="10"
                ).add_to(m)
                
                # Render map
                st_folium(m, width=1200, height=500, returned_objects=[])

            elif response.status_code in [400, 422]:
                err_data = response.json()
                st.warning("⚠️ **Validation or Geocoding Error**")
                st.json(err_data)
            else:
                st.error(f"⚠️ API Error ({response.status_code}): {response.text}")
                
        except requests.exceptions.ConnectionError:
            st.error("⚠️ **Backend server offline.** Please ensure FastAPI is running on `http://127.0.0.1:8000`.")
        except requests.exceptions.Timeout:
            st.error("⏳ **Request Timed Out.** The routing backend took too long to respond.")
        except Exception as e:
            st.error(f"❌ **Unexpected Error:** {str(e)}")
else:
    st.info("👈 Enter your route details in the sidebar and click **Calculate Optimized Route**.")
