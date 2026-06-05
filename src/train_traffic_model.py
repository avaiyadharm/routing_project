import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
import pickle

print("🚀 Step 1: Loading train.csv dataset...")
# Load the dataset (using nrows=500000 to train quickly while maintaining accuracy)
df = pd.read_csv("train.csv", nrows=500000)

print("🧹 Step 2: Cleaning and engineering features...")
# Convert raw pickup string timestamp to a real datetime object
df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])

# Extract temporal features that affect traffic flow
df['hour_of_day'] = df['pickup_datetime'].dt.hour
df['day_of_week'] = df['pickup_datetime'].dt.weekday  # 0=Monday, 6=Sunday
df['month'] = df['pickup_datetime'].dt.month

# 🚨 INJECT CONTEXTUAL FEATURES (Weather and Festivals)
# Since the raw dataset doesn't have weather, we simulate rainy days (20% chance)
np.random.seed(42)
df['is_raining'] = np.random.choice([0, 1], size=len(df), p=[0.8, 0.2])

# We engineer a "Festival Zone" feature. Let's simulate a massive recurring festival 
# happening near the center of Manhattan (lat 40.75 to 40.76) during evening hours.
df['is_festival_zone'] = (
    (df['pickup_latitude'].between(40.75, 40.76)) & 
    (df['hour_of_day'].between(18, 22))
).astype(int)

# Filter out extreme outliers (trips longer than 3 hours or shorter than 30 seconds)
df = df[df['trip_duration'].between(30, 10800)]

# Define Input Features (X) and Target Variable (y)
feature_cols = ['hour_of_day', 'day_of_week', 'month', 'is_raining', 'is_festival_zone']
X = df[feature_cols]
y = df['trip_duration']

# Step 3: Split into Training (80%) and Testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"🏋️ Step 4: Training XGBoost Regressor on {len(X_train)} trips...")
# Initialize the XGBoost regressor model
traffic_model = xgb.XGBRegressor(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    n_jobs=-1 # Uses all processor cores on your Mac for max speed
)

# Train the model
traffic_model.fit(X_train, y_train)

print("📊 Step 5: Evaluating model performance...")
predictions = traffic_model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
print(f"✅ Training Complete! Mean Absolute Error (MAE): {mae:.2f} seconds")

print("💾 Step 6: Exporting model to binary file...")
# Save the trained model as a reusable file
with open("traffic_xgb_model.pkl", "wb") as f:
    pickle.dump(traffic_model, f)

print("🎉 Success! 'traffic_xgb_model.pkl' is ready for your routing engine.")