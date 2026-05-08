import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import joblib
import os

from preprocess import load_data, preprocess_data

# ---------------- LOAD DATA ----------------
df = load_data("data/india_housing_prices.csv")
df = preprocess_data(df)

# ---------------- FEATURE SELECTION ----------------
features = [
    'City', 'Property_Type', 'BHK', 'Size_in_SqFt',
    'Nearby_Schools', 'Nearby_Hospitals',
    'Public_Transport_Accessibility', 'Parking_Space',
    'Age_of_Property'
]

# ---------------- HANDLE CATEGORICAL DATA ----------------
le_dict = {}

# Automatically detect categorical columns
categorical_cols = df[features].select_dtypes(include=['object']).columns

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    le_dict[col] = le

# ---------------- DEFINE X & y ----------------
X = df[features]
y_class = df['Good_Investment']   # classification target
y_reg = df['Price_in_Lakhs']      # regression target

# ---------------- TRAIN TEST SPLIT ----------------
X_train, X_test, y_train_class, y_test_class = train_test_split(
    X, y_class, test_size=0.2, random_state=42
)

# ---------------- CLASSIFICATION MODEL ----------------
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train_class)

# ---------------- REGRESSION MODEL ----------------
reg = RandomForestRegressor(n_estimators=100, random_state=42)
reg.fit(X_train, y_reg.loc[X_train.index])

# ---------------- CREATE MODELS FOLDER ----------------
os.makedirs("models", exist_ok=True)

# ---------------- SAVE MODELS ----------------
joblib.dump(clf, "models/classifier.pkl")
joblib.dump(reg, "models/regressor.pkl")
joblib.dump(le_dict, "models/encoders.pkl")

print("✅ Models trained and saved successfully!")