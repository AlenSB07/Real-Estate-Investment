import pandas as pd
import numpy as np

def load_data(path):
    df = pd.read_csv(path)
    return df

def preprocess_data(df):
    # Remove duplicates
    df = df.drop_duplicates()

    # Fill missing values
    df['Parking_Space'] = df['Parking_Space'].fillna(0)
    df['Nearby_Schools'] = df['Nearby_Schools'].fillna(df['Nearby_Schools'].median())
    df['Nearby_Hospitals'] = df['Nearby_Hospitals'].fillna(df['Nearby_Hospitals'].median())
    df['Furnished_Status'] = df['Furnished_Status'].fillna("Unfurnished")

    # Feature Engineering
    df['Price_per_SqFt'] = df['Price_in_Lakhs'] * 100000 / df['Size_in_SqFt']
    df['Age_of_Property'] = 2025 - df['Year_Built']

    # ---------------- GOOD INVESTMENT LABEL ----------------
    df['Good_Investment'] = np.where(
        (df['Price_per_SqFt'] < 5000) &
        (df['Public_Transport_Accessibility'] == "High") &
        (df['Amenities'] == "High"),
        1, 0
    )

    return df