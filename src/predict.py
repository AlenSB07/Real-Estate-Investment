import joblib
import numpy as np

clf = joblib.load("models/classifier.pkl")
reg = joblib.load("models/regressor.pkl")
encoders = joblib.load("models/encoders.pkl")

def preprocess_input(data):
    processed = {}

    for col in data:
        if col in encoders:
            le = encoders[col]
            
            # Handle unseen values
            if data[col] in le.classes_:
                processed[col] = le.transform([data[col]])[0]
            else:
                # Assign default value (first class)
                processed[col] = 0
        else:
            processed[col] = data[col]

    return np.array([list(processed.values())])

def predict(data):
    processed = preprocess_input(data)

    classification = clf.predict(processed)[0]
    price = reg.predict(processed)[0]

    # Simple future growth logic (can improve later)
    future_price = price * 1.4

    return classification, future_price