from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()

"""
Heart Disease Prediction API
- Listens for requests
- Predicts heart disease based on features
"""

# Load trained components
model = joblib.load("heart_disease.pkl")
scaler = joblib.load("scaler.pkl")
encoder = joblib.load("encoder.pkl")

# Root endpoint
@app.get("/")
def home():
    return {"message": "Heart Disease Prediction API"}

# Predict endpoint
@app.post("/predict")
def predict(data: dict):
    import traceback
    try:
        # Convert input JSON to DataFrame
        df = pd.DataFrame([data])

        # Categorical columns (keep these for encoding)
        cat_cols = ['Chest pain type', 'EKG results', 'Slope of ST', 'Number of vessels fluro', 'Thallium']

        # Encode categorical features
        encoded = encoder.transform(df[cat_cols])
        encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(cat_cols))

        # Drop original categorical columns and append encoded
        df = df.drop(cat_cols, axis=1)
        df = pd.concat([df.reset_index(drop=True), encoded_df], axis=1)

        # Scale all features
        df_scaled = scaler.transform(df)

        # Predict (returns 'Absence' / 'Presence')
        prediction = model.predict(df_scaled)[0]

        return {"prediction": prediction}

    except Exception as e:
        print("Error in /predict:", e)
        traceback.print_exc()
        return {"error": str(e)}
