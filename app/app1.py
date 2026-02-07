import streamlit as st
import joblib
import pandas as pd

st.title("Heart Disease Prediction App")

model=joblib.load("heart_disease.pkl")
scaler=joblib.load("scaler.pkl")
encoder=joblib.load("encoder.pkl")

st.write("Enter the input values: ")

#User inputs (same features as triangle)
Age=st.number_input("Age")
Sex=st.number_input("Sex")
Chest_pain_type=st.number_input("Chest pain type")
BP=st.number_input("BP")
Cholesterol=st.number_input("Cholestrol")
FBS_over_120=st.number_input("FBS over 120")
EKG_results=st.number_input("EKG results")
Max_HR=st.number_input("Max HR")
Exercise_angina=st.number_input("Exercise angina")
ST_depression=st.number_input("ST depression")
Slope_of_ST=st.number_input("Slope of ST")
Number_of_vessel_fluro=st.number_input("Number of vessel")
Thallium=st.number_input("Thallium")


if st.button("Predict Heart Disease"):
    data= {
    "Age":Age,
    "Sex":Sex,
    "Chest pain type":Chest_pain_type,
    "BP":BP,
    "Cholesterol":Cholesterol,
    "FBS over 120":FBS_over_120,
    "EKG results":EKG_results,
    "Max HR":Max_HR,
    "Exercise angina":Exercise_angina,
    "ST depression":ST_depression,
    "Slope of ST":Slope_of_ST,
    "Number of vessels fluro":Number_of_vessel_fluro,
    "Thallium":Thallium
    }
    
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

#Show results
st.success(f"Predicted : {prediction}")
