import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download the model from the Model Hub
model_path = hf_hub_download(repo_id="MissSamyuktha/tourism-prediction-model", filename="best_tourism_pred_model_v1.joblib")

# Load the model
model = joblib.load(model_path)

# Streamlit UI for Customer Churn Prediction
st.title("Tourism Success Prediction App")
st.write("The Tourism Success Prediction App is an internal tool for torism company staff that predicts whether customers are going to buy their products.")
st.write("Kindly enter the customer details to check whether they are likely to purchase.")

# Collect user input
Age = st.number_input("Age (in years)", min_value=18, max_value=100, value=25)
CityTier = st.selectbox("City Tier", [1, 2, 3])
TypeofContact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
Occupation = st.selectbox("Occupation", ["Salaried", "Free Lancer", "Small Business", "Large Business"])
Gender = st.selectbox("Gender", ["Male", "Female"])
NumberOfPersonVisiting = st.number_input("Number of Persons Visiting", min_value=0, max_value=100, value=2)
PreferredPropertyStar = st.selectbox("Preferred Property Star Rating", [3, 4, 5])
MaritalStatus = st.selectbox("Marital Status", ["Single", "Divorced", "Married", "Unmarried"])
NumberOfTrips = st.number_input("Average Number of Trips By The Customer", value=5)
Passport = st.selectbox("Has Passport?", ["Yes", "No"])
OwnCar = st.selectbox("Has Car?", ["Yes", "No"])
NumberOfChildrenVisiting = st.number_input("Number Of Children(Below 5)Accompanying The Customer", min_value=0.0, value=2)
Designation = st.selectbox("Customer's Designation in their Current Company", ["Manager", "Executive", "Senior Manager", "AVP", "VP"])
MonthlyIncome = st.number_input("Gross Monthly Income Of The Customer", min_value=100, value=100)

# Collect Customer Interaction Data if any
PitchSatisfactionScore = st.selectbox("Pitch Satisfaction Score (On Scale of 1 to 5)", [1, 2, 3, 4, 5])
ProductPitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
NumberOfFollowups = st.number_input("Number of Follow-ups", min_value=0.0, value=2)
DurationOfPitch = st.number_input("Duration of Pitch", min_value=0.0, value=10)

# Convert categorical inputs to match model training
input_data = pd.DataFrame([{
    'Age': Age,
    'DurationOfPitch': DurationOfPitch,
    'MonthlyIncome': MonthlyIncome,
    'CityTier': CityTier,
    'NumberOfPersonVisiting': NumberOfPersonVisiting,
    'NumberOfFollowups': NumberOfFollowups,
    'PreferredPropertyStar': PreferredPropertyStar,
    'NumberOfTrips': NumberOfTrips,
    'Passport': 1 if Passport == "Yes" else 0,
    'PitchSatisfactionScore': PitchSatisfactionScore,
    'OwnCar': 1 if OwnCar == "Yes" else 0,
    'NumberOfChildrenVisiting': NumberOfChildrenVisiting,
    'TypeofContact': TypeofContact,
    'Occupation': Occupation,
    'Gender': Gender,
    'ProductPitched': ProductPitched,
    'MaritalStatus': MaritalStatus,
    'Designation': Designation
}])

# Set the classification threshold
classification_threshold = 0.5

# Predict button
if st.button("Predict"):
    prediction_proba = model.predict_proba(input_data)[0, 1]
    prediction = (prediction_proba >= classification_threshold).astype(int)
    result = "purchase" if prediction == 1 else "not purchase"
    st.write(f"Based on the information provided, the customer is likely to {result}.")
