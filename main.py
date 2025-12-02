import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

data = {
    "Years_of_Experience": [0, 2, 5, 7, 10, 3, 8, 12, 1, 4],
    "Education_Level": ["High School", "Bachelor's", "Master's", "PhD", "Bachelor's",
                        "Master's", "PhD", "Bachelor's", "High School", "Master's"],
    "Relevant_Skills_Match": [1, 3, 4, 5, 4, 4, 5, 3, 2, 4],
    "Certifications": [0, 1, 2, 3, 1, 2, 3, 2, 0, 1],
    "Applied_to_Similar_Roles": ["No", "Yes", "Yes", "No", "Yes", "No", "Yes", "Yes", "No", "Yes"],
    "Cover_Letter": ["No", "Yes", "Yes", "No", "Yes", "Yes", "Yes", "No", "No", "Yes"],
    "Shortlisted": ["No", "Yes", "Yes", "Yes", "Yes", "Yes", "Yes", "Yes", "No", "Yes"]
}

df = pd.DataFrame(data)

le_education = LabelEncoder()
le_yesno = LabelEncoder()

df["Education_Level"] = le_education.fit_transform(df["Education_Level"])
df["Applied_to_Similar_Roles"] = le_yesno.fit_transform(df["Applied_to_Similar_Roles"])
df["Cover_Letter"] = le_yesno.fit_transform(df["Cover_Letter"])
df["Shortlisted"] = le_yesno.fit_transform(df["Shortlisted"])

X = df.drop("Shortlisted", axis=1)
y = df["Shortlisted"]

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

st.title("🏢 Job Application Success Predictor")
st.write("Predict the likelihood of a candidate being shortlisted for a job.")

years = st.slider("Years of Experience", 0, 20, 1)
education = st.selectbox("Education Level", ["High School", "Bachelor's", "Master's", "PhD"])
skills = st.slider("Relevant Skills Match", 1, 5, 1)
certs = st.slider("Number of Certifications", 0, 5, 0)
applied_before = st.selectbox("Applied to Similar Roles Before?", ["Yes", "No"])
cover_letter = st.selectbox("Cover Letter Included?", ["Yes", "No"])

if st.button("Predict Shortlist"):
    edu_encoded = le_education.transform([education])[0]
    applied_encoded = le_yesno.transform([applied_before])[0]
    cover_encoded = le_yesno.transform([cover_letter])[0]

    features = [[years, edu_encoded, skills, certs, applied_encoded, cover_encoded]]
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][prediction] * 100
    result = le_yesno.inverse_transform([prediction])[0]

    st.success(f"Shortlist Prediction: {result} ({probability:.2f}% likelihood)")
