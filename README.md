Job Application Success Predictor
📌 Project Overview

The Job Application Success Predictor is a machine learning–based application that predicts the likelihood of a candidate being shortlisted for a job.
The prediction is based on candidate profile attributes such as experience, education level, skills match, certifications, and application behavior.

This project demonstrates the end-to-end ML workflow including data preprocessing, model training, prediction, and deployment using Streamlit.

🎯 Features

Predicts job shortlisting probability in real time

Uses machine learning classification techniques

Interactive web interface built with Streamlit

No external dataset required (uses built-in sample data)

Beginner-friendly and easy to understand

🛠️ Technologies Used

Python

Pandas

NumPy

Scikit-learn

⚙️ How the Model Works

A sample dataset is created with candidate attributes

Categorical values are encoded using Label Encoding

A Random Forest Classifier is trained on the dataset

User inputs are collected via Streamlit UI

The model predicts whether the candidate will be shortlisted
Streamlit

Random Forest Classifier

🌐 Deployment

This project is deployed using Streamlit Cloud.
Simply connect the GitHub repository to Streamlit Cloud and select main.py as the entry file.

📊 Input Parameters

Years of Experience

Education Level

Relevant Skills Match

Number of Certifications

Applied to Similar Roles Before

Cover Letter Included

📈 Output

Prediction: Shortlisted / Not Shortlisted

Confidence score (probability)
