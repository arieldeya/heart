import os
import joblib
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

from flask import Flask, render_template, request
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

# ==========================================
# FLASK APP SETUP
# ==========================================

app = Flask(__name__)

# ==========================================
# DATABASE CONFIGURATION
# ==========================================

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///patients.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# ==========================================
# LOAD MODEL + SCALER
# ==========================================

model = joblib.load("ensemble_model.pkl")
scaler = joblib.load("scaler.pkl")

# ==========================================
# DATABASE MODEL
# ==========================================

class Patient(db.Model):

    id = db.Column(db.Integer, primary_key=True)

    age = db.Column(db.Float)
    gender = db.Column(db.Float)

    bmi = db.Column(db.Float)

    systolic = db.Column(db.Float)
    diastolic = db.Column(db.Float)

    cholesterol = db.Column(db.Float)

    risk = db.Column(db.String(50))
    probability = db.Column(db.Float)

    date = db.Column(db.DateTime, default=datetime.utcnow)

# ==========================================
# CREATE DATABASE
# ==========================================

with app.app_context():
    db.create_all()

# ==========================================
# HOME PAGE
# ==========================================

@app.route("/")
def home():
    return render_template("index.html")

# ==========================================
# HISTORY PAGE
# ==========================================

@app.route("/history")
def history():

    patients = Patient.query.order_by(
        Patient.date.desc()
    ).all()

    return render_template(
        "history.html",
        patients=patients
    )

# ==========================================
# DASHBOARD PAGE
# ==========================================

@app.route("/dashboard")
def dashboard():

    # Create static folder
    if not os.path.exists("static"):
        os.makedirs("static")

    patients = Patient.query.all()

    ages = [p.age for p in patients]
    bmi_values = [p.bmi for p in patients]
    cholesterol_values = [p.cholesterol for p in patients]
    probabilities = [p.probability for p in patients]

    high_risk = Patient.query.filter_by(
        risk="High Risk"
    ).count()

    low_risk = Patient.query.filter_by(
        risk="Low Risk"
    ).count()

    # ======================================
    # RISK CHART
    # ======================================

    plt.figure(figsize=(6, 4))

    plt.bar(
        ["High Risk", "Low Risk"],
        [high_risk, low_risk],
        color=["red", "green"]
    )

    plt.title("Heart Disease Risk Distribution")
    plt.ylabel("Patients")

    plt.savefig("static/risk_chart.png")
    plt.close()

    # ======================================
    # AGE DISTRIBUTION
    # ======================================

    if len(ages) > 0:

        plt.figure(figsize=(6, 4))

        plt.hist(
            ages,
            bins=10,
            color="skyblue"
        )

        plt.title("Age Distribution")
        plt.xlabel("Age")
        plt.ylabel("Patients")

        plt.savefig("static/age_chart.png")
        plt.close()

    # ======================================
    # BMI DISTRIBUTION
    # ======================================

    if len(bmi_values) > 0:

        plt.figure(figsize=(6, 4))

        plt.hist(
            bmi_values,
            bins=10,
            color="orange"
        )

        plt.title("BMI Distribution")
        plt.xlabel("BMI")

        plt.savefig("static/bmi_chart.png")
        plt.close()

    # ======================================
    # CHOLESTEROL DISTRIBUTION
    # ======================================

    if len(cholesterol_values) > 0:

        plt.figure(figsize=(6, 4))

        plt.hist(
            cholesterol_values,
            bins=10,
            color="purple"
        )

        plt.title("Cholesterol Distribution")
        plt.xlabel("Cholesterol")

        plt.savefig("static/cholesterol_chart.png")
        plt.close()

    # ======================================
    # HEATMAP
    # ======================================

    if len(patients) > 0:

        data = pd.DataFrame({
            "Age": ages,
            "BMI": bmi_values,
            "Cholesterol": cholesterol_values,
            "Probability": probabilities
        })

        correlation = data.corr()

        plt.figure(figsize=(8, 6))

        sns.heatmap(
            correlation,
            annot=True,
            cmap="coolwarm"
        )

        plt.title("Correlation Heatmap")

        plt.savefig("static/heatmap.png")
        plt.close()

    return render_template(
        "dashboard.html",
        total_patients=len(patients),
        high_risk=high_risk,
        low_risk=low_risk
    )

# ==========================================
# PREDICTION ROUTE
# ==========================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        # ==================================
        # GET INPUTS
        # ==================================

        features = [

            float(request.form["Age"]),
            float(request.form["Gender"]),
            float(request.form["Weight"]),
            float(request.form["Height"]),
            float(request.form["BMI"]),
            float(request.form["Smoking"]),
            float(request.form["Alcohol_Intake"]),
            float(request.form["Physical_Activity"]),
            float(request.form["Diet"]),
            float(request.form["Stress_Level"]),
            float(request.form["Hypertension"]),
            float(request.form["Diabetes"]),
            float(request.form["Hyperlipidemia"]),
            float(request.form["Family_History"]),
            float(request.form["Systolic_BP"]),
            float(request.form["Diastolic_BP"]),
            float(request.form["Heart_Rate"]),
            float(request.form["Blood_Sugar_Fasting"]),
            float(request.form["Cholesterol_Total"])

        ]

        # ==================================
        # CONVERT TO NUMPY ARRAY
        # ==================================

        input_data = np.array([features])

        # ==================================
        # SCALE INPUT
        # ==================================

        input_scaled = scaler.transform(input_data)

        # ==================================
        # MAKE PREDICTION
        # ==================================

        prediction = model.predict(
            input_scaled
        )[0]

        probability = model.predict_proba(
            input_scaled
        )[0][1] * 100

        probability = round(probability, 2)

        # ==================================
        # DETERMINE RISK
        # ==================================

        if prediction == 1:
            result = "High Risk"
        else:
            result = "Low Risk"

        # ==================================
        # SAVE TO DATABASE
        # ==================================

        patient = Patient(

            age=features[0],
            gender=features[1],

            bmi=features[4],

            systolic=features[14],
            diastolic=features[15],

            cholesterol=features[18],

            risk=result,
            probability=probability
        )

        db.session.add(patient)
        db.session.commit()

        # ==================================
        # RETURN RESULT
        # ==================================

        return render_template(
            "index.html",
            prediction_text=f"{result} ({probability}%)"
        )

    except Exception as e:

        return render_template(
            "index.html",
            prediction_text=f"Error: {str(e)}"
        )

# ==========================================
# RUN APPLICATION
# ==========================================

if __name__ == "__main__":
    app.run(
        debug=True,
        use_reloader=False
    )