import os
import joblib
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

from flask import Flask, render_template, request, redirect, session
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

from flask_login import (
    LoginManager,
    UserMixin,
    login_user,
    login_required,
    logout_user,
    current_user
)

from werkzeug.security import (
    generate_password_hash,
    check_password_hash
)

from flask import redirect, url_for, flash, session

# ==========================================
# FLASK APP SETUP
# ==========================================

app = Flask(__name__)
app.secret_key = "heart_disease_secret_key"

# ==========================================
# DATABASE CONFIGURATION
# ==========================================

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///patients.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
# =========================
# LOGIN MANAGER
# =========================

login_manager = LoginManager()

login_manager.init_app(app)

login_manager.login_view = "login"

@login_manager.user_loader
def load_user(user_id):

    return User.query.get(int(user_id))

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
    
    # =========================
# USER TABLE
# =========================

class User(db.Model):

    id = db.Column(db.Integer, primary_key=True)

    username = db.Column(db.String(100), unique=True)

    password = db.Column(db.String(100))

    role = db.Column(db.String(50))

# ==========================================
# CREATE DATABASE
# ==========================================

with app.app_context():
    db.create_all()
        # CREATE DEFAULT USERS

    if not User.query.filter_by(username="admin").first():

        admin = User(
            username="admin",
            password="admin123",
            role="admin"
        )

        doctor = User(
            username="doctor",
            password="doctor123",
            role="doctor"
        )

        patient = User(
            username="patient",
            password="patient123",
            role="patient"
        )

        db.session.add(admin)
        db.session.add(doctor)
        db.session.add(patient)

        db.session.commit()

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

# =========================
# DASHBOARD PAGE
# =========================

@app.route("/dashboard")
def dashboard():

    # Create static folder if missing
    if not os.path.exists("static"):
        os.makedirs("static")

    # =========================
    # FETCH DATA
    # =========================

    patients = Patient.query.all()

    # Prevent empty database errors
    if len(patients) == 0:

        return render_template(
            "dashboard.html",
            total_patients=0,
            high_risk=0,
            low_risk=0
        )

    ages = [p.age for p in patients]
    bmi_values = [p.bmi for p in patients]
    cholesterol_values = [p.cholesterol for p in patients]
    probabilities = [p.probability for p in patients]

    high_risk = Patient.query.filter_by(risk="High Risk").count()
    low_risk = Patient.query.filter_by(risk="Low Risk").count()

    # =========================
    # RISK CHART
    # =========================

    plt.figure(figsize=(5, 4))

    plt.bar(
        ["High Risk", "Low Risk"],
        [high_risk, low_risk],
        color=["red", "green"]
    )

    plt.title("Risk Distribution")

    plt.savefig("static/risk_chart.png")

    plt.close()

    # =========================
    # AGE CHART
    # =========================

    plt.figure(figsize=(5, 4))

    plt.hist(ages, bins=10)

    plt.title("Age Distribution")

    plt.xlabel("Age")
    plt.ylabel("Patients")

    plt.savefig("static/age_chart.png")

    plt.close()

    # =========================
    # BMI CHART
    # =========================

    plt.figure(figsize=(5, 4))

    plt.hist(bmi_values, bins=10)

    plt.title("BMI Distribution")

    plt.xlabel("BMI")

    plt.savefig("static/bmi_chart.png")

    plt.close()

    # =========================
    # CHOLESTEROL CHART
    # =========================

    plt.figure(figsize=(5, 4))

    plt.hist(cholesterol_values, bins=10)

    plt.title("Cholesterol Distribution")

    plt.xlabel("Cholesterol")

    plt.savefig("static/cholesterol_chart.png")

    plt.close()

    # =========================
    # HEATMAP
    # =========================

    import pandas as pd

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

    # =========================
    # RETURN DASHBOARD
    # =========================

    return render_template(
        "dashboard.html",
        total_patients=len(patients),
        high_risk=high_risk,
        low_risk=low_risk
    )
    # =========================
# LOGIN
# =========================

@app.route("/login", methods=["GET", "POST"])
def login():

    if request.method == "POST":

        username = request.form["username"]
        password = request.form["password"]

        user = User.query.filter_by(
            username=username,
            password=password
        ).first()

        if user:

            session["user"] = user.username
            session["role"] = user.role

            # ADMIN
            if user.role == "admin":
                return redirect("/admin")

            # DOCTOR
            elif user.role == "doctor":
                return redirect("/doctor")

            # PATIENT
            elif user.role == "patient":
                return redirect("/patient")

        return "Invalid Login"

    return render_template("login.html")

# =========================
# LOGOUT
# =========================

@app.route("/logout")
def logout():

    session.clear()

    return redirect("/login")
# =========================
# ADMIN DASHBOARD
# =========================

@app.route("/admin")
def admin_dashboard():

    if "role" not in session:
        return redirect("/login")

    if session["role"] != "admin":
        return "Access Denied"

    total_patients = Patient.query.count()

    high_risk = Patient.query.filter_by(
        risk="High Risk"
    ).count()

    low_risk = Patient.query.filter_by(
        risk="Low Risk"
    ).count()

    return render_template(
        "admin_dashboard.html",
        total_patients=total_patients,
        high_risk=high_risk,
        low_risk=low_risk
    )

# =========================
# DOCTOR DASHBOARD
# =========================

@app.route("/doctor")
def doctor_dashboard():

    if "role" not in session:
        return redirect("/login")

    if session["role"] != "doctor":
        return "Access Denied"

    patients = Patient.query.all()

    return render_template(
        "doctor_dashboard.html",
        patients=patients
    )

# =========================
# PATIENT PORTAL
# =========================

@app.route("/patient")
def patient_dashboard():

    if "role" not in session:
        return redirect("/login")

    if session["role"] != "patient":
        return "Access Denied"

    return render_template("patient_dashboard.html")

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