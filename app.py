import os
import joblib
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    session
)

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
from io import BytesIO

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle
)


# ==========================================
# FLASK APP SETUP
# ==========================================

app = Flask(__name__)

app.secret_key = "heart_disease_secret_key"


# ==========================================
# DATABASE CONFIGURATION
# ==========================================

app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///patients.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)


# ==========================================
# LOGIN MANAGER
# ==========================================

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
# PATIENT DATABASE MODEL
# ==========================================

class Patient(db.Model):

    id = db.Column(
        db.Integer,
        primary_key=True
    )

    age = db.Column(
        db.Float
    )

    gender = db.Column(
        db.Float
    )

    bmi = db.Column(
        db.Float
    )

    systolic = db.Column(
        db.Float
    )

    diastolic = db.Column(
        db.Float
    )

    cholesterol = db.Column(
        db.Float
    )

    risk = db.Column(
        db.String(50)
    )

    probability = db.Column(
        db.Float
    )

    date = db.Column(
        db.DateTime,
        default=datetime.utcnow
    )


# ==========================================
# MEDICAL NOTES DATABASE MODEL
# ==========================================

class MedicalNote(db.Model):

    id = db.Column(
        db.Integer,
        primary_key=True
    )

    patient_id = db.Column(
        db.Integer,
        db.ForeignKey("patient.id"),
        nullable=False
    )

    note = db.Column(
        db.Text,
        nullable=False
    )

    created_at = db.Column(
        db.DateTime,
        default=datetime.utcnow
    )

    patient = db.relationship(
        "Patient",
        backref=db.backref(
            "medical_notes",
            lazy=True
        )
    )

    # ==========================================
# PRESCRIPTION DATABASE MODEL
# ==========================================

class Prescription(db.Model):

    id = db.Column(
        db.Integer,
        primary_key=True
    )

    patient_id = db.Column(
        db.Integer,
        db.ForeignKey("patient.id"),
        nullable=False
    )

    medication = db.Column(
        db.String(200),
        nullable=False
    )

    dosage = db.Column(
        db.String(100),
        nullable=False
    )

    frequency = db.Column(
        db.String(100),
        nullable=False
    )

    duration = db.Column(
        db.String(100),
        nullable=False
    )

    instructions = db.Column(
        db.Text,
        nullable=True
    )

    created_at = db.Column(
        db.DateTime,
        default=datetime.utcnow
    )

    patient = db.relationship(
        "Patient",
        backref=db.backref(
            "prescriptions",
            lazy=True
        )
    )


# ==========================================
# USER DATABASE MODEL
# ==========================================

class User(db.Model):

    id = db.Column(
        db.Integer,
        primary_key=True
    )

    username = db.Column(
        db.String(100),
        unique=True
    )

    password = db.Column(
        db.String(100)
    )

    role = db.Column(
        db.String(50)
    )


# ==========================================
# CREATE DATABASE
# ==========================================

with app.app_context():

    db.create_all()

    # ==========================================
    # CREATE DEFAULT USERS
    # ==========================================

    if not User.query.filter_by(
        username="admin"
    ).first():

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
# MEDICAL NOTES
# ==========================================

@app.route(
    "/doctor/patient/<int:patient_id>/notes",
    methods=["GET", "POST"]
)
def medical_notes(patient_id):

    patient = Patient.query.get_or_404(
        patient_id
    )

    if request.method == "POST":

        note_text = request.form.get(
            "note",
            ""
        ).strip()

        if not note_text:

            return render_template(
                "medical_notes.html",
                patient=patient,
                error="Medical note cannot be empty."
            )

        new_note = MedicalNote(
            patient_id=patient.id,
            note=note_text
        )

        db.session.add(new_note)

        db.session.commit()

        return redirect(
            url_for(
                "medical_notes",
                patient_id=patient.id
            )
        )

    notes = MedicalNote.query.filter_by(
        patient_id=patient.id
    ).order_by(
        MedicalNote.created_at.desc()
    ).all()

    return render_template(
        "medical_notes.html",
        patient=patient,
        notes=notes
    )

@app.route("/doctor/patient/<int:patient_id>/report")
def medical_report(patient_id):

    # Get patient
    patient = Patient.query.get_or_404(patient_id)

    # Get patient's medical notes
    notes = MedicalNote.query.filter_by(
        patient_id=patient.id
    ).order_by(
        MedicalNote.created_at.desc()
    ).all()

    # Get patient's prescriptions
    prescriptions = Prescription.query.filter_by(
        patient_id=patient.id
    ).order_by(
        Prescription.created_at.desc()
    ).all()

    # Display medical report
    return render_template(
        "medical_report.html",
        patient=patient,
        notes=notes,
        prescriptions=prescriptions
    )

@app.route("/doctor/patient/<int:patient_id>/report/pdf")
def medical_report_pdf(patient_id):

    # Get patient
    patient = Patient.query.get_or_404(patient_id)

    # Get medical notes
    notes = MedicalNote.query.filter_by(
        patient_id=patient.id
    ).order_by(
        MedicalNote.created_at.desc()
    ).all()

    # Get prescriptions
    prescriptions = Prescription.query.filter_by(
        patient_id=patient.id
    ).order_by(
        Prescription.created_at.desc()
    ).all()

    # Create PDF in memory
    buffer = BytesIO()

    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        alignment=TA_CENTER,
        fontSize=18,
        spaceAfter=8
    )

    subtitle_style = ParagraphStyle(
        "ReportSubtitle",
        parent=styles["Normal"],
        alignment=TA_CENTER,
        fontSize=10,
        textColor=colors.grey,
        spaceAfter=20
    )

    heading_style = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontSize=13,
        spaceBefore=15,
        spaceAfter=8
    )

    normal_style = ParagraphStyle(
        "NormalText",
        parent=styles["Normal"],
        fontSize=9,
        leading=13
    )

    story = []

    # ==========================================
    # REPORT HEADER
    # ==========================================

    story.append(
        Paragraph(
            "HEART DISEASE PREDICTION SYSTEM",
            title_style
        )
    )

    story.append(
        Paragraph(
            "Patient Medical Report",
            subtitle_style
        )
    )

    # ==========================================
    # PATIENT INFORMATION
    # ==========================================

    story.append(
        Paragraph(
            "Patient Information",
            heading_style
        )
    )

    patient_data = [
        ["Patient ID", str(patient.id),
         "Date", str(patient.date)],

        ["Age", str(patient.age),
         "Gender", str(patient.gender)],

        ["BMI", str(patient.bmi),
         "Systolic BP", f"{patient.systolic} mmHg"],

        ["Diastolic BP", f"{patient.diastolic} mmHg",
         "Cholesterol", str(patient.cholesterol)]
    ]

    patient_table = Table(
        patient_data,
        colWidths=[90, 130, 90, 170]
    )

    patient_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#e5e7eb")),
            ("BACKGROUND", (2, 0), (2, -1), colors.HexColor("#e5e7eb")),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ("FONTNAME", (2, 0), (2, -1), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 7),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 7)
        ])
    )

    story.append(patient_table)

    # ==========================================
    # PREDICTION RESULT
    # ==========================================

    story.append(
        Paragraph(
            "Heart Disease Prediction",
            heading_style
        )
    )

    prediction_data = [
        ["Risk Level", str(patient.risk)],
        ["Prediction Probability", f"{patient.probability}%"]
    ]

    prediction_table = Table(
        prediction_data,
        colWidths=[180, 300]
    )

    prediction_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#e5e7eb")),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 10),
            ("TOPPADDING", (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8)
        ])
    )

    story.append(prediction_table)

    # ==========================================
    # MEDICAL NOTES
    # ==========================================

    story.append(
        Paragraph(
            "Medical Notes",
            heading_style
        )
    )

    if notes:

        for note in notes:

            note_date = note.created_at.strftime(
                "%d %B %Y, %H:%M"
            )

            story.append(
                Paragraph(
                    f"<b>{note_date}</b>",
                    normal_style
                )
            )

            story.append(
                Paragraph(
                    str(note.note),
                    normal_style
                )
            )

            story.append(Spacer(1, 8))

    else:

        story.append(
            Paragraph(
                "No medical notes have been recorded.",
                normal_style
            )
        )

    # ==========================================
    # PRESCRIPTIONS
    # ==========================================

    story.append(
        Paragraph(
            "Prescriptions",
            heading_style
        )
    )

    if prescriptions:

        prescription_data = [
            [
                "Medication",
                "Dosage",
                "Frequency",
                "Duration"
            ]
        ]

        for prescription in prescriptions:

            prescription_data.append([
                str(prescription.medication),
                str(prescription.dosage),
                str(prescription.frequency),
                str(prescription.duration)
            ])

        prescription_table = Table(
            prescription_data,
            colWidths=[145, 105, 110, 100],
            repeatRows=1
        )

        prescription_table.setStyle(
            TableStyle([
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.HexColor("#164e63")
                ),

                (
                    "TEXTCOLOR",
                    (0, 0),
                    (-1, 0),
                    colors.white
                ),

                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    colors.grey
                ),

                (
                    "FONTNAME",
                    (0, 0),
                    (-1, 0),
                    "Helvetica-Bold"
                ),

                (
                    "FONTSIZE",
                    (0, 0),
                    (-1, -1),
                    8
                ),

                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "TOP"
                ),

                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    7
                ),

                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    7
                )
            ])
        )

        story.append(prescription_table)

    else:

        story.append(
            Paragraph(
                "No prescriptions have been recorded.",
                normal_style
            )
        )

    # ==========================================
    # FOOTER
    # ==========================================

    story.append(Spacer(1, 25))

    story.append(
        Paragraph(
            "Generated by Heart Disease Prediction System",
            ParagraphStyle(
                "Footer",
                parent=normal_style,
                alignment=TA_CENTER,
                textColor=colors.grey
            )
        )
    )

    story.append(
        Paragraph(
            "For clinical review",
            ParagraphStyle(
                "Footer2",
                parent=normal_style,
                alignment=TA_CENTER,
                textColor=colors.grey
            )
        )
    )

    # Build PDF
    document.build(story)

    # Return PDF
    buffer.seek(0)

    from flask import send_file

    return send_file(
        buffer,
        as_attachment=True,
        download_name=f"medical_report_patient_{patient.id}.pdf",
        mimetype="application/pdf"
    )

# ==========================================
# PRESCRIPTIONS
# ==========================================

@app.route(
    "/doctor/patient/<int:patient_id>/prescription",
    methods=["GET", "POST"]
)
def prescription(patient_id):

    patient = Patient.query.get_or_404(
        patient_id
    )

    if request.method == "POST":

        medication = request.form.get(
            "medication",
            ""
        ).strip()

        dosage = request.form.get(
            "dosage",
            ""
        ).strip()

        frequency = request.form.get(
            "frequency",
            ""
        ).strip()

        duration = request.form.get(
            "duration",
            ""
        ).strip()

        instructions = request.form.get(
            "instructions",
            ""
        ).strip()


        # ==========================================
        # VALIDATION
        # ==========================================

        if not medication:

            return render_template(
                "prescription.html",
                patient=patient,
                prescriptions=Prescription.query.filter_by(
                    patient_id=patient.id
                ).order_by(
                    Prescription.created_at.desc()
                ).all(),
                error="Medication name is required."
            )


        if not dosage:

            return render_template(
                "prescription.html",
                patient=patient,
                prescriptions=Prescription.query.filter_by(
                    patient_id=patient.id
                ).order_by(
                    Prescription.created_at.desc()
                ).all(),
                error="Dosage is required."
            )


        if not frequency:

            return render_template(
                "prescription.html",
                patient=patient,
                prescriptions=Prescription.query.filter_by(
                    patient_id=patient.id
                ).order_by(
                    Prescription.created_at.desc()
                ).all(),
                error="Frequency is required."
            )


        if not duration:

            return render_template(
                "prescription.html",
                patient=patient,
                prescriptions=Prescription.query.filter_by(
                    patient_id=patient.id
                ).order_by(
                    Prescription.created_at.desc()
                ).all(),
                error="Duration is required."
            )


        # ==========================================
        # CREATE PRESCRIPTION
        # ==========================================

        new_prescription = Prescription(

            patient_id=patient.id,

            medication=medication,

            dosage=dosage,

            frequency=frequency,

            duration=duration,

            instructions=instructions

        )


        db.session.add(
            new_prescription
        )

        db.session.commit()


        # ==========================================
        # REDIRECT
        # ==========================================

        return redirect(
            url_for(
                "prescription",
                patient_id=patient.id
            )
        )


    # ==========================================
    # GET PRESCRIPTION HISTORY
    # ==========================================

    prescriptions = Prescription.query.filter_by(
        patient_id=patient.id
    ).order_by(
        Prescription.created_at.desc()
    ).all()


    return render_template(
        "prescription.html",

        patient=patient,

        prescriptions=prescriptions
    )


# ==========================================
# HOME PAGE
# ==========================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


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

    # Create static folder if missing

    if not os.path.exists("static"):

        os.makedirs("static")


    # ==========================================
    # FETCH DATA
    # ==========================================

    patients = Patient.query.all()


    # Prevent empty database errors

    if len(patients) == 0:

        return render_template(
            "dashboard.html",
            total_patients=0,
            high_risk=0,
            low_risk=0
        )


    ages = [
        p.age
        for p in patients
    ]

    bmi_values = [
        p.bmi
        for p in patients
    ]

    cholesterol_values = [
        p.cholesterol
        for p in patients
    ]

    probabilities = [
        p.probability
        for p in patients
    ]


    high_risk = Patient.query.filter_by(
        risk="High Risk"
    ).count()


    low_risk = Patient.query.filter_by(
        risk="Low Risk"
    ).count()


    # ==========================================
    # RISK CHART
    # ==========================================

    plt.figure(
        figsize=(5, 4)
    )

    plt.bar(
        ["High Risk", "Low Risk"],
        [high_risk, low_risk],
        color=["red", "green"]
    )

    plt.title(
        "Risk Distribution"
    )

    plt.savefig(
        "static/risk_chart.png"
    )

    plt.close()


    # ==========================================
    # AGE CHART
    # ==========================================

    plt.figure(
        figsize=(5, 4)
    )

    plt.hist(
        ages,
        bins=10
    )

    plt.title(
        "Age Distribution"
    )

    plt.xlabel(
        "Age"
    )

    plt.ylabel(
        "Patients"
    )

    plt.savefig(
        "static/age_chart.png"
    )

    plt.close()


    # ==========================================
    # BMI CHART
    # ==========================================

    plt.figure(
        figsize=(5, 4)
    )

    plt.hist(
        bmi_values,
        bins=10
    )

    plt.title(
        "BMI Distribution"
    )

    plt.xlabel(
        "BMI"
    )

    plt.savefig(
        "static/bmi_chart.png"
    )

    plt.close()


    # ==========================================
    # CHOLESTEROL CHART
    # ==========================================

    plt.figure(
        figsize=(5, 4)
    )

    plt.hist(
        cholesterol_values,
        bins=10
    )

    plt.title(
        "Cholesterol Distribution"
    )

    plt.xlabel(
        "Cholesterol"
    )

    plt.savefig(
        "static/cholesterol_chart.png"
    )

    plt.close()


    # ==========================================
    # HEATMAP
    # ==========================================

    data = pd.DataFrame({

        "Age": ages,

        "BMI": bmi_values,

        "Cholesterol": cholesterol_values,

        "Probability": probabilities

    })


    correlation = data.corr()


    plt.figure(
        figsize=(8, 6)
    )

    sns.heatmap(
        correlation,
        annot=True,
        cmap="coolwarm"
    )

    plt.title(
        "Correlation Heatmap"
    )

    plt.savefig(
        "static/heatmap.png"
    )

    plt.close()


    # ==========================================
    # RETURN DASHBOARD
    # ==========================================

    return render_template(
        "dashboard.html",

        total_patients=len(patients),

        high_risk=high_risk,

        low_risk=low_risk
    )


# ==========================================
# LOGIN
# ==========================================

@app.route(
    "/login",
    methods=["GET", "POST"]
)
def login():

    if request.method == "POST":

        username = request.form[
            "username"
        ]

        password = request.form[
            "password"
        ]


        user = User.query.filter_by(
            username=username,
            password=password
        ).first()


        if user:

            session["user"] = user.username

            session["role"] = user.role


            # ==================================
            # ADMIN
            # ==================================

            if user.role == "admin":

                return redirect(
                    "/admin"
                )


            # ==================================
            # DOCTOR
            # ==================================

            elif user.role == "doctor":

                return redirect(
                    "/doctor"
                )


            # ==================================
            # PATIENT
            # ==================================

            elif user.role == "patient":

                return redirect(
                    "/patient"
                )


        return "Invalid Login"


    return render_template(
        "login.html"
    )


# ==========================================
# LOGOUT
# ==========================================

@app.route("/logout")
def logout():

    session.clear()

    return redirect(
        "/login"
    )


# ==========================================
# ADMIN DASHBOARD
# ==========================================

@app.route("/admin")
def admin_dashboard():

    if "role" not in session:

        return redirect(
            "/login"
        )


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


# ==========================================
# DOCTOR DASHBOARD
# ==========================================

@app.route("/doctor")
def doctor_dashboard():

    search = request.args.get(
        "search",
        ""
    ).strip()

    risk = request.args.get(
        "risk",
        ""
    ).strip()


    query = Patient.query


    # ==========================================
    # SEARCH BY PATIENT ID OR GENDER
    # ==========================================

    if search:

        if search.isdigit():

            query = query.filter(
                db.or_(
                    Patient.id == int(search),
                    Patient.gender.ilike(
                        f"%{search}%"
                    )
                )
            )

        else:

            query = query.filter(
                Patient.gender.ilike(
                    f"%{search}%"
                )
            )


    # ==========================================
    # FILTER BY RISK
    # ==========================================

    if risk:

        query = query.filter(
            Patient.risk == risk
        )


    # ==========================================
    # NEWEST RECORDS FIRST
    # ==========================================

    patients = query.order_by(
        Patient.date.desc()
    ).all()


    return render_template(
        "doctor_dashboard.html",

        patients=patients,

        search=search,

        risk=risk
    )


# ==========================================
# DOCTOR SEARCH PATIENTS
# ==========================================

@app.route(
    "/doctor/search-patients"
)
def doctor_search_patients():

    # Get search values from URL

    search = request.args.get(
        "search",
        ""
    ).strip()

    risk = request.args.get(
        "risk",
        ""
    ).strip()


    # Start with all patients

    query = Patient.query


    # ==========================================
    # SEARCH BY PATIENT ID OR GENDER
    # ==========================================

    if search:

        if search.isdigit():

            query = query.filter(
                db.or_(
                    Patient.id == int(search),

                    Patient.gender.ilike(
                        f"%{search}%"
                    )
                )
            )

        else:

            query = query.filter(
                Patient.gender.ilike(
                    f"%{search}%"
                )
            )


    # ==========================================
    # FILTER BY RISK
    # ==========================================

    if risk:

        query = query.filter(
            Patient.risk == risk
        )


    # ==========================================
    # NEWEST PATIENTS FIRST
    # ==========================================

    patients = query.order_by(
        Patient.date.desc()
    ).all()


    return render_template(
        "doctor_search_patients.html",

        patients=patients,

        search=search,

        risk=risk
    )


# ==========================================
# PATIENT PORTAL
# ==========================================

@app.route("/patient")
def patient_dashboard():

    if "role" not in session:

        return redirect(
            "/login"
        )


    if session["role"] != "patient":

        return "Access Denied"


    return render_template(
        "patient_dashboard.html"
    )


# ==========================================
# HEALTH RECOMMENDATIONS
# ==========================================

def get_health_recommendations(
    risk,
    probability
):
    """
    Generate general health recommendations
    based on the predicted cardiovascular risk.

    This is educational guidance and is not
    a medical diagnosis.
    """

    recommendations = []


    # ==========================================
    # HIGH RISK
    # ==========================================

    if risk == "High Risk":

        recommendations = [

            "Consult a qualified healthcare professional for further assessment.",

            "Monitor your blood pressure regularly.",

            "Monitor cholesterol and blood sugar levels.",

            "Reduce excessive salt and saturated-fat intake.",

            "Eat more vegetables, fruits, whole grains and other heart-healthy foods.",

            "Avoid smoking and exposure to tobacco smoke.",

            "Limit alcohol consumption.",

            "Maintain regular physical activity appropriate for your health condition.",

            "Maintain a healthy body weight.",

            "Take prescribed medication according to your healthcare professional's instructions."

        ]


    # ==========================================
    # LOW RISK
    # ==========================================

    else:

        recommendations = [

            "Maintain a balanced and heart-healthy diet.",

            "Exercise regularly according to your fitness level.",

            "Maintain a healthy body weight.",

            "Monitor your blood pressure periodically.",

            "Monitor cholesterol and blood sugar levels.",

            "Avoid smoking and tobacco products.",

            "Limit excessive alcohol consumption.",

            "Get adequate sleep and manage stress.",

            "Continue regular medical check-ups.",

            "Maintain your current healthy lifestyle."

        ]


    # ==========================================
    # ADDITIONAL RECOMMENDATION
    # ==========================================

    if probability >= 75:

        recommendations.insert(
            0,
            "Your predicted risk is relatively high. Consider seeking professional medical evaluation."
        )


    elif probability >= 50:

        recommendations.insert(
            0,
            "Your predicted risk is elevated. Consider discussing your cardiovascular risk with a healthcare professional."
        )


    return recommendations


# ==========================================
# PREDICTION ROUTE
# ==========================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    try:

        # ==========================================
        # GET INPUTS
        # ==========================================

        features = [

            float(
                request.form["Age"]
            ),

            float(
                request.form["Gender"]
            ),

            float(
                request.form["Weight"]
            ),

            float(
                request.form["Height"]
            ),

            float(
                request.form["BMI"]
            ),

            float(
                request.form["Smoking"]
            ),

            float(
                request.form["Alcohol_Intake"]
            ),

            float(
                request.form["Physical_Activity"]
            ),

            float(
                request.form["Diet"]
            ),

            float(
                request.form["Stress_Level"]
            ),

            float(
                request.form["Hypertension"]
            ),

            float(
                request.form["Diabetes"]
            ),

            float(
                request.form["Hyperlipidemia"]
            ),

            float(
                request.form["Family_History"]
            ),

            float(
                request.form["Systolic_BP"]
            ),

            float(
                request.form["Diastolic_BP"]
            ),

            float(
                request.form["Heart_Rate"]
            ),

            float(
                request.form["Blood_Sugar_Fasting"]
            ),

            float(
                request.form["Cholesterol_Total"]
            )

        ]


        # ==========================================
        # CONVERT TO NUMPY ARRAY
        # ==========================================

        input_data = np.array(
            [features]
        )


        # ==========================================
        # SCALE INPUT
        # ==========================================

        input_scaled = scaler.transform(
            input_data
        )


        # ==========================================
        # MAKE PREDICTION
        # ==========================================

        prediction = model.predict(
            input_scaled
        )[0]


        probability = model.predict_proba(
            input_scaled
        )[0][1] * 100


        probability = round(
            probability,
            2
        )


        # ==========================================
        # DETERMINE RISK
        # ==========================================

        if prediction == 1:

            result = "High Risk"

        else:

            result = "Low Risk"


        # ==========================================
        # GENERATE HEALTH RECOMMENDATIONS
        # ==========================================

        recommendations = get_health_recommendations(
            result,
            probability
        )


        # ==========================================
        # SAVE TO DATABASE
        # ==========================================

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


        db.session.add(
            patient
        )


        db.session.commit()


        # ==========================================
        # RETURN RESULT
        # ==========================================

        return render_template(

            "index.html",

            prediction_text=(
                f"{result} ({probability}%)"
            ),

            risk=result,

            probability=probability,

            recommendations=recommendations

        )


    except Exception as e:

        return render_template(

            "index.html",

            prediction_text=(
                f"Error: {str(e)}"
            )

        )


# ==========================================
# RUN APPLICATION
# ==========================================

if __name__ == "__main__":

    app.run(
        debug=True,
        use_reloader=False
    )