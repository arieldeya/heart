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
    session,
    send_file
)

from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from sqlalchemy import text

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


# ============================================================
# FLASK APP SETUP
# ============================================================

app = Flask(__name__)

app.secret_key = "heart_disease_secret_key"


# ============================================================
# DATABASE CONFIGURATION
# ============================================================

app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///patients.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)


# ============================================================
# LOGIN MANAGER
# ============================================================

login_manager = LoginManager()

login_manager.init_app(app)

login_manager.login_view = "login"


@login_manager.user_loader
def load_user(user_id):

    return User.query.get(int(user_id))


# ============================================================
# LOAD MACHINE LEARNING MODEL
# ============================================================

model = joblib.load("ensemble_model.pkl")

scaler = joblib.load("scaler.pkl")


# ============================================================
# PATIENT MODEL
# ============================================================

class Patient(db.Model):

    id = db.Column(
        db.Integer,
        primary_key=True
    )

    patient_code = db.Column(
        db.String(50),
        nullable=True,
        index=True
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


# ============================================================
# MEDICAL NOTES MODEL
# ============================================================

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


# ============================================================
# PRESCRIPTION MODEL
# ============================================================

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


# ============================================================
# USER MODEL
# ============================================================

class User(
    UserMixin,
    db.Model
):

    id = db.Column(
        db.Integer,
        primary_key=True
    )

    username = db.Column(
        db.String(100),
        unique=True
    )

    password = db.Column(
        db.String(255)
    )

    role = db.Column(
        db.String(50)
    )

    # Links patient login account to patient records
    patient_code = db.Column(
        db.String(50),
        nullable=True,
        index=True
    )


# ============================================================
# CREATE DATABASE + SAFE MIGRATION
# ============================================================

with app.app_context():

    db.create_all()

    # --------------------------------------------------------
    # Add patient_code to Patient table if missing
    # --------------------------------------------------------

    try:

        columns = db.session.execute(
            text("PRAGMA table_info(patient)")
        ).fetchall()

        column_names = [
            column[1]
            for column in columns
        ]

        if "patient_code" not in column_names:

            db.session.execute(
                text(
                    "ALTER TABLE patient "
                    "ADD COLUMN patient_code VARCHAR(50)"
                )
            )

            db.session.commit()

    except Exception as e:

        print(
            "Patient migration:",
            repr(e)
        )

        db.session.rollback()


    # --------------------------------------------------------
    # Add patient_code to User table if missing
    # --------------------------------------------------------

    try:

        columns = db.session.execute(
            text("PRAGMA table_info(user)")
        ).fetchall()

        column_names = [
            column[1]
            for column in columns
        ]

        if "patient_code" not in column_names:

            db.session.execute(
                text(
                    "ALTER TABLE user "
                    "ADD COLUMN patient_code VARCHAR(50)"
                )
            )

            db.session.commit()

    except Exception as e:

        print(
            "User migration:",
            repr(e)
        )

        db.session.rollback()


    # --------------------------------------------------------
    # Create default accounts
    # --------------------------------------------------------

    admin = User.query.filter_by(
        username="admin"
    ).first()

    if not admin:

        admin = User(
            username="admin",
            password="admin123",
            role="admin"
        )

        db.session.add(admin)


    doctor = User.query.filter_by(
        username="doctor"
    ).first()

    if not doctor:

        doctor = User(
            username="doctor",
            password="doctor123",
            role="doctor"
        )

        db.session.add(doctor)


    patient_user = User.query.filter_by(
        username="patient"
    ).first()

    if not patient_user:

        patient_user = User(
            username="patient",
            password="patient123",
            role="patient",
            patient_code="PATIENT-001"
        )

        db.session.add(patient_user)

    else:

        # Make sure existing patient account
        # receives a patient code.
        if not patient_user.patient_code:

            patient_user.patient_code = "PATIENT-001"


    db.session.commit()


# ============================================================
# ACCESS CONTROL HELPERS
# ============================================================

def require_role(role):

    if "role" not in session:

        return False

    return session["role"] == role


def patient_records_for_current_user():

    if "patient_code" not in session:

        return []

    patient_code = session.get(
        "patient_code"
    )

    if not patient_code:

        return []

    return Patient.query.filter_by(
        patient_code=patient_code
    ).order_by(
        Patient.date.desc()
    ).all()


# ============================================================
# MEDICAL NOTES
# ============================================================

@app.route(
    "/doctor/patient/<int:patient_id>/notes",
    methods=["GET", "POST"]
)
def medical_notes(patient_id):

    if "role" not in session:

        return redirect(
            "/login"
        )

    if session["role"] not in [
        "doctor",
        "admin"
    ]:

        return "Access Denied"

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

        db.session.add(
            new_note
        )

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


# ============================================================
# MEDICAL REPORT
# ============================================================

@app.route(
    "/doctor/patient/<int:patient_id>/report"
)
def medical_report(patient_id):

    if "role" not in session:

        return redirect(
            "/login"
        )

    patient = Patient.query.get_or_404(
        patient_id
    )

    # Patient can only view their own record
    if session["role"] == "patient":

        if patient.patient_code != session.get(
            "patient_code"
        ):

            return "Access Denied"

    elif session["role"] not in [
        "doctor",
        "admin"
    ]:

        return "Access Denied"


    notes = MedicalNote.query.filter_by(
        patient_id=patient.id
    ).order_by(
        MedicalNote.created_at.desc()
    ).all()


    prescriptions = Prescription.query.filter_by(
        patient_id=patient.id
    ).order_by(
        Prescription.created_at.desc()
    ).all()


    return render_template(
        "medical_report.html",
        patient=patient,
        notes=notes,
        prescriptions=prescriptions
    )


# ============================================================
# PATIENT MEDICAL REPORT
# ============================================================

@app.route(
    "/patient/report/<int:patient_id>"
)
def patient_report(patient_id):

    if "role" not in session:

        return redirect(
            "/login"
        )

    if session["role"] != "patient":

        return "Access Denied"


    patient = Patient.query.get_or_404(
        patient_id
    )


    # SECURITY CHECK
    if patient.patient_code != session.get(
        "patient_code"
    ):

        return "Access Denied"


    notes = MedicalNote.query.filter_by(
        patient_id=patient.id
    ).order_by(
        MedicalNote.created_at.desc()
    ).all()


    prescriptions = Prescription.query.filter_by(
        patient_id=patient.id
    ).order_by(
        Prescription.created_at.desc()
    ).all()


    return render_template(
        "medical_report.html",
        patient=patient,
        notes=notes,
        prescriptions=prescriptions
    )


# ============================================================
# MEDICAL REPORT PDF
# ============================================================

@app.route(
    "/doctor/patient/<int:patient_id>/report/pdf"
)
def medical_report_pdf(patient_id):

    if "role" not in session:

        return redirect(
            "/login"
        )


    patient = Patient.query.get_or_404(
        patient_id
    )


    # --------------------------------------------------------
    # Security
    # --------------------------------------------------------

    if session["role"] == "patient":

        if patient.patient_code != session.get(
            "patient_code"
        ):

            return "Access Denied"

    elif session["role"] not in [
        "doctor",
        "admin"
    ]:

        return "Access Denied"


    notes = MedicalNote.query.filter_by(
        patient_id=patient.id
    ).order_by(
        MedicalNote.created_at.desc()
    ).all()


    prescriptions = Prescription.query.filter_by(
        patient_id=patient.id
    ).order_by(
        Prescription.created_at.desc()
    ).all()


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


    # --------------------------------------------------------
    # Patient information
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "Patient Information",
            heading_style
        )
    )


    patient_data = [

        [
            "Patient ID",
            str(patient.id),
            "Patient Code",
            str(patient.patient_code or "N/A")
        ],

        [
            "Date",
            str(patient.date),
            "Age",
            str(patient.age)
        ],

        [
            "Gender",
            "Male" if patient.gender == 1 else "Female",
            "BMI",
            str(patient.bmi)
        ],

        [
            "Systolic BP",
            f"{patient.systolic} mmHg",
            "Diastolic BP",
            f"{patient.diastolic} mmHg"
        ],

        [
            "Cholesterol",
            str(patient.cholesterol),
            "Risk",
            str(patient.risk)
        ]

    ]


    patient_table = Table(
        patient_data,
        colWidths=[
            90,
            130,
            90,
            170
        ]
    )


    patient_table.setStyle(
        TableStyle([

            (
                "BACKGROUND",
                (0, 0),
                (0, -1),
                colors.HexColor("#e5e7eb")
            ),

            (
                "BACKGROUND",
                (2, 0),
                (2, -1),
                colors.HexColor("#e5e7eb")
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
                (0, -1),
                "Helvetica-Bold"
            ),

            (
                "FONTNAME",
                (2, 0),
                (2, -1),
                "Helvetica-Bold"
            ),

            (
                "FONTSIZE",
                (0, 0),
                (-1, -1),
                9
            ),

            (
                "VALIGN",
                (0, 0),
                (-1, -1),
                "MIDDLE"
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


    story.append(
        patient_table
    )


    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "Heart Disease Prediction",
            heading_style
        )
    )


    prediction_data = [

        [
            "Risk Level",
            str(patient.risk)
        ],

        [
            "Prediction Probability",
            f"{patient.probability}%"
        ]

    ]


    prediction_table = Table(
        prediction_data,
        colWidths=[
            180,
            300
        ]
    )


    prediction_table.setStyle(
        TableStyle([

            (
                "BACKGROUND",
                (0, 0),
                (0, -1),
                colors.HexColor("#e5e7eb")
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
                (0, -1),
                "Helvetica-Bold"
            ),

            (
                "FONTSIZE",
                (0, 0),
                (-1, -1),
                10
            ),

            (
                "TOPPADDING",
                (0, 0),
                (-1, -1),
                8
            ),

            (
                "BOTTOMPADDING",
                (0, 0),
                (-1, -1),
                8
            )

        ])
    )


    story.append(
        prediction_table
    )


    # --------------------------------------------------------
    # Medical notes
    # --------------------------------------------------------

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

            story.append(
                Spacer(
                    1,
                    8
                )
            )

    else:

        story.append(
            Paragraph(
                "No medical notes have been recorded.",
                normal_style
            )
        )


    # --------------------------------------------------------
    # Prescriptions
    # --------------------------------------------------------

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

                str(
                    prescription.medication
                ),

                str(
                    prescription.dosage
                ),

                str(
                    prescription.frequency
                ),

                str(
                    prescription.duration
                )

            ])


        prescription_table = Table(
            prescription_data,
            colWidths=[
                145,
                105,
                110,
                100
            ],
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


        story.append(
            prescription_table
        )

    else:

        story.append(
            Paragraph(
                "No prescriptions have been recorded.",
                normal_style
            )
        )


    story.append(
        Spacer(
            1,
            25
        )
    )


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


    document.build(
        story
    )


    buffer.seek(0)


    return send_file(
        buffer,
        as_attachment=True,
        download_name=(
            f"medical_report_patient_{patient.id}.pdf"
        ),
        mimetype="application/pdf"
    )


# ============================================================
# PRESCRIPTIONS
# ============================================================

@app.route(
    "/doctor/patient/<int:patient_id>/prescription",
    methods=["GET", "POST"]
)
def prescription(patient_id):

    if "role" not in session:

        return redirect(
            "/login"
        )

    if session["role"] not in [
        "doctor",
        "admin"
    ]:

        return "Access Denied"


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


        return redirect(
            url_for(
                "prescription",
                patient_id=patient.id
            )
        )


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


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# ============================================================
# HISTORY PAGE
# ============================================================

@app.route("/history")
def history():

    patients = Patient.query.order_by(
        Patient.date.desc()
    ).all()


    return render_template(
        "history.html",
        patients=patients
    )


# ============================================================
# ANALYTICS DASHBOARD
# ============================================================

@app.route("/dashboard")
def dashboard():

    if not os.path.exists("static"):

        os.makedirs("static")


    patients = Patient.query.all()


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
        if p.age is not None
    ]


    bmi_values = [
        p.bmi
        for p in patients
        if p.bmi is not None
    ]


    cholesterol_values = [
        p.cholesterol
        for p in patients
        if p.cholesterol is not None
    ]


    probabilities = [
        p.probability
        for p in patients
        if p.probability is not None
    ]


    high_risk = Patient.query.filter_by(
        risk="High Risk"
    ).count()


    low_risk = Patient.query.filter_by(
        risk="Low Risk"
    ).count()


    # Risk chart

    plt.figure(
        figsize=(5, 4)
    )

    plt.bar(
        [
            "High Risk",
            "Low Risk"
        ],
        [
            high_risk,
            low_risk
        ],
        color=[
            "red",
            "green"
        ]
    )

    plt.title(
        "Risk Distribution"
    )

    plt.savefig(
        "static/risk_chart.png"
    )

    plt.close()


    # Age chart

    if ages:

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


    # BMI chart

    if bmi_values:

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


    # Cholesterol chart

    if cholesterol_values:

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


    # Heatmap

    data = pd.DataFrame({

        "Age": ages,

        "BMI": bmi_values[:len(ages)],

        "Cholesterol": cholesterol_values[:len(ages)],

        "Probability": probabilities[:len(ages)]

    })


    if len(data) > 1:

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


    return render_template(
        "dashboard.html",
        total_patients=len(patients),
        high_risk=high_risk,
        low_risk=low_risk
    )


# ============================================================
# LOGIN
# ============================================================

@app.route(
    "/login",
    methods=["GET", "POST"]
)
def login():

    if request.method == "POST":

        username = request.form.get(
            "username",
            ""
        ).strip()

        password = request.form.get(
            "password",
            ""
        )


        user = User.query.filter_by(
            username=username
        ).first()


        # ----------------------------------------------------
        # Current project accounts use plaintext passwords.
        # Keep compatibility with your existing database.
        # ----------------------------------------------------

        if user and user.password == password:

            session.clear()

            session["user"] = user.username

            session["role"] = user.role

            session["user_id"] = user.id

            session["patient_code"] = (
                user.patient_code
            )


            if user.role == "admin":

                return redirect(
                    "/admin"
                )


            elif user.role == "doctor":

                return redirect(
                    "/doctor"
                )


            elif user.role == "patient":

                return redirect(
                    "/patient"
                )


        return render_template(
            "login.html",
            error="Invalid username or password."
        )


    return render_template(
        "login.html"
    )


# ============================================================
# LOGOUT
# ============================================================

@app.route("/logout")
def logout():

    session.clear()

    return redirect(
        "/login"
    )


@app.route("/admin")
def admin_dashboard():

    # --------------------------------------------------
    # ADMIN ACCESS CONTROL
    # --------------------------------------------------

    if "role" not in session:
        return redirect("/login")

    if session["role"] != "admin":
        return "Access Denied"

    # --------------------------------------------------
    # GET ALL PATIENT RECORDS
    # --------------------------------------------------

    records = Patient.query.order_by(
        Patient.date.desc()
    ).all()

    # --------------------------------------------------
    # BASIC STATISTICS
    # --------------------------------------------------

    total_patients = len(records)

    high_risk_count = len([
        record for record in records
        if record.risk == "High Risk"
    ])

    low_risk_count = len([
        record for record in records
        if record.risk == "Low Risk"
    ])

    # --------------------------------------------------
    # AVERAGE RISK PROBABILITY
    # --------------------------------------------------

    probabilities = [
        record.probability
        for record in records
        if record.probability is not None
    ]

    if probabilities:
        average_probability = round(
            sum(probabilities) / len(probabilities),
            2
        )
    else:
        average_probability = 0

    # --------------------------------------------------
    # RISK PERCENTAGES
    # --------------------------------------------------

    if total_patients > 0:

        high_risk_percentage = round(
            (high_risk_count / total_patients) * 100,
            1
        )

        low_risk_percentage = round(
            (low_risk_count / total_patients) * 100,
            1
        )

    else:

        high_risk_percentage = 0
        low_risk_percentage = 0

    # --------------------------------------------------
    # RECENT PREDICTIONS
    # --------------------------------------------------

    recent_predictions = records[:10]

    # --------------------------------------------------
    # CHART DATA
    # --------------------------------------------------

    chart_labels = [
        "High Risk",
        "Low Risk"
    ]

    chart_values = [
        high_risk_count,
        low_risk_count
    ]

    # --------------------------------------------------
    # UNIQUE PATIENT CODES
    # --------------------------------------------------

    unique_patient_codes = set()

    for record in records:

        if record.patient_code:
            unique_patient_codes.add(
                record.patient_code
            )

    unique_patient_count = len(
        unique_patient_codes
    )

    # --------------------------------------------------
    # RENDER ADMIN DASHBOARD
    # --------------------------------------------------

    return render_template(
        "admin_dashboard.html",

        records=records,

        recent_predictions=recent_predictions,

        total_patients=total_patients,

        unique_patient_count=unique_patient_count,

        high_risk_count=high_risk_count,

        low_risk_count=low_risk_count,

        average_probability=average_probability,

        high_risk_percentage=high_risk_percentage,

        low_risk_percentage=low_risk_percentage,

        chart_labels=chart_labels,

        chart_values=chart_values
    )


# ============================================================
# DOCTOR DASHBOARD
# ============================================================

@app.route("/doctor")
def doctor_dashboard():

    if "role" not in session:

        return redirect(
            "/login"
        )


    if session["role"] not in [
        "doctor",
        "admin"
    ]:

        return "Access Denied"


    search = request.args.get(
        "search",
        ""
    ).strip()


    risk = request.args.get(
        "risk",
        ""
    ).strip()


    query = Patient.query


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


    if risk:

        query = query.filter(
            Patient.risk == risk
        )


    patients = query.order_by(
        Patient.date.desc()
    ).all()


    return render_template(
        "doctor_dashboard.html",

        patients=patients,

        search=search,

        risk=risk
    )


# ============================================================
# DOCTOR SEARCH
# ============================================================

@app.route(
    "/doctor/search-patients"
)
def doctor_search_patients():

    if "role" not in session:

        return redirect(
            "/login"
        )


    if session["role"] not in [
        "doctor",
        "admin"
    ]:

        return "Access Denied"


    search = request.args.get(
        "search",
        ""
    ).strip()


    risk = request.args.get(
        "risk",
        ""
    ).strip()


    query = Patient.query


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


    if risk:

        query = query.filter(
            Patient.risk == risk
        )


    patients = query.order_by(
        Patient.date.desc()
    ).all()


    return render_template(
        "doctor_search_patients.html",

        patients=patients,

        search=search,

        risk=risk
    )


# ============================================================
# DOCTOR PATIENT HISTORY
# ============================================================

@app.route(
    "/doctor/patient/<int:patient_id>/history"
)
def patient_history(patient_id):

    if "role" not in session:

        return redirect(
            "/login"
        )


    if session["role"] not in [
        "doctor",
        "admin"
    ]:

        return "Access Denied"


    patient = Patient.query.get_or_404(
        patient_id
    )


    if patient.patient_code:

        history_records = Patient.query.filter_by(
            patient_code=patient.patient_code
        ).order_by(
            Patient.date.desc()
        ).all()

    else:

        history_records = [
            patient
        ]


    chart_labels = []

    chart_values = []


    for record in reversed(
        history_records
    ):

        chart_labels.append(

            record.date.strftime(
                "%d %b %Y"
            )

            if record.date

            else "Unknown"

        )


        chart_values.append(
            record.probability or 0
        )


    return render_template(
        "patient_history.html",

        patient=patient,

        history=history_records,

        chart_labels=chart_labels,

        chart_values=chart_values
    )


@app.route("/patient")
def patient_dashboard():

    if "role" not in session:
        return redirect("/login")

    if session["role"] != "patient":
        return "Access Denied"

    patient_code = session.get("patient_code")

    records = Patient.query.filter_by(
        patient_code=patient_code
    ).order_by(
        Patient.date.desc()
    ).all()

    # --------------------------------------------------
    # NO RECORDS
    # --------------------------------------------------

    if not records:
        return render_template(
            "patient_dashboard.html",
            patient_code=patient_code,
            records=[],
            latest=None,
            total_predictions=0,
            high_risk_count=0,
            low_risk_count=0,
            average_probability=0,
            chart_labels=[],
            chart_values=[],
            trend=[],
            notes=[],
            prescriptions=[]
        )

    # --------------------------------------------------
    # LATEST PREDICTION
    # --------------------------------------------------

    latest = records[0]

    # --------------------------------------------------
    # TOTAL PREDICTIONS
    # --------------------------------------------------

    total_predictions = len(records)

    # --------------------------------------------------
    # HIGH / LOW RISK COUNTS
    # --------------------------------------------------

    high_risk_count = len([
        record for record in records
        if record.risk == "High Risk"
    ])

    low_risk_count = len([
        record for record in records
        if record.risk == "Low Risk"
    ])

    # --------------------------------------------------
    # AVERAGE PROBABILITY
    # --------------------------------------------------

    probabilities = [
        record.probability
        for record in records
        if record.probability is not None
    ]

    if probabilities:
        average_probability = round(
            sum(probabilities) / len(probabilities),
            2
        )
    else:
        average_probability = 0

    # --------------------------------------------------
    # CHART DATA
    # --------------------------------------------------

    chart_labels = []
    chart_values = []

    for record in reversed(records):

        chart_labels.append(
            record.date.strftime("%d %b")
            if record.date
            else "Unknown"
        )

        chart_values.append(
            round(record.probability or 0, 2)
        )

    # --------------------------------------------------
    # HEART RISK TREND
    # --------------------------------------------------

    trend = []

    for record in reversed(records):

        trend.append({
            "label": (
                record.date.strftime("%d %b")
                if record.date
                else "Unknown"
            ),
            "value": round(
                record.probability or 0,
                2
            )
        })

    # --------------------------------------------------
    # PATIENT IDs
    # --------------------------------------------------

    patient_ids = [
        record.id
        for record in records
    ]

    # --------------------------------------------------
    # MEDICAL NOTES
    # --------------------------------------------------

    notes = []

    if patient_ids:

        notes = MedicalNote.query.filter(
            MedicalNote.patient_id.in_(patient_ids)
        ).order_by(
            MedicalNote.created_at.desc()
        ).limit(5).all()

    # --------------------------------------------------
    # PRESCRIPTIONS
    # --------------------------------------------------

    prescriptions = []

    if patient_ids:

        prescriptions = Prescription.query.filter(
            Prescription.patient_id.in_(patient_ids)
        ).order_by(
            Prescription.created_at.desc()
        ).limit(5).all()

    # --------------------------------------------------
    # RENDER DASHBOARD
    # --------------------------------------------------

    return render_template(
        "patient_dashboard.html",

        patient_code=patient_code,

        records=records,

        latest=latest,

        total_predictions=total_predictions,

        high_risk_count=high_risk_count,

        low_risk_count=low_risk_count,

        average_probability=average_probability,

        chart_labels=chart_labels,

        chart_values=chart_values,

        trend=trend,

        notes=notes,

        prescriptions=prescriptions
    )


# ============================================================
# PATIENT HISTORY
# ============================================================

@app.route(
    "/patient/history"
)
def patient_history_portal():

    if "role" not in session:

        return redirect(
            "/login"
        )


    if session["role"] != "patient":

        return "Access Denied"


    patient_code = session.get(
        "patient_code"
    )


    records = Patient.query.filter_by(
        patient_code=patient_code
    ).order_by(
        Patient.date.desc()
    ).all()


    return render_template(
        "patient_history.html",

        patient=records[0]
        if records
        else None,

        history=records,

        chart_labels=[
            r.date.strftime("%d %b %Y")
            if r.date
            else "Unknown"
            for r in reversed(records)
        ],

        chart_values=[
            r.probability or 0
            for r in reversed(records)
        ]
    )


# ============================================================
# PATIENT VIEW PRESCRIPTION
# ============================================================

@app.route(
    "/patient/prescriptions"
)
def patient_prescriptions():

    if "role" not in session:

        return redirect(
            "/login"
        )


    if session["role"] != "patient":

        return "Access Denied"


    patient_code = session.get(
        "patient_code"
    )


    records = Patient.query.filter_by(
        patient_code=patient_code
    ).all()


    patient_ids = [
        record.id
        for record in records
    ]


    prescriptions = []


    if patient_ids:

        prescriptions = Prescription.query.filter(
            Prescription.patient_id.in_(
                patient_ids
            )
        ).order_by(
            Prescription.created_at.desc()
        ).all()


    return render_template(
        "patient_prescriptions.html",
        prescriptions=prescriptions
    )


# ============================================================
# PATIENT VIEW NOTES
# ============================================================

@app.route(
    "/patient/notes"
)
def patient_notes():

    if "role" not in session:

        return redirect(
            "/login"
        )


    if session["role"] != "patient":

        return "Access Denied"


    patient_code = session.get(
        "patient_code"
    )


    records = Patient.query.filter_by(
        patient_code=patient_code
    ).all()


    patient_ids = [
        record.id
        for record in records
    ]


    notes = []


    if patient_ids:

        notes = MedicalNote.query.filter(
            MedicalNote.patient_id.in_(
                patient_ids
            )
        ).order_by(
            MedicalNote.created_at.desc()
        ).all()


    return render_template(
        "patient_notes.html",
        notes=notes
    )


# ============================================================
# HEALTH RECOMMENDATIONS
# ============================================================

def get_health_recommendations(
    risk,
    probability
):

    recommendations = []


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


# ============================================================
# PREDICTION
# ============================================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    try:

        # ====================================================
        # GET FORM VALUES
        # ====================================================

        age = float(
            request.form["Age"]
        )

        gender = float(
            request.form["Gender"]
        )

        weight = float(
            request.form["Weight"]
        )

        height = float(
            request.form["Height"]
        )

        bmi = float(
            request.form["BMI"]
        )

        smoking = float(
            request.form["Smoking"]
        )

        alcohol = float(
            request.form["Alcohol_Intake"]
        )

        physical_activity = float(
            request.form["Physical_Activity"]
        )

        diet = float(
            request.form["Diet"]
        )

        stress = float(
            request.form["Stress_Level"]
        )

        hypertension = float(
            request.form["Hypertension"]
        )

        diabetes = float(
            request.form["Diabetes"]
        )

        hyperlipidemia = float(
            request.form["Hyperlipidemia"]
        )

        family_history = float(
            request.form["Family_History"]
        )

        previous_heart_attack = float(
            request.form[
                "Previous_Heart_Attack"
            ]
        )

        systolic = float(
            request.form["Systolic_BP"]
        )

        diastolic = float(
            request.form["Diastolic_BP"]
        )

        heart_rate = float(
            request.form["Heart_Rate"]
        )

        blood_sugar = float(
            request.form[
                "Blood_Sugar_Fasting"
            ]
        )

        cholesterol = float(
            request.form[
                "Cholesterol_Total"
            ]
        )


        # ====================================================
        # CHECK FINITE NUMBERS
        # ====================================================

        values_to_check = [

            age,
            gender,
            weight,
            height,
            bmi,
            smoking,
            alcohol,
            physical_activity,
            diet,
            stress,
            hypertension,
            diabetes,
            hyperlipidemia,
            family_history,
            previous_heart_attack,
            systolic,
            diastolic,
            heart_rate,
            blood_sugar,
            cholesterol

        ]


        if not all(
            np.isfinite(value)
            for value in values_to_check
        ):

            raise ValueError(
                "Please enter valid numeric values."
            )


        # ====================================================
        # VALIDATION
        # ====================================================

        if not 1 <= age <= 120:

            raise ValueError(
                "Age must be between 1 and 120."
            )


        if not 1 <= weight <= 300:

            raise ValueError(
                "Weight must be between 1 and 300 kg."
            )


        if not 50 <= height <= 250:

            raise ValueError(
                "Height must be between 50 and 250 cm."
            )


        if not 10 <= bmi <= 70:

            raise ValueError(
                "BMI must be between 10 and 70."
            )


        binary_values = {

            "Gender": gender,

            "Smoking": smoking,

            "Hypertension": hypertension,

            "Diabetes": diabetes,

            "Hyperlipidemia": hyperlipidemia,

            "Family_History": family_history,

            "Previous_Heart_Attack":
                previous_heart_attack

        }


        for field_name, value in binary_values.items():

            if value not in [0, 1]:

                raise ValueError(
                    f"{field_name} must be either 0 or 1."
                )


        if alcohol not in [
            0,
            1,
            2,
            3
        ]:

            raise ValueError(
                "Alcohol Intake must be between 0 and 3."
            )


        if physical_activity not in [
            0,
            1,
            2
        ]:

            raise ValueError(
                "Physical Activity must be between 0 and 2."
            )


        if diet not in [
            0,
            1,
            2
        ]:

            raise ValueError(
                "Diet must be between 0 and 2."
            )


        if stress not in [
            0,
            1,
            2
        ]:

            raise ValueError(
                "Stress Level must be between 0 and 2."
            )


        if not 60 <= systolic <= 250:

            raise ValueError(
                "Systolic BP must be between 60 and 250."
            )


        if not 30 <= diastolic <= 150:

            raise ValueError(
                "Diastolic BP must be between 30 and 150."
            )


        if not 30 <= heart_rate <= 220:

            raise ValueError(
                "Heart Rate must be between 30 and 220."
            )


        if not 30 <= blood_sugar <= 600:

            raise ValueError(
                "Blood Sugar must be between 30 and 600."
            )


        if not 50 <= cholesterol <= 500:

            raise ValueError(
                "Cholesterol must be between 50 and 500."
            )


        # ====================================================
        # MODEL FEATURES
        # ====================================================

        features = [

            age,

            gender,

            weight,

            height,

            bmi,

            smoking,

            alcohol,

            physical_activity,

            diet,

            stress,

            hypertension,

            diabetes,

            hyperlipidemia,

            family_history,

            previous_heart_attack,

            systolic,

            diastolic,

            heart_rate,

            blood_sugar,

            cholesterol

        ]


        input_data = np.array(
            [features],
            dtype=float
        )


        # ====================================================
        # MODEL CHECK
        # ====================================================

        if hasattr(
            scaler,
            "n_features_in_"
        ):

            if len(features) != scaler.n_features_in_:

                raise ValueError(

                    f"Feature mismatch: application sent "
                    f"{len(features)} features, but scaler expects "
                    f"{scaler.n_features_in_}."

                )


        # ====================================================
        # SCALE
        # ====================================================

        input_scaled = scaler.transform(
            input_data
        )


        # ====================================================
        # PREDICT
        # ====================================================

        prediction = model.predict(
            input_scaled
        )[0]


        probability = None


        if hasattr(
            model,
            "predict_proba"
        ):

            probability = (
                model.predict_proba(
                    input_scaled
                )[0][1]
                * 100
            )


        # ====================================================
        # RISK
        # ====================================================

        if prediction == 1:

            risk = "High Risk"

        else:

            risk = "Low Risk"


        # ====================================================
        # RESULT TEXT
        # ====================================================

        if probability is not None:

            prediction_text = (

                f"Prediction: {risk} | "
                f"Heart Disease Probability: "
                f"{probability:.2f}%"

            )

        else:

            prediction_text = (

                f"Prediction: {risk}"

            )


        # ====================================================
        # GET CURRENT PATIENT CODE
        # ====================================================

        patient_code = None


        if session.get("role") == "patient":

            patient_code = session.get(
                "patient_code"
            )


        # ====================================================
        # SAVE RESULT
        # ====================================================

        patient = Patient(

            patient_code=patient_code,

            age=float(age),

            gender=float(gender),

            bmi=float(bmi),

            systolic=float(systolic),

            diastolic=float(diastolic),

            cholesterol=float(cholesterol),

            risk=risk,

            probability=(

                float(probability)

                if probability is not None

                else None

            ),

            date=datetime.now()

        )


        db.session.add(
            patient
        )

        db.session.commit()


        # ====================================================
        # RECOMMENDATIONS
        # ====================================================

        recommendations = get_health_recommendations(

            risk,

            probability or 0

        )


        # ====================================================
        # PATIENT REDIRECT
        # ====================================================

        if session.get("role") == "patient":

            return redirect(
                url_for(
                    "patient_dashboard"
                )
            )


        # ====================================================
        # NORMAL RESULT
        # ====================================================

        return render_template(

            "index.html",

            prediction_text=prediction_text,

            risk=risk,

            probability=probability,

            recommendations=recommendations

        )


    # ========================================================
    # VALUE ERROR
    # ========================================================

    except ValueError as e:

        print(
            "VALUE ERROR:",
            repr(e)
        )


        return render_template(

            "index.html",

            prediction_text=(
                f"Processing error: {str(e)}"
            )

        )


    # ========================================================
    # OTHER ERROR
    # ========================================================

    except Exception as e:

        print(
            "PREDICTION ERROR:",
            repr(e)
        )


        db.session.rollback()


        return render_template(

            "index.html",

            prediction_text=(
                f"Prediction error: {str(e)}"
            )

        )


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    app.run(

        debug=True,

        use_reloader=False

    )