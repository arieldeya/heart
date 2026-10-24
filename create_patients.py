from app import app, db, User

patients = [
    ("patient2", "Patient@123", "PATIENT-002"),
    ("patient3", "Patient@123", "PATIENT-003"),
    ("patient4", "Patient@123", "PATIENT-004"),
]

with app.app_context():
    for username, password, patient_code in patients:
        existing_user = User.query.filter_by(username=username).first()

        if existing_user:
            print("Account already exists:", username)
            continue

        existing_code = User.query.filter_by(patient_code=patient_code).first()

        if existing_code:
            print("Patient code already assigned:", patient_code)
            continue

        user = User(
            username=username,
            password=password,
            role="patient",
            patient_code=patient_code
        )

        db.session.add(user)
        print("Prepared account:", username)

    db.session.commit()

    print("\nPatient accounts:")
    for user in User.query.filter_by(role="patient").all():
        print(user.username, "|", user.patient_code)
