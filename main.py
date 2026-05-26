# ============================================
# HEART DISEASE PREDICTION SYSTEM
# FINAL YEAR PROJECT VERSION
# Models:
# 1. XGBoost
# 2. LSTM Deep Learning
# 3. Hybrid Ensemble
# ============================================

import warnings
warnings.filterwarnings("ignore")

# ============================================
# IMPORTS
# ============================================

import pandas as pd
import numpy as np
import joblib

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression

from xgboost import XGBClassifier

# ============================================
# TENSORFLOW / KERAS
# ============================================

try:
    from keras.models import Sequential
    from keras.layers import Dense, LSTM, Dropout

    tensorflow_available = True

except Exception as e:

    tensorflow_available = False

    print("\nTensorFlow Import Error:")
    print(e)

    print("\nLSTM model will be skipped.")

# ============================================
# LOAD DATASET
# ============================================

print("\n===================================")
print("LOADING DATASET...")
print("===================================")

df = pd.read_csv("heart_disease_dataset.csv")

print("\nFIRST 5 ROWS:")
print(df.head())

# ============================================
# HANDLE MISSING VALUES
# ============================================

print("\n===================================")
print("CHECKING MISSING VALUES...")
print("===================================")

print(df.isnull().sum())

# Fill categorical missing values
categorical_fill = [
    "Alcohol_Intake",
    "Smoking",
    "Diet",
    "Stress_Level",
    "Physical_Activity"
]

for col in categorical_fill:

    if col in df.columns:
        df[col] = df[col].fillna("Unknown")

# Fill numeric missing values
numeric_cols = df.select_dtypes(include=np.number).columns

for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

print("\nMISSING VALUES AFTER CLEANING:")
print(df.isnull().sum())

# ============================================
# ENCODE CATEGORICAL DATA
# ============================================

print("\n===================================")
print("ENCODING CATEGORICAL FEATURES...")
print("===================================")

categorical_columns = [
    "Gender",
    "Smoking",
    "Alcohol_Intake",
    "Physical_Activity",
    "Diet",
    "Stress_Level"
]

encoders = {}

for col in categorical_columns:

    if col in df.columns:

        encoder = LabelEncoder()

        df[col] = encoder.fit_transform(
            df[col].astype(str)
        )

        encoders[col] = encoder

print("\nENCODING COMPLETE!")

# ============================================
# FEATURES + TARGET
# ============================================

TARGET_COLUMN = "Heart_Disease"

if TARGET_COLUMN not in df.columns:

    raise Exception(
        f"\nERROR: '{TARGET_COLUMN}' column not found in dataset."
    )

X = df.drop(TARGET_COLUMN, axis=1)
y = df[TARGET_COLUMN]

# ============================================
# TRAIN TEST SPLIT
# ============================================

print("\n===================================")
print("SPLITTING DATA...")
print("===================================")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTRAIN SHAPE:", X_train.shape)
print("TEST SHAPE:", X_test.shape)

# ============================================
# FEATURE SCALING
# ============================================

print("\n===================================")
print("SCALING FEATURES...")
print("===================================")

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save scaler
joblib.dump(scaler, "scaler.pkl")

print("\nSCALER SAVED!")

# ============================================
# XGBOOST MODEL
# ============================================

print("\n===================================")
print("TRAINING XGBOOST MODEL...")
print("===================================")

xgb_model = XGBClassifier(
    n_estimators=150,
    learning_rate=0.1,
    max_depth=6,
    random_state=42,
    eval_metric='logloss'
)

xgb_model.fit(X_train_scaled, y_train)

# Predictions
xgb_predictions = xgb_model.predict(X_test_scaled)

# Accuracy
xgb_accuracy = accuracy_score(
    y_test,
    xgb_predictions
)

print(
    "\nXGBOOST ACCURACY:",
    round(xgb_accuracy * 100, 2),
    "%"
)

# Save model
joblib.dump(xgb_model, "xgb_model.pkl")

print("XGBOOST MODEL SAVED!")

# ============================================
# LSTM MODEL
# ============================================

lstm_accuracy = 0

if tensorflow_available:

    print("\n===================================")
    print("TRAINING LSTM MODEL...")
    print("===================================")

    # Reshape for LSTM
    X_train_lstm = np.reshape(
        X_train_scaled,
        (
            X_train_scaled.shape[0],
            X_train_scaled.shape[1],
            1
        )
    )

    X_test_lstm = np.reshape(
        X_test_scaled,
        (
            X_test_scaled.shape[0],
            X_test_scaled.shape[1],
            1
        )
    )

    # Build model
    lstm_model = Sequential()

    lstm_model.add(
        LSTM(
            64,
            return_sequences=True,
            input_shape=(
                X_train_lstm.shape[1],
                1
            )
        )
    )

    lstm_model.add(Dropout(0.3))

    lstm_model.add(LSTM(32))

    lstm_model.add(Dropout(0.3))

    lstm_model.add(
        Dense(
            1,
            activation='sigmoid'
        )
    )

    # Compile model
    lstm_model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    # Train model
    history = lstm_model.fit(
        X_train_lstm,
        y_train,
        epochs=10,
        batch_size=64,
        validation_split=0.2,
        verbose=1
    )

    # Predict
    lstm_predictions = lstm_model.predict(
        X_test_lstm
    )

    lstm_predictions = (
        lstm_predictions > 0.5
    ).astype(int)

    lstm_accuracy = accuracy_score(
        y_test,
        lstm_predictions
    )

    print(
        "\nLSTM ACCURACY:",
        round(lstm_accuracy * 100, 2),
        "%"
    )

    # Save model
    lstm_model.save("lstm_model.h5")

    print("LSTM MODEL SAVED!")

else:

    print("\nLSTM MODEL SKIPPED!")

# ============================================
# HYBRID ENSEMBLE MODEL
# ============================================

print("\n===================================")
print("TRAINING ENSEMBLE MODEL...")
print("===================================")

ensemble_model = VotingClassifier(

    estimators=[

        (
            'xgb',
            XGBClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42,
                eval_metric='logloss'
            )
        ),

        (
            'lr',
            LogisticRegression(
                max_iter=1000
            )
        )

    ],

    voting='soft'
)

ensemble_model.fit(
    X_train_scaled,
    y_train
)

ensemble_predictions = ensemble_model.predict(
    X_test_scaled
)

ensemble_accuracy = accuracy_score(
    y_test,
    ensemble_predictions
)

print(
    "\nENSEMBLE ACCURACY:",
    round(ensemble_accuracy * 100, 2),
    "%"
)

# Save ensemble model
joblib.dump(
    ensemble_model,
    "ensemble_model.pkl"
)

print("ENSEMBLE MODEL SAVED!")

# ============================================
# MODEL COMPARISON
# ============================================

print("\n===================================")
print("GENERATING VISUALIZATIONS...")
print("===================================")

models = [
    "XGBoost",
    "LSTM",
    "Hybrid Ensemble"
]

accuracies = [
    xgb_accuracy * 100,
    lstm_accuracy * 100,
    ensemble_accuracy * 100
]

# ============================================
# ACCURACY BAR CHART
# ============================================

plt.figure(figsize=(10, 6))

sns.barplot(
    x=models,
    y=accuracies
)

plt.title(
    "Model Accuracy Comparison",
    fontsize=16
)

plt.ylabel("Accuracy (%)")

plt.ylim(0, 100)

for i, value in enumerate(accuracies):

    plt.text(
        i,
        value + 1,
        f"{value:.2f}%",
        ha='center',
        fontsize=11
    )

plt.savefig(
    "model_comparison.png",
    bbox_inches='tight'
)

plt.close()

print("MODEL COMPARISON SAVED!")

# ============================================
# CONFUSION MATRIX
# ============================================

cm = confusion_matrix(
    y_test,
    ensemble_predictions
)

plt.figure(figsize=(7, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues'
)

plt.title(
    "Ensemble Model Confusion Matrix"
)

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.savefig(
    "confusion_matrix.png",
    bbox_inches='tight'
)

plt.close()

print("CONFUSION MATRIX SAVED!")

# ============================================
# CLASSIFICATION REPORT
# ============================================

print("\n===================================")
print("CLASSIFICATION REPORT")
print("===================================")

print(
    classification_report(
        y_test,
        ensemble_predictions
    )
)

# ============================================
# SAVE ENCODERS
# ============================================

joblib.dump(
    encoders,
    "encoders.pkl"
)

print("\nENCODERS SAVED!")

# ============================================
# FINAL OUTPUT
# ============================================

print("\n===================================")
print("ALL MODELS TRAINED SUCCESSFULLY!")
print("===================================")

print("\nSAVED FILES:")

print("✔ scaler.pkl")
print("✔ xgb_model.pkl")
print("✔ ensemble_model.pkl")
print("✔ encoders.pkl")
print("✔ model_comparison.png")
print("✔ confusion_matrix.png")

if tensorflow_available:
    print("✔ lstm_model.h5")

print("\nPROJECT TRAINING COMPLETE!")