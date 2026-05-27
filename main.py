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

from sklearn.model_selection import (
    train_test_split,
    cross_val_score
)

from sklearn.preprocessing import (
    LabelEncoder,
    StandardScaler
)

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from sklearn.ensemble import (
    VotingClassifier,
    RandomForestClassifier
)

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
# ENCODE CATEGORICAL FEATURES
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
# REMOVE DUPLICATES
# ============================================

print("\n===================================")
print("REMOVING DUPLICATES...")
print("===================================")

print("Rows Before:", len(df))

df = df.drop_duplicates()

print("Rows After :", len(df))

# ============================================
# FEATURES + TARGET
# ============================================

TARGET_COLUMN = "Heart_Disease"

if TARGET_COLUMN not in df.columns:

    raise Exception(
        f"ERROR: '{TARGET_COLUMN}' column not found."
    )

# ============================================
# REMOVE DATA LEAKAGE FEATURES
# ============================================

leakage_columns = []

if "Previous_Heart_Attack" in df.columns:
    leakage_columns.append("Previous_Heart_Attack")

print("\nRemoved Leakage Features:")
print(leakage_columns)

X = df.drop(
    [TARGET_COLUMN] + leakage_columns,
    axis=1
)

y = df[TARGET_COLUMN]

# ============================================
# TRAIN / VALIDATION / TEST SPLIT
# ============================================

print("\n===================================")
print("SPLITTING DATA...")
print("===================================")

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

print("TRAIN:", X_train.shape)
print("VALIDATION:", X_val.shape)
print("TEST:", X_test.shape)

# ============================================
# FEATURE SCALING
# ============================================

print("\n===================================")
print("SCALING FEATURES...")
print("===================================")

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_val_scaled = scaler.transform(X_val)

X_test_scaled = scaler.transform(X_test)

joblib.dump(
    scaler,
    "scaler.pkl"
)

print("SCALER SAVED!")

# ============================================
# XGBOOST MODEL
# ============================================

print("\n===================================")
print("TRAINING XGBOOST MODEL...")
print("===================================")

xgb_model = XGBClassifier(
    n_estimators=80,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric="logloss"
)

# ============================================
# CROSS VALIDATION
# ============================================

cv_scores = cross_val_score(
    xgb_model,
    X_train_scaled,
    y_train,
    cv=5,
    scoring="accuracy"
)

print("\nCross Validation Scores:")
print(cv_scores)

print(
    "Mean CV Accuracy:",
    round(cv_scores.mean() * 100, 2),
    "%"
)

# ============================================
# TRAIN XGBOOST
# ============================================

xgb_model.fit(
    X_train_scaled,
    y_train
)

# ============================================
# PREDICTIONS
# ============================================

xgb_predictions = xgb_model.predict(
    X_test_scaled
)

xgb_probabilities = xgb_model.predict_proba(
    X_test_scaled
)[:, 1]

# ============================================
# EVALUATION METRICS
# ============================================

xgb_accuracy = accuracy_score(
    y_test,
    xgb_predictions
)

precision = precision_score(
    y_test,
    xgb_predictions
)

recall = recall_score(
    y_test,
    xgb_predictions
)

f1 = f1_score(
    y_test,
    xgb_predictions
)

roc_auc = roc_auc_score(
    y_test,
    xgb_probabilities
)

print(
    "\nXGBOOST TEST ACCURACY:",
    round(xgb_accuracy * 100, 2),
    "%"
)

print(
    "Precision:",
    round(precision * 100, 2),
    "%"
)

print(
    "Recall:",
    round(recall * 100, 2),
    "%"
)

print(
    "F1 Score:",
    round(f1 * 100, 2),
    "%"
)

print(
    "ROC AUC:",
    round(roc_auc, 4)
)

# ============================================
# SAVE XGBOOST MODEL
# ============================================

joblib.dump(
    xgb_model,
    "xgb_model.pkl"
)

print("XGBOOST MODEL SAVED!")

# ============================================
# FEATURE IMPORTANCE
# ============================================

importance = pd.DataFrame(
    {
        "Feature": X.columns,
        "Importance": xgb_model.feature_importances_
    }
)

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nTOP 10 IMPORTANT FEATURES")
print(importance.head(10))

# ============================================
# FEATURE IMPORTANCE PLOT
# ============================================

plt.figure(figsize=(10, 6))

sns.barplot(
    data=importance.head(10),
    x="Importance",
    y="Feature"
)

plt.title(
    "Top 10 Important Features"
)

plt.savefig(
    "feature_importance.png",
    bbox_inches="tight"
)

plt.close()

print("FEATURE IMPORTANCE SAVED!")

# ============================================
# LSTM MODEL
# ============================================

lstm_accuracy = 0

if tensorflow_available:

    print("\n===================================")
    print("TRAINING LSTM MODEL...")
    print("===================================")

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

    lstm_model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    lstm_model.fit(
        X_train_lstm,
        y_train,
        epochs=10,
        batch_size=64,
        validation_split=0.2,
        verbose=1
    )

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

    lstm_model.save("lstm_model.h5")

    print("LSTM MODEL SAVED!")

else:

    print("\nLSTM MODEL SKIPPED!")

# ============================================
# ENSEMBLE MODEL
# ============================================

print("\n===================================")
print("TRAINING ENSEMBLE MODEL...")
print("===================================")

ensemble_model = VotingClassifier(

    estimators=[

        ('xgb', xgb_model),

        (
            'lr',
            LogisticRegression(
                max_iter=1000
            )
        ),

        (
            'rf',
            RandomForestClassifier(
                n_estimators=200,
                random_state=42
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

ensemble_probabilities = ensemble_model.predict_proba(
    X_test_scaled
)[:, 1]

ensemble_accuracy = accuracy_score(
    y_test,
    ensemble_predictions
)

ensemble_auc = roc_auc_score(
    y_test,
    ensemble_probabilities
)

print(
    "\nENSEMBLE ACCURACY:",
    round(ensemble_accuracy * 100, 2),
    "%"
)

print(
    "ENSEMBLE ROC AUC:",
    round(ensemble_auc, 4)
)

# ============================================
# SAVE ENSEMBLE MODEL
# ============================================

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
# MODEL COMPARISON PLOT
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
print("✔ feature_importance.png")
print("✔ model_comparison.png")
print("✔ confusion_matrix.png")

if tensorflow_available:
    print("✔ lstm_model.h5")

print("\nPROJECT TRAINING COMPLETE!")