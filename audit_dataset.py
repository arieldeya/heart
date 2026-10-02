import pandas as pd
import numpy as np

print("==========================================")
print("HEART DISEASE DATASET LEAKAGE AUDIT")
print("==========================================")

df = pd.read_csv("heart_disease_dataset.csv")

print("\nDataset shape:")
print(df.shape)

print("\nTarget distribution:")
print(df["Heart_Disease"].value_counts())

print("\nTarget percentages:")
print(
    df["Heart_Disease"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

# ==========================================
# NUMERIC CORRELATIONS
# ==========================================

print("\n==========================================")
print("NUMERIC FEATURE CORRELATION WITH TARGET")
print("==========================================")

numeric_df = df.select_dtypes(include=np.number)

correlations = (
    numeric_df
    .corr()["Heart_Disease"]
    .drop("Heart_Disease")
    .sort_values(ascending=False)
)

print(correlations)

# ==========================================
# PREVIOUS HEART ATTACK
# ==========================================

print("\n==========================================")
print("PREVIOUS HEART ATTACK VS TARGET")
print("==========================================")

if "Previous_Heart_Attack" in df.columns:

    print(
        pd.crosstab(
            df["Previous_Heart_Attack"],
            df["Heart_Disease"],
            margins=True
        )
    )

    print("\nRow percentages:")

    print(
        pd.crosstab(
            df["Previous_Heart_Attack"],
            df["Heart_Disease"],
            normalize="index"
        ).round(4)
    )

# ==========================================
# CATEGORICAL FEATURES
# ==========================================

categorical_columns = [
    "Gender",
    "Smoking",
    "Alcohol_Intake",
    "Physical_Activity",
    "Diet",
    "Stress_Level"
]

print("\n==========================================")
print("CATEGORICAL FEATURE DISTRIBUTIONS")
print("==========================================")

for column in categorical_columns:

    if column in df.columns:

        print("\n------------------------------------------")
        print(column)
        print("------------------------------------------")

        print(
            pd.crosstab(
                df[column],
                df["Heart_Disease"],
                normalize="index"
            ).round(4)
        )

# ==========================================
# DUPLICATE CHECK
# ==========================================

print("\n==========================================")
print("DUPLICATE CHECK")
print("==========================================")

print(
    "Duplicate rows:",
    df.duplicated().sum()
)

# ==========================================
# POSSIBLE TARGET-DEFINING PATTERNS
# ==========================================

print("\n==========================================")
print("FEATURE VALUE CHECK")
print("==========================================")

for column in df.columns:

    if column != "Heart_Disease":

        unique_values = df[column].nunique()

        if unique_values <= 10:

            print(
                f"{column}: {unique_values} unique values -> "
                f"{sorted(df[column].dropna().astype(str).unique())}"
            )

print("\n==========================================")
print("AUDIT COMPLETE")
print("==========================================")