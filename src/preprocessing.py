import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


def load_data(path):
    df = pd.read_csv(path)
    return df


def preprocess(df):
    df = df.copy()

    # Drop unnecessary columns if present
    if "SK_ID_CURR" in df.columns:
        df.drop(columns=["SK_ID_CURR"], inplace=True)

    # Target variable
    target_col = "TARGET"
    y = df[target_col]
    X = df.drop(columns=[target_col])

    # Handle categorical variables
    cat_cols = X.select_dtypes(include=["object"]).columns

    for col in cat_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

    # Fill missing values
    X = X.fillna(X.median())

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scaling (important for DL)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler