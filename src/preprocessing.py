"""Data preprocessing pipeline for Customer Churn Prediction."""
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import os
from pathlib import Path

FEATURE_COLS = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
    'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
    'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
    'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
    'MonthlyCharges', 'TotalCharges'
]

CATEGORICAL_COLS = [
    'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
    'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
    'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
    'PaperlessBilling', 'PaymentMethod'
]

NUMERICAL_COLS = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']

CATEGORY_OPTIONS = {
    'gender': ['Male', 'Female'],
    'Partner': ['Yes', 'No'],
    'Dependents': ['Yes', 'No'],
    'PhoneService': ['Yes', 'No'],
    'MultipleLines': ['Yes', 'No', 'No phone service'],
    'InternetService': ['DSL', 'Fiber optic', 'No'],
    'OnlineSecurity': ['Yes', 'No', 'No internet service'],
    'OnlineBackup': ['Yes', 'No', 'No internet service'],
    'DeviceProtection': ['Yes', 'No', 'No internet service'],
    'TechSupport': ['Yes', 'No', 'No internet service'],
    'StreamingTV': ['Yes', 'No', 'No internet service'],
    'StreamingMovies': ['Yes', 'No', 'No internet service'],
    'Contract': ['Month-to-month', 'One year', 'Two year'],
    'PaperlessBilling': ['Yes', 'No'],
    'PaymentMethod': ['Electronic check', 'Mailed check',
                      'Bank transfer (automatic)', 'Credit card (automatic)']
}


def load_and_prepare_data(sample_size=50000):
    """Load training data, preprocess, and return splits + encoders."""
    train_path = get_train_data_path()
    if not os.path.exists(train_path):
        raise FileNotFoundError(
            f"Training data not found at {train_path}. "
            "Deploy with artifacts/ and preprocessor.joblib, or include data/raw/train.csv."
        )
    df = pd.read_csv(train_path)

    # Sample for performance
    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)

    # Drop id column
    if 'id' in df.columns:
        df = df.drop('id', axis=1)

    # Handle TotalCharges - convert to numeric
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

    # Encode categoricals
    label_encoders = {}
    df_encoded = df.copy()
    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
        label_encoders[col] = le

    # Convert Churn to int
    if df_encoded['Churn'].dtype == object:
        df_encoded['Churn'] = df_encoded['Churn'].map({'Yes': 1, 'No': 0})
    df_encoded['Churn'] = df_encoded['Churn'].astype(int)

    X = df_encoded[FEATURE_COLS].values
    y = df_encoded['Churn'].values

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test, label_encoders, scaler, df


def get_train_data_path():
    """Return absolute path to train.csv."""
    return os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        'data', 'raw', 'train.csv'
    )


def save_preprocessing_artifacts(label_encoders, scaler, output_path):
    """Persist encoders and scaler for inference."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"label_encoders": label_encoders, "scaler": scaler}, output_path)


def load_preprocessing_artifacts(input_path):
    """Load persisted encoders and scaler."""
    payload = joblib.load(input_path)
    return payload["label_encoders"], payload["scaler"]


def encode_user_input(user_data, label_encoders, scaler):
    """Transform raw user input dict into scaled feature array."""
    row = {}
    for col in FEATURE_COLS:
        if col in CATEGORICAL_COLS:
            le = label_encoders[col]
            val = user_data[col]
            if val in le.classes_:
                row[col] = le.transform([val])[0]
            else:
                row[col] = 0
        else:
            row[col] = float(user_data[col])

    arr = np.array([[row[c] for c in FEATURE_COLS]])
    return scaler.transform(arr)
