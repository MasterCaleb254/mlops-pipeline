import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle

def preprocess(df):
    # Drop customer ID column
    df = df.drop("customerID", axis=1)
    
    # Convert TotalCharges to numeric, coerce errors to NaN
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    
    # Drop rows with missing TotalCharges
    df = df.dropna(subset=["TotalCharges"])
    
    # Encode target: Churn Yes=1, No=0
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    
    # Separate features and target
    X = df.drop("Churn", axis=1)
    y = df["Churn"]
    
    # Label encode categorical columns
    cat_cols = X.select_dtypes(include="object").columns
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        encoders[col] = le
    
    # Scale numeric features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    
    # Save preprocessing objects
    with open("models/encoders.pkl", "wb") as f:
        pickle.dump(encoders, f)
    with open("models/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    
    print(f"Preprocessed: {X_scaled.shape[0]} rows, {X_scaled.shape[1]} features")
    return X_scaled, y

if __name__ == "__main__":
    df = pd.read_csv("data/raw/Telco-Customer-Churn.csv")
    X, y = preprocess(df)
    print(X.head())