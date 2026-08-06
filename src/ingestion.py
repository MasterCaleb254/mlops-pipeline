import pandas as pd

def load_data(filepath):
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} rows")
    return df

if __name__ == "__main__":
    df = load_data("data/raw/Telco-Customer-Churn.csv")
    print(df.head())