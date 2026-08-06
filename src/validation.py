import pandas as pd

def validate(df):
    issues = []
    
    # Check missing values
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if len(missing) > 0:
        issues.append(f"Missing values: {dict(missing)}")
    
    # Check duplicates
    dupes = df.duplicated().sum()
    if dupes > 0:
        issues.append(f"Duplicate rows: {dupes}")
    
    if issues:
        for i in issues:
            print(f"FAIL: {i}")
        return False
    else:
        print("PASS: Validation successful")
        return True

if __name__ == "__main__":
    df = pd.read_csv("data/raw/Telco-Customer-Churn.csv")
    validate(df)