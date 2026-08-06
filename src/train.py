import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import json

def train():
    # Load preprocessed data
    df = pd.read_csv("data/raw/Telco-Customer-Churn.csv")
    from preprocessing import preprocess
    X, y = preprocess(df)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42)
    }
    
    results = {}
    best_model = None
    best_f1 = 0
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred)
        }
        results[name] = metrics
        print(f"{name}: {metrics}")
        
        # Save best model
        with open(f"models/{name}.pkl", "wb") as f:
            pickle.dump(model, f)
        
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_model = name
    
    # Save best model name and metrics
    with open("models/best_model.txt", "w") as f:
        f.write(best_model)
    with open("models/metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nBest model: {best_model} (F1: {best_f1:.4f})")

if __name__ == "__main__":
    train()