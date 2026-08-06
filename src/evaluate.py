import json

def evaluate(threshold_accuracy=0.75, threshold_f1=0.50):
    with open("models/metrics.json", "r") as f:
        results = json.load(f)
    
    with open("models/best_model.txt", "r") as f:
        best_model = f.read().strip()
    
    metrics = results[best_model]
    acc = metrics["accuracy"]
    f1 = metrics["f1"]
    
    print(f"Evaluating {best_model}: accuracy={acc:.4f}, f1={f1:.4f}")
    
    if acc >= threshold_accuracy and f1 >= threshold_f1:
        print("PASS: Model meets deployment criteria")
        return True
    else:
        print("FAIL: Model does not meet criteria")
        return False

if __name__ == "__main__":
    evaluate()