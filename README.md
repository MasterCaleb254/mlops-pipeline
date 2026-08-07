# MLOps Pipeline: Customer Churn Prediction

End-to-end MLOps pipeline for customer churn prediction — automated data ingestion, validation, preprocessing, training, evaluation, and deployment via FastAPI + Docker. Includes CI/CD with GitHub Actions.

---

## Architecture
Raw Data → Ingestion → Validation → Preprocessing → Training → Evaluation → Deploy API
│
(if metrics pass)

text

---

## Tech Stack

| Component           | Tool                    |
| ------------------- | ----------------------- |
| Language            | Python 3.10             |
| Data Processing     | Pandas, NumPy           |
| ML                  | scikit-learn            |
| API                 | FastAPI, Uvicorn        |
| Containerization    | Docker                  |
| Testing             | Pytest                  |
| CI/CD               | GitHub Actions          |

---

## Project Structure
mlops-pipeline/
├── data/
│ ├── raw/ # Raw dataset
│ └── processed/
├── src/
│ ├── ingestion.py # Load raw data
│ ├── validation.py # Validate schema & quality
│ ├── preprocessing.py # Clean, encode, scale
│ ├── train.py # Train & select best model
│ ├── evaluate.py # Deployment gate
│ └── deploy.py # FastAPI app
├── models/ # Saved models & artifacts
├── tests/
│ └── test_validation.py # Unit tests
├── .github/workflows/
│ └── ci.yml # CI pipeline
├── Dockerfile
├── requirements.txt
└── README.md

text

---

## Quick Start

### 1. Clone & setup

```bash
git clone https://github.com/MasterCaleb254/mlops-pipeline.git
cd mlops-pipeline
python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows
pip install -r requirements.txt
2. Run the pipeline
bash
python src/ingestion.py      # Load data
python src/validation.py     # Validate
python src/preprocessing.py  # Preprocess
python src/train.py          # Train models
python src/evaluate.py       # Evaluate
3. Start the API
bash
uvicorn src.deploy:app --reload
Endpoints:

GET /health — Health check

POST /predict — Predict churn

GET /metrics — Model metrics

4. Run with Docker
bash
docker build -t churn-api .
docker run -p 8000:8000 churn-api
5. Run tests
bash
pytest tests/ -v
CI/CD
On every push, GitHub Actions:

Sets up Python

Installs dependencies

Runs the test suite

Dataset
Telco Customer Churn — IBM sample dataset, 7,043 customers, 21 features.

License
MIT