# Insurance Claims Fraud Detection API

Flask API for the Insurance Claims Fraud Detection and Triage project.

## Model Results from Notebook

- Best model: XGBoost
- ROC-AUC: 0.8278
- Fraud F1-score: 0.5918
- Fraud precision: 0.5918
- Fraud recall: 0.5918
- HIGH-priority bucket fraud concentration: approximately 93%

## Local Run

```bash
pip install -r requirements.txt
python app.py
```

Open:

```bash
http://localhost:5000/health
```

## Docker Run

```bash
docker build -t insurance-claims-api .
docker run -p 5000:5000 insurance-claims-api
```

## API Endpoints

- `GET /health`
- `GET /info`
- `POST /predict`
- `POST /predict-batch`

## Important

Before running, copy these files from your notebook into the `models/` folder:

- `fraud_model_xgboost.pkl`
- `feature_names.pkl`
- `feature_scaler.pkl` optional for XGBoost
