# Financial Transaction Purpose Classification

## Overview

This project builds a text classification pipeline for financial transaction descriptions.  
The goal is to predict the `transaction_type` from the `purpose_text` using classical machine learning models.

The project includes:

- synthetic dataset generation
- data preprocessing
- training and comparison of multiple ML models
- evaluation using standard classification metrics
- a minimal REST API for inference
- notes on how a transformer / LLM-based approach could be applied

---

## Setup instructions
### Step 1: Install dependencies
Run:
```bash
pip install -r requirements.txt
```

### Step 2: Generate the dataset
Run:
```bash
python data/data_generation.py
```

This creates:
`data/transactions.csv`

### Step 3: Preprocess the dataset
Run:
```bash
python src/preprocessing.py
```

This creates:
`data/transactions_cleaned.csv`

### Step 4: Train and evaluate the models
Run:
```bash
python src/train_models.py
```

This creates:
`models/best_model.pkl`
`models/vectorizer.pkl`
`models/classifier.pkl`
`reports/model_comparison.csv`
`reports/classification_reports.txt`

### Step 5: Run the REST API
Run:
```bash
uvicorn api.app:app --reload
```

This creates:
Endpoints: `GET /`, `GET /health`, `POST /classify`
UI: http://127.0.0.1:8000/docs

Example request:
POST /classify
```JSON
{
  "purpose_text": "Lidl grocery store purchase"
}
```