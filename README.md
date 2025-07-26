# Home Credit Default Risk Prediction

## Overview

This project builds a robust machine learning pipeline to predict the likelihood of a client defaulting on a loan using the **Home Credit Default Risk** dataset. It applies end-to-end steps from data exploration, statistical inference, and model tuning to deployment via a FastAPI REST service.

---

## Project Structure

```text
Home_Credit_Risk/
├── credit_risk.ipynb           # Full EDA, modeling, evaluation, and DNN
├── main.py                     # FastAPI backend with stacked + DNN endpoints
├── Dockerfile                  # Docker containerization config
├── requirements.txt            # Python dependencies
├── stacked_model.pkl           # Final tuned ensemble model
├── dnn_model.h5                # Saved deep neural network
├── preprocessor.pkl            # Fitted preprocessor pipeline
├── test_final_compact.csv      # Final test set features
├── submission.csv              # Final Kaggle submission format
├── sample_submission.csv       # Template submission file
├── utils/                      # Utility modules
└── README.md                   # This file
```
---

## Data Sources

- `application_train.csv` 

- `application_test.csv` 

- `bureau.csv`, `bureau_balance.csv` 

- `previous_application.csv` 

- `credit_card_balance.csv` 

- `installments_payments.csv` 

- `POS_CASH_balance.csv`

---

## Goal

To build a predictive model that estimates the risk of loan default at the time of application, aiding financial institutions in credit decision-making.

---

## Workflow Summary

### Data Preprocessing & Feature Engineering
- Merged external tables using `SK_ID_CURR` 

- Aggregated temporal and categorical behavior features 

- Applied missing value strategies: dropping, median imputation, or KNN where relevant 

- Categorical encoding with one-hot and domain logic

### EDA & Statistical Inference
- Explored key distributions, outliers, and correlations 

- Applied hypothesis tests to validate risk-related assumptions

### Modeling
- Tuned base models:
  - Logistic Regression 

  - Random Forest 

  - XGBoost 

- Applied SMOTE / class weighting to address imbalance 

- Stacked ensemble using XGBoost as meta-learner 

- Trained a deep neural network (`dnn_model.h5`) with:
  - Batch Normalization 

  - Dropout regularization 

  - Early stopping

### Threshold Optimization
- Evaluated model thresholds using:
  - **F1-score** maximization 

  - **Youden’s J** index 

- Selected optimal thresholds to balance recall and precision in imbalanced settings

### Deployment
- Exported models & preprocessing pipeline using `joblib` and `Keras` 

- Deployed via **FastAPI** 

- Dockerized for production readiness (GCP compatible)

---

## Models & Performance

| Model            | ROC-AUC | F1-Score (Optimized) | Notes                           |
|------------------|---------|-----------------------|---------------------------------|
| Logistic Reg.    | ~0.714  | 0.28–0.30              | Strong baseline                 |
| Random Forest    | ~0.737  | 0.30–0.32              | Tuned with GridSearchCV         |
| XGBoost          | ~0.75+  | 0.33–0.34              | Tuned via Bayesian Optimization |
| **Stacked Model**| **~0.76** | **0.35–0.36**          | Final model for deployment      |
| **DNN**          | ~0.73–0.75 | ~0.32–0.34            | Integrated into FastAPI         |

---

## API Usage (FastAPI)

### POST `/predict`

```json
{
  "data": {
    "AMT_INCOME_TOTAL": 202500.0,
    "DAYS_BIRTH": -12005,
    "CODE_GENDER": "F",
    ...
  }
}
```

### Response
``` json
{
  "probability": 0.1792,
  "prediction": 1
}
``` 

### Docker Deployment 
``` bash
# Build the image
docker build -t credit-default-api .

# Run the API on port 8000
docker run -p 8000:8000 credit-default-api
```

## Appendix B: ML Project Checklist
Adapted from Hands-On ML with Scikit-Learn, Keras & TensorFlow (2nd Edition)

| Checklist Item                       | Status                            |
| ------------------------------------ | --------------------------------- |
| Frame the problem                    | ✅ Done                            |
| Get the data                         | ✅ Done (from Kaggle)              |
| Explore the data                     | ✅ Performed full EDA              |
| Prepare the data                     | ✅ Feature engineering, imputation |
| Shortlist promising models           | ✅ LR, RF, XGBoost, DNN            |
| Train and fine-tune the system       | ✅ Bayesian tuning, early stopping |
| Evaluate on the test set             | ✅ ROC-AUC, F1, threshold tuning   |
| Optimize decision threshold          | ✅ Youden’s J & F1-optimal         |
| Interpret model and results          | ✅ SHAP, feature importance        |
| Present final solution               | ✅ Dockerized FastAPI app          |
| Launch, monitor, and maintain system | 🟡 Pending deployment to GCP      |


