Credit Card Fraud Detection using Machine Learning

A machine learning app that detects fraudulent transactions using an imbalanced dataset of over 280,000 records. 
Powered by XGBoost, balanced using SMOTE, and deployed via Streamlit.


 Project Overview

-**Due to GitHub’s file size restrictions, the full `creditcard.csv` dataset is **not included** in this repository.To run the project locally, please download the dataset from [this Kaggle link](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud), and place it inside the `data/` folder as: `creditcard.csv`
- **Dataset:** [Kaggle Credit Card Fraud Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Goal:** Accurately identify fraudulent transactions with high precision and recall.
- **Best Performing Model:** XGBoost with a ROC AUC score of **0.97**.
  

Key Features
- Handles severe class imbalance using ** SMOTE** oversampling.
- Compares four ML models; Logistic Regression, Random Forest, Gradient Boosting and XGBoost.
- Uses **StandardScaler** for consistent scaling.
- Saves trained model ('xgboost_model.pkl') and scaler ('scaler.pkl') for deployment or future
  

Models Trained


| Model               | ROC AUC Score | Precision (Fraud) | Recall (Fraud) |
|---------------------|---------------|-------------------|----------------|
| Logistic Regression | 0.96          | 0.84              | 0.70           |
| Random Forest       | 0.95          | 0.88              | 0.80           |
| **XGBoost**         | **0.98**      | **0.84**          | **0.80**       |
| Gradient Boosting   | 0.97          | 0.20              | 0.86           |

> **XGBoost** selected as the final model due to its excellent balance of precision and recall.


 Tech Stack
- **Python**
- **Pandas**, **Scikit-learn**, **Imbalanced-learn**, **XGBoost**
- **Joblib** for model persistence
  

File Structure

Credit-Card-Fraud-Detector/
│
├── Data/
│   └── creditcard.csv
├── Train_model.py        # Full training pipeline
├── app.py                # Streamlit app
├── scaler.pkl            # Saved StandardScaler
├── xgboost_model.pkl     # Trained model
├── requirements.txt
└── README.md




How to Run

1. Clone the repo: https://github.com/felicitymk/CreditCardFraudDetection.git
2. Install dependencies:pip install -r requirements.txt
3. Run the training script:python Train_model.py

Demo
-Try it live
-🔗 Streamlit App Link Here]
-Or clone and run locally:
git clone https://github.com/FelicityMK/Credit-Card-Fraud-Detector.git
cd Credit-Card-Fraud-Detector
pip install -r requirements.txt
streamlit run app.py



Next Steps

- [ ] Create a Streamlit app to allow users to upload transactions for real-time prediction.
- [ ] Deploy model via FastAPI for integration with frontend tools.
- [ ] Monitor model drift with new transaction data.
- [ ] 

Author
**Felistas Kandenye**  
Hybrid Techie & Storyteller/Writer | Python + Cloud + AI | [LinkedIn](https://linkedin.com/Felistasmuthoni)


 Saved Artifacts
xgboost_model.pkl: Trained XGBoost model
scaler.pkl: StandardScaler used for test data normalization


Model Performance
Best performing model: XGBoost
ROC AUC Score: 0.9679


Notes
The dataset is highly imbalanced. SMOTE helps rebalance before model training.
Only the best-performing model and scaler are saved for later inference.


License
MIT License
