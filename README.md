Credit Card Fraud Detection App


This project is a machine learning web application designed to detect fraudulent credit card transactions. Built with Python, Streamlit, and scikit-learn, the app utilizes an XGBoost model trained on a real-world dataset to provide accurate and fast fraud predictions.


Project Overview

Due to GitHub’s file size restrictions, the full creditcard.csv dataset is not included in this repo.

To run the project locally, download it from Kaggle and place it in the Data/ folder as: Data/creditcard.csv

Dataset: Kaggle Credit Card Fraud Dataset

Goal: Accurately identify fraudulent transactions with high precision and recall.

Best Performing Model: XGBoost with a ROC AUC score of 0.97


Key Features

1.Tackles extreme class imbalance using SMOTE

2.Trains and compares 4 ML models:Logistic Regression, Random Forest, Gradient Boosting, and XGBoost

4.Applies StandardScaler for consistent feature scaling

5.Saves best model (xgboost_model.pkl) and scaler (scaler.pkl) for deployment



Models Trained

| Model               | ROC AUC Score | Precision (Fraud) | Recall (Fraud) |
| ------------------- | ------------- | ----------------- | -------------- |
| Logistic Regression | 0.96          | 0.84              | 0.70           |
| Random Forest       | 0.95          | 0.88              | 0.80           |
| **XGBoost**         | **0.98**      | **0.84**          | **0.80**       |
| Gradient Boosting   | 0.97          | 0.20              | 0.86           |

XGBoost was selected as the final model due to its strong balance of precision and recall.


Tech Stack

1. Python
2. Pandas, Scikit-learn, Imbalanced-learn, XGBoost
3. Joblib for model persistence
4. Streamlit for frontend deployment


File Structure

Credit-Card-Fraud-Detector/
├── Data/
│   └── creditcard.csv
├── Train_model.py          # Training pipeline
├── app.py                  # Streamlit web app
├── scaler.pkl              # Saved StandardScaler
├── xgboost_model.pkl       # Trained model
├── requirements.txt
└── README.md

How to Run

1.Clone the repository; git clone https://github.com/FelicityMK/Credit-Card-Fraud-Detector.git
cd Credit-Card-Fraud-Detector


2.Install dependencies; pip install -r requirements.txt


3.Train the model; python Train_model.py


4.Run the app; streamlit run app.py


Demo

Try it live on Streamlit (add your Streamlit app link here)

Or run it locally using the steps above.



Saved Artifacts

1. xgboost_model.pkl – Best performing trained model
2. scaler.pkl – StandardScaler used during model training


Model Performance


Best Model: XGBoost

ROC AUC Score: 0.9679

Precision (Fraud): 0.84

Recall (Fraud): 0.80

The dataset is highly imbalanced—only ~0.17% of transactions are fraudulent.

SMOTE rebalancing was essential to improve model performance.



Next Steps

1.Add file upload functionality for real-time predictions

2.Deploy model with FastAPI for integration with other tools

3.Monitor model drift with incoming transaction data


Author; Felistas Kandenye

Hybrid Techie & Storyteller | Python • Cloud • AI

LinkedIn




License

This project is licensed under the MIT License.
