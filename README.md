Credit Card Fraud Detection using Machine Learning
A machine learning app that detects fraudulent transactions using an imbalanced dataset of over 280,000 records.
Powered by XGBoost, balanced with SMOTE, and deployed via Streamlit.

Project Overview

Due to GitHub’s file size restrictions, the full creditcard.csv dataset is not included in this repo.
To run the project locally, download it from Kaggle and place it in the Data/ folder as:
Data/creditcard.csv

Dataset: Kaggle Credit Card Fraud Dataset
Goal: Accurately identify fraudulent transactions with high precision and recall.
Best Performing Model: XGBoost with a ROC AUC score of 0.97

Key Features
1.Tackles extreme class imbalance using SMOTE
2.Trains and compares 4 ML models:
3.Logistic Regression, Random Forest, Gradient Boosting, and XGBoost
4.Applies StandardScaler for consistent feature scaling

Saves best model (xgboost_model.pkl) and scaler (scaler.pkl) for deployment

Models Trained
| Model               | ROC AUC Score | Precision (Fraud) | Recall (Fraud) |
| ------------------- | ------------- | ----------------- | -------------- |
| Logistic Regression | 0.96          | 0.84              | 0.70           |
| Random Forest       | 0.95          | 0.88              | 0.80           |
| **XGBoost**         | **0.98**      | **0.84**          | **0.80**       |
| Gradient Boosting   | 0.97          | 0.20              | 0.86           |

XGBoost was selected as the final model due to its strong balance of precision and recall.

Tech Stack
Python
Pandas, Scikit-learn, Imbalanced-learn, XGBoost
Joblib for model persistence
Streamlit for frontend deployment

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
2.Install dependencies; pip install -r requirements.txt
3.Train the model; python Train_model.py
4.Run the app; streamlit run app.py

Demo
🔗 Try it live on Streamlit (add your Streamlit app link here)
Or run it locally using the steps above.

Saved Artifacts
xgboost_model.pkl – Best performing trained model
scaler.pkl – StandardScaler used during model training

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

Author
Felistas Kandenye
Hybrid Techie & Storyteller ✨ | Python • Cloud • AI
🔗 LinkedIn

License
This project is licensed under the MIT License.
