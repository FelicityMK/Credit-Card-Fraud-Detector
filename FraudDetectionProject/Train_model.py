
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score


#load the data set
df= pd.read_csv('Data/creditcard.csv')

#split the data into features and target
X= df.drop('Class', axis=1)
y= df['Class']

#Split into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

#Apply SMOTE to the training data only
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)


# Step 1: Scale both resampled train and original test sets
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_sm)   #Fit on resampled train
X_test_scaled = scaler.transform(X_test)               # Transform test using same scaler


#check the new class distribution
#print("After SMOTE resampling:")
#print(pd.Series(y_train_sm).value_counts(normalize=True))


#Train Logistic Regression
logreg_model = LogisticRegression(max_iter=2000, solver= 'saga', random_state=42)
logreg_model.fit (X_train_sm, y_train_sm)

# Predict on test data
y_pred_logreg = logreg_model.predict(X_test_scaled)
y_prob_logreg = logreg_model.predict_proba(X_test_scaled)[:, 1]

#Evaluate Logistic Regression
print("\n Logistic Regression:")
print("Confusion Matrix:\n", confusion_matrix(y_test,y_pred_logreg))
print("\nClassification Report:\n", classification_report(y_test, y_pred_logreg))
print("ROC AUC Score:", roc_auc_score(y_test, y_prob_logreg))


#Train and evaluate other models
models = {
    "Random Forest": RandomForestClassifier(n_estimators= 100, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric= 'logloss', random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42)
}
 #Train and evaluate each

results = {}

for name, clf in models.items():
     print(f"\n Training {name}...")
     clf.fit(X_train_scaled, y_train_sm)
     y_pred = clf.predict(X_test_scaled)
     y_prob = clf.predict_proba(X_test_scaled)[:, 1]

     results[name] = {
         "Confusion matrix": confusion_matrix(y_test, y_pred),
         "Classification_report": classification_report(y_test, y_pred, output_dict=True),
         "roc_auc": roc_auc_score(y_test, y_prob)
     }

     # Evaluate performance
     print(f"\n Results for {name}:")
     print("Confusion Matrix: \n", confusion_matrix(y_test, y_pred))
     print("\nClassification Report:\n", classification_report(y_test, y_pred))
     print("ROC AUC Score:", roc_auc_score(y_test,y_prob))




# Save the best performing model (XGBoost) and the scaler
joblib.dump(models["XGBoost"], "xgboost_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("XGBoost model and scaler saved successfully!")




