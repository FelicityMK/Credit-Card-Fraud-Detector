import pandas as pd

# Load the dataset
df = pd.read_csv('data/creditcard.csv')

# Display the first 5 rows
print("First 5 rows of the dataset:")
print(df.head())

# Basic info
print("\n Dataset Info:")
print(df.info())

# Check for missing values
print("\n Missing values in each column:")
print(df.isnull().sum())

# Class distribution
print("\n Class distribution (0 = Legit, 1 = Fraud):")
print(df['Class'].value_counts())
