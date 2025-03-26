# Fraud-Detection-System
Develop a fraud detection system using a dataset like the Credit Card Fraud Dataset.

Fraud Detection System using Random Forest & SMOTE

>Project Overview:

This project builds a fraud detection system using Machine Learning with a Random Forest Classifier. It leverages SMOTE (Synthetic Minority Over-sampling Technique) to handle the severe class imbalance in the dataset. The model predicts whether a transaction is legitimate or fraudulent.

>Installation:

pip install pandas numpy scikit-learn imbalanced-learn

>Importing Required Libraries:

import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report, confusion_matrix

from imblearn.over_sampling import SMOTE

import warnings

warnings.filterwarnings('ignore')

>pandas and numpy: Handle data

scikit-learn: Train-test split, RandomForest, and model evaluation

imbalanced-learn: SMOTE for class balancing

warnings.filterwarnings('ignore'): Suppress warnings

>Load the Dataset:

data = pd.read_csv('creditcard.csv')

Reads creditcard.csv into a Pandas DataFrame.

The dataset contains anonymized transaction details.

>Feature Selection:

X = data.drop('Class', axis=1)

y = data['Class']

X contains all features except Class.

y contains the labels (0 = Legitimate, 1 = Fraud).

>Handling Class Imbalance with SMOTE:

smote = SMOTE(random_state=42)

X_resampled, y_resampled = smote.fit_resample(X, y)

Since fraud cases are only 0.17%, the model would be biased.

SMOTE (Synthetic Minority Over-sampling Technique) generates synthetic fraud cases to balance the dataset.

>Splitting Data into Training & Testing Sets:

X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.3, random_state=42
)

Splits the balanced dataset into 70% training and 30% testing.

>Train a Random Forest Classifier:

model = RandomForestClassifier(n_estimators=100, random_state=42)

model.fit(X_train, y_train)

Uses a Random Forest classifier with 100 trees.

Trains the model on the balanced training data.

>Make Predictions:

y_pred = model.predict(X_test)

The trained model makes predictions on the test data.

>Evaluate Model Performance:

print("Classification Report:")

print(classification_report(y_test, y_pred))

print("Confusion Matrix:")

print(confusion_matrix(y_test, y_pred))

The Classification Report shows precision, recall, and F1-score.

The Confusion Matrix provides insight into fraud detection accuracy.

>Output:

High accuracy

Low false positives

Zero false negatives (all frauds detected!)

>User-Input Transaction Testing:

def test_transaction():
   print("\nTest the Fraud Detection System:")
    
  input_data = []

  max_attempts = 5

  num_features = X.shape[1]

  for i in range(max_attempts):
  
  try:
  
  value = float(input(f"Enter value for feature {i + 1}: "))
           
  input_data.append(value)
            
   if len(input_data) == num_features:
  
  break

  except ValueError:
  
  print("❌ Invalid input. Please enter a valid number.")

  input_data += [0] * (num_features - len(input_data))

  input_array = np.array(input_data).reshape(1, -1)

  prediction = model.predict(input_array)[0]

  if prediction == 1:

  print("\n⚠️ FRAUD DETECTED!")
  
  else:

  print("\n✅ Transaction is legitimate.")

if __name__ == "__main__":
    
  test_transaction()

Allows real-time transaction testing by user input.

>Observations:

The model detects fraud with high accuracy.

0 false negatives (No fraud cases missed).

Only 16 false positives (misclassified legitimate transactions).
