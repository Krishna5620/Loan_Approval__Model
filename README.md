# Install Libraries:

```python
pip install pandas numpy scikit-learn matplotlib seaborn
```
![Dashboard Screenshot](https://github.com/RushiSonar123/Loan_Approval_Model/blob/main/library%20install.png)
```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
# >>>>>> ENSURE THIS LINE IS PRESENT <<<<<<
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
```

# Data Understanding and Loading

```python
import pandas as pd

# --- 1. Load the Data ---
try:
    # Use the 'skipinitialspace=True' argument for robust loading of CSVs
    # This helps ignore spaces after the delimiter, which often affects column names.
    df = pd.read_csv('loan_approval_dataset.csv', skipinitialspace=True)
    print("Data loaded successfully.")
except FileNotFoundError:
    print("Error: 'loan_approval_dataset.csv' not found. Please ensure the file is in the correct directory.")
    exit()

# --- 2. Clean Column Names ---
# This ensures column names like ' loan_status' become 'loan_status'
df.columns = df.columns.str.strip()

# --- 3. Drop Identifier Column ---
# The loan_id is an index and provides no predictive value.
# We use try/except in case 'loan_id' was already dropped or incorrectly named.
if 'loan_id' in df.columns:
    df = df.drop('loan_id', axis=1)
    print("Successfully dropped 'loan_id' column.")
else:
    print("'loan_id' column not found, assuming it was already dropped or renamed.")

# --- 4. Initial Inspection ---
print("\n--- Final Data Columns (Check for loan_id absence) ---")
print(df.columns) 

print("\n--- Data Information (Types and Nulls) ---")
df.info()
# ... (rest of the original Step 1)
```
![Dashboard Screenshot](https://github.com/RushiSonar123/Loan_Approval_Model/blob/main/Traing%20logisting%20and%20regression%20model.png)

# Exploratory Data Analysis (EDA)

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Set a style for better visualization
sns.set_style("whitegrid")

# --- 1. Analyze the Target Variable Distribution ---
plt.figure(figsize=(6, 4))
sns.countplot(x='loan_status', data=df)
plt.title('Target Variable Distribution (Loan Status)')
plt.show()

# Check for balance
target_counts = df['loan_status'].value_counts(normalize=True) * 100
print("\n--- Target Variable Distribution ---")
print(target_counts)
if target_counts.min() < 20:
    print("\nWarning: The target classes are imbalanced. Consider using techniques like SMOTE or class weighting later.")


# --- 2. Bivariate Analysis: CIBIL Score vs. Loan Status ---
plt.figure(figsize=(8, 5))
sns.histplot(df, x='cibil_score', hue='loan_status', kde=True, palette='coolwarm')
plt.title('CIBIL Score Distribution by Loan Status')
plt.show()
print("Observation: Approved loans have significantly higher CIBIL scores, suggesting this is a primary predictor.")
```
![Dashboard Screenshot](https://github.com/RushiSonar123/Loan_Approval_Model/blob/main/Loan%20status.png)
![Dashboard Screenshot](https://github.com/RushiSonar123/Loan_Approval_Model/blob/main/Cibil%20score%20distribution.png)
# Data Cleaning and Anomaly Correction

```python
# ----- Clean up Target Variable Values ----
# Remove any leading/trailing whitespace from the loan_status values
df['loan_status'] = df['loan_status'].str.strip()

# --- 4. Target Variable Encoding ---
# Map 'Approved' -> 1, 'Rejected' -> 0
# NOTE: The .map() function now works reliably because the values are clean.
df['loan_status_encoded'] = df['loan_status'].map({'Approved': 1, 'Rejected': 0})

# Verify that no NaNs were introduced in the encoded column
if df['loan_status_encoded'].isnull().any():
    print("FATAL ERROR: NaN found in the encoded target column. Investigate raw data labels.")
    # Show the problematic labels if any exist
    problem_labels = df[df['loan_status_encoded'].isnull()]['loan_status'].unique()
    print(f"Unmapped raw labels found: {problem_labels}")
    # If the project is time-sensitive, you can drop rows with unmapped labels:
    df.dropna(subset=['loan_status_encoded'], inplace=True)
    print("Dropped rows with unmapped target labels.")

y = df['loan_status_encoded']
X = df.drop(['loan_status', 'loan_status_encoded'], axis=1)
```
# Setting up the Preprocessing Pipeline

```python
# Identify column types
# Use 'number' to capture both int64 and float64 (including engineered features like DTI ratio)
all_numerical_features = X.select_dtypes(include=['number']).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

# Define the transformers
# 1. Standard Scaler for ALL Numerical Features
numerical_transformer = StandardScaler()

# 2. One-Hot Encoding for Categorical Features (education, self_employed)
categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

# Create the preprocessor using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, all_numerical_features), # Now scales ALL numerical features
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='drop' # Explicitly drop any non-used columns, ensuring feature count consistency.
)

print("\n--- Preprocessor Setup Complete ---")
print(f"Numerical Features to scale (including engineered): {all_numerical_features}")
print(f"Categorical Features to encode: {categorical_features}")
```
![Dashboard Screenshot](https://github.com/RushiSonar123/Loan_Approval_Model/blob/main/pipeline%20processing.png)

# Model Training, Evaluation, and Tuning

```python
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42,
    stratify=y # Use stratify to ensure equal proportion of Approved/Rejected in both sets
)

print(f"\nTraining set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")
```
![Dashboard Screenshot](https://github.com/RushiSonar123/Loan_Approval_Model/blob/main/Training%20%26%20testing%20set.png)

# Model Pipelines and Initial Training

```python
# --- 1. Define Model Pipelines ---

# A. Logistic Regression Pipeline (Good Baseline)
lr_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(random_state=42, solver='liblinear'))
])

# B. Random Forest Pipeline (Strong Ensemble Model)
rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42, n_estimators=200, max_depth=10, class_weight='balanced')) # Added class_weight due to imbalance
])


# --- 2. Train and Evaluate Logistic Regression ---
print("\n--- Training Logistic Regression Model ---")
lr_pipeline.fit(X_train, y_train)
lr_pred = lr_pipeline.predict(X_test)
lr_proba = lr_pipeline.predict_proba(X_test)[:, 1]

print("\nLogistic Regression Evaluation:")
print(classification_report(y_test, lr_pred, target_names=['Rejected', 'Approved']))
print(f"ROC AUC Score: {roc_auc_score(y_test, lr_proba):.4f}")


# --- 3. Train and Evaluate Random Forest ---
print("\n--- Training Random Forest Model ---")
rf_pipeline.fit(X_train, y_train)
rf_pred = rf_pipeline.predict(X_test)
rf_proba = rf_pipeline.predict_proba(X_test)[:, 1]

print("\nRandom Forest Evaluation:")
print(classification_report(y_test, rf_pred, target_names=['Rejected', 'Approved']))
print(f"ROC AUC Score: {roc_auc_score(y_test, rf_proba):.4f}")
```
![Dashboard Screenshot](https://github.com/RushiSonar123/Loan_Approval_Model/blob/main/Traing%20logisting%20and%20regression%20model.png)

# Final Model Interpretation and Conclusion

```python
# --- 1. Final Model Selection (Assuming Random Forest is the best) ---
best_model = rf_pipeline

# --- 2. Feature Importance Analysis (Only for tree-based models like Random Forest) ---

# Get the list of feature names directly from the ColumnTransformer in the trained pipeline
# 1. Get names of numerical features (which were all scaled, including engineered features)
# The index [0] corresponds to the 'num' transformer defined in Step 3.2
numerical_features_out = best_model.named_steps['preprocessor'].transformers_[0][2]

# 2. Get names of one-hot encoded categorical features
# The 'cat' transformer is accessed by name
categorical_features_out = list(best_model.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(categorical_features))

# Combine them in the correct order to match the importances array
feature_names = numerical_features_out + categorical_features_out

# Get importance from the Random Forest classifier
importances = best_model.named_steps['classifier'].feature_importances_

# Check for array length mismatch before creating DataFrame (for safety)
if len(feature_names) != len(importances):
    print(f"Error: Mismatch in feature names ({len(feature_names)}) and importances ({len(importances)}) length. Please check Step 3.2 again.")
else:
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False).head(15) # Show top 15

    print("\n--- Top 15 Feature Importances from Random Forest ---")
    print(feature_importance_df)

    # --- 3. Visualization ---
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
    plt.title('Feature Importance for Loan Approval Prediction')
    plt.tight_layout()
    plt.show()

    print("\n--- Conclusion ---")
    print("The model is built and evaluated. The Random Forest classifier typically performs better, with CIBIL score, Income, and Loan Amount being the most influential factors in the approval decision.")

```
![Dashboard Screenshot](https://github.com/RushiSonar123/Loan_Approval_Model/blob/main/Top%2015%20Feature%20Importances%20from%20Random%20Forest.png)
![Dashboard Screenshot](https://github.com/RushiSonar123/Loan_Approval_Model/blob/main/feature%20importance%20for%20loan%20approval.png)
![Dashboard Screenshot](https://github.com/RushiSonar123/Loan_Approval_Model/blob/main/conclusion.png)

# Deployment Preparation and Real-World Application

```python
import joblib

final_model = rf_pipeline 

# Define the filename
filename = 'final_loan_approval_model.pkl'

# Save the model
joblib.dump(final_model, filename)

print(f"\nFinal model pipeline saved to: {filename}")
print("This file contains the Random Forest model and all the necessary preprocessing steps.")
```

![Dashboard Screenshot](https://github.com/RushiSonar123/Loan_Approval_Model/blob/main/New%20Deployment%20model.png)

 # Creating a Prediction Function (Simulation)

 ```python
# Load the model back (simulation of a new request)
loaded_model = joblib.load(filename)

# Create a sample applicant (Data MUST match the columns used in training X)
new_applicant_data = pd.DataFrame({
    'no_of_dependents': [2],
    'education': ['Graduate'],
    'self_employed': ['No'],
    'income_annum': [12000000], # High income
    'loan_amount': [30000000],
    'loan_term': [10],
    'cibil_score': [850],       # Excellent score
    'residential_assets_value': [5000000],
    'commercial_assets_value': [1000000],
    'luxury_assets_value': [4000000],
    'bank_asset_value': [2000000],
    # NOTE: Engineered features (total_assets_value, debt_to_income_ratio) are calculated
    # automatically by the saved pipeline as part of the transform process!
})

# Make a prediction (0=Rejected, 1=Approved)
prediction = loaded_model.predict(new_applicant_data)[0]
probability = loaded_model.predict_proba(new_applicant_data)[0, 1]

status = "Approved" if prediction == 1 else "Rejected"

print("\n--- Live Prediction Simulation ---")
print(f"Prediction: {status}")
print(f"Probability of Approval: {probability:.4f}")
```
![Dashboard Screenshot](https://github.com/RushiSonar123/Loan_Approval_Model/blob/main/Final%20Ouput.png)
