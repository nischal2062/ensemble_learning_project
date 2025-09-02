from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, roc_auc_score

df = pd.read_csv('telco.csv')

# --- Initial Exploration & Cleaning ---
print('Dataset Information: ')
df.info()

# Drop the unique identifier column as it provides no value for prediction
if 'customerID' in df.columns:
    df = df.drop('customerID', axis=1)

# Convert TotalCharges to numeric, coercing errors. We will fill NaNs *after* splitting.
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Map the target variable to 0 and 1 for clarity
df['Churn'] = df['Churn'].map({'No': 0, 'Yes': 1})

print('\n Class Distribution: \n', df['Churn'].value_counts())
print('\n Sample Data: \n', df.head())


# --- Train-Test Split ---
# Separate features (X) and target (y)
X = df.drop(columns=['Churn'])
y = df['Churn']

# Split the data BEFORE any preprocessing to prevent data leakage
# Use stratify=y to maintain the same class distribution in train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# --- Preprocessing Pipeline ---

# Identify numerical and categorical features from the training data
numerical_features = X_train.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X_train.select_dtypes(include=['object']).columns

# 1. Impute missing values in TotalCharges
# Calculate median on the training set to avoid data leakage
median_total_charges = X_train['TotalCharges'].median()
X_train['TotalCharges'].fillna(median_total_charges, inplace=True)
X_test['TotalCharges'].fillna(median_total_charges, inplace=True) # Use training median on test set

# 2. Scale numerical features
# Fit the scaler ONLY on the training data
scaler = StandardScaler()
X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])
# Transform the test data using the scaler fitted on the training data
X_test[numerical_features] = scaler.transform(X_test[numerical_features])

# 3. One-Hot Encode categorical features
# This is the correct way to handle nominal categorical data
X_train = pd.get_dummies(X_train, columns=categorical_features, drop_first=True)
X_test = pd.get_dummies(X_test, columns=categorical_features, drop_first=True)

# Align columns to ensure train and test sets have the same features after encoding
X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

# --- Sanitize Feature Names for LightGBM ---
# LightGBM cannot handle special characters in feature names.
X_train.columns = ["".join (c if c.isalnum() else '_' for c in str(x)) for x in X_train.columns]
X_test.columns = ["".join (c if c.isalnum() else '_' for c in str(x)) for x in X_test.columns]


# --- Handle Class Imbalance with SMOTE ---
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print('\nClass Distribution after SMOTE: \n', pd.Series(y_train_resampled).value_counts())


# --- Model Training and Evaluation ---
# Random Forest
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train_resampled, y_train_resampled)
y_pred_rf = rf_model.predict(X_test)
roc_auc_rf = roc_auc_score(y_test, rf_model.predict_proba(X_test)[:, 1])

# XGBoost - use_label_encoder=False is recommended to avoid deprecation warnings
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb_model.fit(X_train_resampled, y_train_resampled)
y_pred_xgb = xgb_model.predict(X_test)
roc_auc_xgb = roc_auc_score(y_test, xgb_model.predict_proba(X_test)[:, 1])

# LightGBM
lgb_model = LGBMClassifier(random_state=42)
lgb_model.fit(X_train_resampled, y_train_resampled)
y_pred_lgb = lgb_model.predict(X_test)
roc_auc_lgb = roc_auc_score(y_test, lgb_model.predict_proba(X_test)[:, 1])

print('\nRandom Forest Report: \n', classification_report(y_test, y_pred_rf))
print('XGBoost Report: \n', classification_report(y_test, y_pred_xgb))
print('Light GBM Report: \n', classification_report(y_test, y_pred_lgb))

print('\nROC-AUC Scores: \n')
print(f'ROC-AUC Random Forest: {roc_auc_rf:.4f}')
print(f'ROC-AUC XGBoost: {roc_auc_xgb:.4f}')
print(f'ROC-AUC LightGBM: {roc_auc_lgb:.4f}')

