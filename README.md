# Telco Customer Churn Prediction with Ensemble Models

This project demonstrates a complete machine learning workflow for predicting customer churn using the Telco customer dataset. It covers data cleaning, feature engineering, and the implementation and evaluation of three powerful ensemble models: **Random Forest**, **XGBoost**, and **LightGBM**.

A key focus of the project is on handling real-world data challenges, such as:
*   **Data Leakage:** The data is split into training and testing sets *before* any preprocessing to ensure the model is evaluated on truly unseen data.
*   **Class Imbalance:** The Synthetic Minority Over-sampling Technique (SMOTE) is used on the training data to address the imbalance between customers who churned and those who stayed.
*   **Feature Engineering:** It correctly applies One-Hot Encoding for categorical features and StandardScaler for numerical features.

The performance of each model is evaluated using a classification report and the ROC-AUC score, providing a clear comparison of their effectiveness on this prediction task.

## How to Run

1. Install the required libraries: `pip install -r requirements.txt`
2. Place the `telco.csv` dataset in the same directory.
3. Run the script: `python ensemble_learning_project.py`
