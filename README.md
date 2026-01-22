# Loan Risk Assessment & Approval Prediction (Machine Learning)

## Overview
This project implements an **end-to-end machine learning system** to evaluate **loan approval risk** based on applicant demographic, financial, employment, and credit-related data.

The system is designed as a **production-style ML pipeline** and deployed using **Streamlit**, allowing real-time predictions with probability-based decision thresholds.  
It predicts whether a loan application should be **Approved or Rejected**, along with a **confidence score**, making it suitable for fintech and banking use cases.

---

## Key Features
- End-to-end machine learning pipeline (preprocessing + model)
- Probability-based loan approval decisioning
- Threshold-driven risk classification
- Interactive Streamlit web application
- Resume and interview ready fintech project

---

## Dataset
The dataset contains structured applicant-level data across multiple dimensions.

### Applicant & Demographics
- Age  
- Gender  
- Marital Status  
- Number of Dependents  
- Education Level  

### Employment & Income
- Employment Status  
- Employer Category (Private, Government, MNC)  
- Applicant Income  
- Co-applicant Income  

### Financial & Credit Profile
- Credit Score  
- Existing Loans  
- Debt-to-Income (DTI) Ratio  
- Savings  
- Collateral Value  

### Loan Details
- Loan Amount  
- Loan Term  
- Loan Purpose  
- Property Area  

### Target Variable
- **Loan_Approved**
  - `1` → Approved  
  - `0` → Rejected  

---

## Exploratory Data Analysis (EDA)
EDA was conducted to understand the factors influencing loan approval decisions.

Key analysis steps included:
- Distribution analysis of numerical features
- Comparison of approved vs rejected loans using boxplots
- Impact of categorical features using bar charts
- Correlation analysis using heatmaps
- Class imbalance analysis

### Visualizations Used
- Histograms  
- Boxplots  
- Count plots  
- Correlation matrix heatmap  

---

## Data Preprocessing
A structured preprocessing workflow was implemented to ensure consistency between training and inference.

- Separation of numerical and categorical features
- Missing value handling:
  - Mean imputation for numerical features
  - Most frequent imputation for categorical features
- One-Hot Encoding for categorical variables
- Feature scaling using `StandardScaler`
- Stratified train-test split to preserve class distribution

All preprocessing steps are integrated into a **single Scikit-learn pipeline** to prevent data leakage.

---

## Model Architecture
- **Algorithm:** Logistic Regression  

### Why Logistic Regression?
- Strong baseline model for binary classification
- Interpretable coefficients
- Stable probability outputs suitable for risk-based decisions

The model outputs:
- Binary class prediction (Approved / Rejected)
- Probability score used for threshold-based decisions

---

## Decision Logic
Loan approval decisions are made using probability thresholds.

- **Approved:** ≥ 70% probability  
- **Borderline:** 55% – 69% probability  
- **Rejected:** < 55% probability  

This approach closely reflects real-world credit risk assessment systems.

---

## Model Evaluation
The model was evaluated using standard classification metrics.

### Metrics Used
- Accuracy  
- Precision  
- Recall  
- F1-Score  
- Confusion Matrix  
- Classification Report  

### Performance Summary
- **Accuracy:** ~80.5%  
- **Precision:** ~64.8%  
- **Recall:** ~76.7%  
- **F1-Score:** ~70.2%  

The model demonstrates strong recall, making it effective at identifying eligible applicants while minimizing false rejections.

---

## Deployment & Application
The trained pipeline is serialized and deployed using **Streamlit**, enabling:

- Real-time loan risk evaluation
- Interactive user input forms
- Probability visualization and risk categorization
- Clean, production-style UI for demos and presentations

The deployed application ensures that preprocessing during inference exactly matches the training workflow.

---

## Tech Stack
- **Programming Language:** Python  
- **Data Processing:** Pandas, NumPy  
- **Visualization:** Matplotlib, Seaborn  
- **Machine Learning:** Scikit-learn  
- **Deployment:** Streamlit  
- **Development Environment:** Jupyter Notebook, Python scripts  

---

## Why This Project Stands Out
- Demonstrates the complete machine learning lifecycle
- Uses pipeline-based preprocessing (industry best practice)
- Includes deployment with an interactive UI
- Implements probability-driven decision thresholds
- Highly relevant to fintech and banking roles

---

## Future Enhancements
- Model explainability using SHAP
- Advanced models (Random Forest, XGBoost)
- Bias and fairness evaluation
- Feature importance dashboard
- Cloud deployment (Streamlit Cloud / AWS)

---
#### Live Demo: https://lyvmgbcdvkpatdsznacbnv.streamlit.app
---

### 👤 Author
**Ankit Gupta**  
AIML & Data Science Enthusitic
