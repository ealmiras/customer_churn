# Customer Churn Predictions (E-Commerce)

This project analyzes customer churn behavior and builds a machine learning model to predict which customers are likely to churn.

## What is customer churn?
Customer churn refers to customers who stop doing business with a company. 

In an e-commerce context, churn typically means a customer becomes inactive (e.g., no purchases) for a defined period of time.

## Project goals
- Understand the main drivers of churn through exploratory data analysis (EDA)
- Build a baseline churn prediction model
- Improve performance via model comparison and decision threshold tuning
- Produce an interpretable model and clear insights

## Dataset
Source: *https://www.kaggle.com/datasets/ankitverma2010/ecommerce-customer-churn-analysis-and-prediction*

"The data set belongs to a leading online E-Commerce company. (...) Company wants to know the customers who are going to churn, so accordingly they can approach customer to offer some promos."

## Methodology (high level)
1. Data validation & cleaning
2. Exploratory data analysis (EDA)
3. Train/validation split and evaluation metrics selection
4. Baseline model training
5. Model comparison and performance improvement
6. Model interpretation and results reporting

## Final Verdict
A Random Forest model was selected as the final churn predictor due to its superior ranking and decision performance, with a calibrated decision threshold enabling balanced and cost-aware retention actions.

## Project structure
- `data/`       Raw and processed datasets
- `notebooks/`  Exploratory analysis, modeling, and experiments
- `src/`        Reusable preprocessing and modeling code (future work)
- `models/`     Saved model artifacts and inference utilities
- `reports/`    Generated reports and visualizations (HTML/PDF)

Note: This project is currently notebook-driven. The src/ directory is reserved for future refactoring and deployment-oriented extensions.

## Setup
```bash
python -m venv .venv
# source .venv/bin/activate  # macOS/Linux
.venv\Scripts\activate   # Windows

pip install -r requirements.txt
```

## Project Artifacts
The HTML files in this repository are static exports of Python notebooks for easy viewing. 
They are not part of the application code.

The PDF files are also added for quick access.

These artifacts are intended for documentation and review purposes only.