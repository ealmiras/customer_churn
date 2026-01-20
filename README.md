# Customer Churn Predictions (E-Commerce)

This project analyzes customer churn behavior and builds a machine learning model to predict which customers are likely to churn.

## What is customer churn?
Customer churn refers to customers who stop doing business with a company. 

In an e-commerce context, churn typically means a customer becomes inactive (e.g., no purchases for a defined period) or is labeled as churned in the dataset.

**For the purpose of this project the churn definition will be accepted as the prior.**

## Project goals
- Understand the main drivers of churn through exploratory data analysis (EDA)
- Build a baseline churn prediction model
- Improve performance via feature engineering and model tuning
- Produce an interpretable model and clear insights

## Dataset
Source: *https://www.kaggle.com/datasets/ankitverma2010/ecommerce-customer-churn-analysis-and-prediction*

"The data set belongs to a leading online E-Commerce company. (...) Company wants to know the customers who are going to churn, so accordingly they can approach customer to offer some promos."

## Methodology (high level)
1. Data validation & cleaning
2. Exploratory data analysis (EDA)
3. Train/validation split and evaluation metrics selection
4. Baseline model training
5. Feature engineering and model improvement
6. Model interpretation and results reporting

## Project structure
- `data/` — raw and processed datasets
- `notebooks/` — EDA and experiments
- `src/` — reusable code (preprocessing, training, evaluation)
- `models/` — saved model artifacts
- `reports/` — figures and written summaries

## Setup
```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows

pip install -r requirements.txt