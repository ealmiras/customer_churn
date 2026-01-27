# %% [markdown]
## Baseline Model
# Given a leakage-safe, well-reasoned preprocessing strategy, can we predict churn better than chance in a way that is interpretable and defensible?
#
# Logistic Regression is used as a baseline model to establish a transparent and interpretable performance benchmark.
# 
# *Scikit-learn pipeline is used to learn preprocessing only from the training set, then apply the same transformations to validation/test, then fit the model.*

# %%
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    RocCurveDisplay
)
import matplotlib.pyplot as plt

# %%
# Load data
df = pd.read_csv('../data/ecommerce_churn_data_eda.csv')
print(df.isnull().sum())

# %%
# Define Target & Drop identifier + leakage-prone features
TARGET = 'Churn'
df.drop(columns=['CustomerID', 'DaySinceLastOrder'], inplace=True)

X = df.drop(columns=[TARGET])
y = df[TARGET]

# %%
# Identify numeric vs categorical columns
numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
categorical_features = X.select_dtypes(exclude=["number"]).columns.tolist()

if "CityTier" in numeric_features:
    numeric_features.remove("CityTier")
    categorical_features.append("CityTier")

numeric_features, categorical_features

# %%
# Preprocessing pipelines
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

# %%
# Train/validation split
test_split = 0.2
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y,
    test_size=test_split,
    random_state=42,
    stratify=y
)

# %%
# Define the model pipeline
clf = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", clf)
])

# %%
# Train the model
model.fit(X_train, y_train)

# %%
# Validate the model
y_pred = model.predict(X_valid)
y_proba = model.predict_proba(X_valid)[:, 1]

print(classification_report(y_valid, y_pred))

roc_auc = roc_auc_score(y_valid, y_proba)
print(f"ROC AUC: {roc_auc:.3f}")

# Plot ROC Curve
RocCurveDisplay.from_predictions(y_valid, y_proba)
plt.title("ROC Curve - Baseline Logistic Regression")
plt.show()

# %%
# Feature importance interpretation
ohe = model.named_steps["preprocessor"].named_transformers_["cat"].named_steps["onehot"]
cat_feature_names = ohe.get_feature_names_out(categorical_features)

all_feature_names = numeric_features + cat_feature_names.tolist()

coefs = model.named_steps["classifier"].coef_[0]
coef_df = pd.DataFrame({"feature": all_feature_names, "coef": coefs}).sort_values("coef", ascending=False)

coef_df.head(10), coef_df.tail(10)

# %% [markdown]
# ## Summary
# The baseline Logistic Regression model demonstrates strong discriminatory power (ROC-AUC = 0.885) while prioritizing churn recall. 
# 
# Coefficient analysis aligns with EDA findings, confirming tenure, payment behavior, and product categories as key churn drivers. 
# 
# This model establishes a reliable and interpretable benchmark for further experimentation.

# %%
