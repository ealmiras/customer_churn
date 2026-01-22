# %% [markdown]
### Exploratory Data Analysis (EDA)
# The goal of this analysis is to understand customer churn behavior, assess data quality, explore feature distributions, and identify potential predictors of churn.
# This EDA is intended to inform preprocessing decisions, feature engineering, and model selection in later stages.

# %% 
# Load Libraries
import kagglehub
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# %% 
# Set Working Directory
folder_path = os.path.dirname(os.path.realpath(__file__))[:-10] # on .py, set the working directory to the script location
# folder_path = os.getcwd()[:-10] # on jupyter, set the working directory to the notebook location

plt.style.use('ggplot')

# %% 
# Download and Save Dataset (Uncomment to run)
# data_path = kagglehub.dataset_download("ankitverma2010/ecommerce-customer-churn-analysis-and-prediction")
# print("Path to dataset files:", data_path)

# info_df = pd.read_excel(data_path + '\\E Commerce Dataset.xlsx', sheet_name='Data Dict', skiprows=1)[['Data','Variable','Discerption']].rename(columns={'Discerption':'Description'})
# df = pd.read_excel(data_path + '\\E Commerce Dataset.xlsx', sheet_name='E Comm', index_col='CustomerID')

# df.to_csv(folder_path + '\\data\\ecommerce_churn_data.csv')
# info_df.to_csv(folder_path + '\\data\\ecommerce_churn_data_info.csv')

# print("Data and info files saved to data directory.")


# %% 
# Load Dataset
df = pd.read_csv(folder_path + '/data/ecommerce_churn_data.csv', index_col='CustomerID')
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

# %% 
# Churn balance
print("\nChurn Value Counts:")
print(pd.merge(df['Churn'].value_counts(), 
               df['Churn'].value_counts(normalize=True), 
               left_index=True, right_index=True, suffixes=('_count', '_percentage'))
               .rename(columns={'count':'Count', 'proportion':'Percentage'}))

plt.figure(figsize=(6,4))
df['Churn'].value_counts(normalize=True).plot(kind='bar')
plt.title('Churn Distribution')
plt.xlabel('Churn')
plt.ylabel('Percentage')
plt.bar_label(
    plt.gca().containers[0],
    labels=[f"{v*100:.1f}%" for v in plt.gca().containers[0].datavalues]
)
plt.tight_layout()
plt.show()

# %% [markdown]
# #### EDA Snapshot 1
# **Dataset Overview:**
# - 5,630 rows
# - 20 columns
# 
# CustomerID → unique ID 
# - Each row represents a unique customer
# 
# Churn → binary target
# - 1 = churned   
# - 0 = retained
# 
# **Notes:**
# - Several numerical columns have missing values (eg. Tenure, OrderCount, DaySinceLastOrder etc.)
# - Categorical features will need encoding (eg. PreferredPaymentMode, Gender etc.)
# 
# 
# #### Findings
# Churn Value Counts:
# 
#     Churn   Count   Percentage
#     0       4682    83.2 %
#     1       948     16.8 %
# 
# The dataset shows a churn rate of 16.8%, meaning that approximately one in six customers has churned.
# This level of churn is reasonable for an e-commerce context and does not raise immediate concerns about data quality.
# 
# The class distribution is moderately imbalanced, with churned customers representing only 16.8% of the dataset. 
# This imbalance suggests that accuracy alone would be a misleading evaluation metric, and alternative metrics such as recall, precision, and ROC-AUC should be considered.
#
# %% [markdown]
# #### Numeric Features vs Churn
# Good candidates for predictive features will show distinct distributions between churned and retained customers.

# %%
# 1. Tenure
fig, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

ax[0].hist(df['Tenure'].dropna(), bins=30, color='darkgrey')
ax[0].set_title('Tenure Distribution')
ax[0].set_xlabel('Tenure (months)')
ax[0].set_ylabel('Count')

for churn_value, group in df.groupby('Churn'):
    group['Tenure'].plot(kind='kde', label=f'Churn={churn_value}', ax=ax[1])
ax[1].set_title('Distribution of Tenure by Churn')
ax[1].set_xlabel('Tenure (months)')
ax[1].set_ylabel('Density')
ax[1].legend()
ax[1].set_xlim(0, df['Tenure'].max()+1)

fig.tight_layout()
plt.show()

# %% [markdown]
# **Tenure Analysis:**
# - The tenure distribution is right-skewed, with most customers having a tenure of less than 20 months.
# - Churned customers tend to have shorter tenures, with a peak around 5 months, while retained customers have a broader distribution extending to higher tenures.
# - This suggests that customers with shorter tenures are more likely to churn, indicating that tenure could be a significant predictor of churn.

# %%
# 2. OrderCount
fig, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

ax[0].hist(df['OrderCount'].dropna(), bins=30, color='darkgrey')
ax[0].set_title('OrderCount Distribution')  
ax[0].set_xlabel('OrderCount')
ax[0].set_ylabel('Count')

for churn_value, group in df.groupby('Churn'):
    group['OrderCount'].plot(kind='kde', label=f'Churn={churn_value}', ax=ax[1])
ax[1].set_title('Distribution of OrderCount by Churn')
ax[1].set_xlabel('OrderCount')
ax[1].set_ylabel('Density')
ax[1].legend()
ax[1].set_xlim(0, df['OrderCount'].max()+1)

fig.tight_layout()
plt.show()

# %% [markdown]
# **OrderCount Analysis:**
# - The OrderCount distribution is right-skewed, with most customers having placed fewer than 3 orders.
# - No significant difference is observed in the OrderCount distributions between churned and retained customers.
# - This suggests that OrderCount may not be a strong predictor of churn in this dataset but could still be considered in combination with other features.

# %%
# 3. DaySinceLastOrder
fig, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

ax[0].hist(df['DaySinceLastOrder'].dropna(), bins=30, color='darkgrey')
ax[0].set_title('DaySinceLastOrder Distribution')
ax[0].set_xlabel('Days Since Last Order')
ax[0].set_ylabel('Count')

for churn_value, group in df.groupby('Churn'):
    group['DaySinceLastOrder'].plot(kind='kde', label=f'Churn={churn_value}', ax=ax[1])
ax[1].set_title('Distribution of DaySinceLastOrder by Churn')
ax[1].set_xlabel('Days Since Last Order')
ax[1].set_ylabel('Density')
ax[1].legend()
ax[1].set_xlim(0, df['DaySinceLastOrder'].max()+1)

fig.tight_layout()
plt.show()

# %% [markdown]
# **DaySinceLastOrder Analysis:**
# - The DaySinceLastOrder distribution is right-skewed, with most customers having placed an order within the last 10 days.
# - Churned customers tend to have higher values for DaySinceLastOrder, indicating they have not placed an order in a longer time compared to retained customers.
# - This suggests that DaySinceLastOrder may have some predictive power for churn..

# %%
# 4. HourSpendOnApp
fig, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

ax[0].hist(df['HourSpendOnApp'].dropna(), bins=10, color='darkgrey')
ax[0].set_title('HourSpendOnApp Distribution')
ax[0].set_xlabel('Hours Spent on App')
ax[0].set_ylabel('Count')

for churn_value, group in df.groupby('Churn'):
    group['HourSpendOnApp'].plot(kind='kde', label=f'Churn={churn_value}', ax=ax[1])
ax[1].set_title('Distribution of HourSpendOnApp by Churn')
ax[1].set_xlabel('Hours Spent on App')
ax[1].set_ylabel('Density')
ax[1].legend()
ax[1].set_xlim(0, df['HourSpendOnApp'].max()+1)

fig.tight_layout()
plt.show()

# %% [markdown]
# **HourSpendOnApp Analysis:**
# - The HourSpendOnApp distribution is slightly right-skewed, with most customers spending between 1 to 4 hours on the app.
# - No significant difference is observed in the HourSpendOnApp distributions between churned and retained customers.
# - This suggests that HourSpendOnApp shows limited predictive power for churn in isolation.

# %%
# 5. SatisfactionScore
fig, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

ax[0].hist(df['SatisfactionScore'].dropna(), bins=10, color='darkgrey')
ax[0].set_title('SatisfactionScore Distribution')
ax[0].set_xlabel('Satisfaction Score')
ax[0].set_ylabel('Count')

for churn_value, group in df.groupby('Churn'):
    group['SatisfactionScore'].plot(kind='kde', label=f'Churn={churn_value}', ax=ax[1])
ax[1].set_title('Distribution of SatisfactionScore by Churn')
ax[1].set_xlabel('Satisfaction Score')
ax[1].set_ylabel('Density')
ax[1].legend()
ax[1].set_xlim(0, df['SatisfactionScore'].max()+1)

fig.tight_layout()
plt.show()

# %% [markdown]
# **SatisfactionScore Analysis:**
# - The SatisfactionScore distribution is approximately normal, centered around a score of 3.
# - Shows an interesting pattern where churned customers have higher density for higher satisfaction scores (3-5), while retained customers peak around lower scores (1-3).
# - This counterintuitive finding suggests that SatisfactionScore may not be a straightforward predictor of churn and warrants further investigation.
# - This pattern may indicate reporting bias, delayed dissatisfaction, or that satisfaction scores are collected before churn occurs.

# %%
# 6. WarehouseToHome
fig, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

ax[0].hist(df['WarehouseToHome'].dropna(), bins=30, color='darkgrey')
ax[0].set_title('WarehouseToHome Distribution')
ax[0].set_xlabel('Distance from Warehouse to Home (km)')
ax[0].set_ylabel('Count')

for churn_value, group in df.groupby('Churn'):
    group['WarehouseToHome'].plot(kind='kde', label=f'Churn={churn_value}', ax=ax[1])
ax[1].set_title('Distribution of WarehouseToHome by Churn')
ax[1].set_xlabel('Distance from Warehouse to Home (km)')
ax[1].set_ylabel('Density')
ax[1].legend()
ax[1].set_xlim(0, df['WarehouseToHome'].max()+1)

fig.tight_layout()
plt.show()

# %% [markdown]
# **WarehouseToHome Analysis:**
# - The WarehouseToHome distribution is right-skewed, with most customers living within 50 km of the warehouse.
# - No significant difference is observed in the WarehouseToHome distributions between churned and retained customers.
# - This suggests that WarehouseToHome shows limited predictive power for churn in isolation.

# %%
# 7. CouponUsed
fig, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

ax[0].hist(df['CouponUsed'].dropna(), bins=15, color='darkgrey')
ax[0].set_title('CouponUsed Distribution')
ax[0].set_xlabel('Number of Coupons Used')
ax[0].set_ylabel('Count')

for churn_value, group in df.groupby('Churn'):
    group['CouponUsed'].plot(kind='kde', label=f'Churn={churn_value}', ax=ax[1])
ax[1].set_title('Distribution of CouponUsed by Churn')
ax[1].set_xlabel('Number of Coupons Used')
ax[1].set_ylabel('Density')
ax[1].legend()
ax[1].set_xlim(0, df['CouponUsed'].max()+1)

fig.tight_layout()
plt.show()

# %% [markdown]
# **CouponUsed Analysis:**
# - The CouponUsed distribution is right-skewed, with most customers using fewer than 5 coupons.
# - No significant difference is observed in the CouponUsed distributions between churned and retained customers.
# - This suggests that CouponUsed shows limited predictive power for churn in isolation.
# - However, coupon usage may reflect retention interventions rather than organic behavior.

# %%
# 8. CashbackAmount
fig, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

ax[0].hist(df['CashbackAmount'].dropna(), bins=30, color='darkgrey')
ax[0].set_title('CashbackAmount Distribution')
ax[0].set_xlabel('Cashback Amount')
ax[0].set_ylabel('Count')

for churn_value, group in df.groupby('Churn'):
    group['CashbackAmount'].plot(kind='kde', label=f'Churn={churn_value}', ax=ax[1])
ax[1].set_title('Distribution of CashbackAmount by Churn')
ax[1].set_xlabel('Cashback Amount')
ax[1].set_ylabel('Density')
ax[1].legend()
ax[1].set_xlim(0, df['CashbackAmount'].max()+1)

fig.tight_layout()
plt.show()

# %% [markdown]
# **CashbackAmount Analysis:**
# - The CashbackAmount distribution is right-skewed with a long tail toward higher cashback values, with most customers receiving a cashback amount between 100 and 200.
# - No significant difference is observed in the CashbackAmount distributions between churned and retained customers.
# - This suggests that CashbackAmount shows limited predictive power for churn in isolation.

# %% [markdown]
# #### Summary of Numeric Feature Analysis:
#
# 1. Tenure shows clear separation between churned and retained customers, suggesting strong predictive potential.
# 2. DaySinceLastOrder appears strongly associated with churn but may introduce data leakage depending on churn definition.
# 3. OrderCount, HourSpendOnApp, WarehouseToHome, CouponUsed, and CashbackAmount show limited standalone separation but may still contribute predictive value in combination with other features.
# 4. SatisfactionScore displays a counterintuitive pattern that warrants further investigation.

# %%
