# %%
import kagglehub
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
# %%
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

# %% [markdown]
## Exploratory Data Analysis (EDA)

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
# ## EDA Snapshot
# **Dataset Overview:**
# - 5,630 rows
# - 20 columns
# 
# CustomerID → unique ID 
# - Each row represents one customer → *Will be dropped before modeling*
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
# ### Findings
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
# %%
