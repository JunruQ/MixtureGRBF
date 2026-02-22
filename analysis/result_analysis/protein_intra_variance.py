# ==============================================================================
# 0. SETUP: LIBRARIES AND CONFIGURATION
# ==============================================================================
import pandas as pd
import numpy as np
import warnings
import sys
import seaborn as sns
import os
import utils.utils as utils

# --- [!] 用户配置: 文件路径 ---
nsubtype = 5
result_folder = 'ukb_MixtureGRBF_cv_nsubtype_biom17'
output_dir = f'./output/result_analysis/{result_folder}/{nsubtype}_subtypes'

# --- 输入文件路径 ---
SUBTYPE_FILE_PATH = f'./output/{result_folder}/{nsubtype}_subtypes/subtype_stage.csv'
PROTEIN_FILE_PATH = 'input/ukb/ukb_covreg1_trans1_nanf1_biom17.csv'

# --- 输出文件路径 ---
OUTPUT_IMAGE_PATH = output_dir + '/protein_intra_variance_boxplot.png'

# --- 定义非蛋白质数据列 (元数据列) ---
NON_PROTEIN_COLUMNS = ['eid', 'subtype', 'stage', 'sex', 'education', 'centre', 'Ethnic', 'years']

print("--- Step 1: Loading and Preparing Data ---")

try:
    protein_df = pd.read_csv(PROTEIN_FILE_PATH).rename(columns={'RID': 'eid'})
    print(f"✅ Successfully loaded protein data from: {PROTEIN_FILE_PATH}")
except FileNotFoundError:
    print(f"❌ ERROR: Protein data file not found at '{PROTEIN_FILE_PATH}'.")
    sys.exit()

subtype_df = utils.get_subtype_stage(result_folder, nsubtype, subtype_order=True)
subtype_df.rename(columns={'PTID': 'eid'}, inplace=True)
# --- 合并数据 ---
merged_df = pd.merge(subtype_df[['eid', 'subtype']], protein_df, on='eid', how='inner')
print(f"✅ Data merged successfully. Found {len(merged_df)} samples.")

from sklearn.linear_model import LinearRegression

# --- Identify protein columns ---
protein_columns = [col for col in merged_df.columns if col not in NON_PROTEIN_COLUMNS + ['eid', 'subtype']]
print(f"Info: Found {len(protein_columns)} protein columns.")

## !!! Maybe not need to regress, due to the preprocess of the data where we regress out the age effect

# # --- Initialize DataFrame to store regressed protein values ---
regressed_proteins = merged_df[['eid', 'subtype']].copy()

# --- Regress each protein to age 55 ---
reference_age = 55
for protein in protein_columns:
    # Prepare data for regression
    X = merged_df[['stage']].values  # Age as predictor
    y = merged_df[protein].values     # Protein levels as response
    
    # Remove rows with NaN values for this protein
    mask = ~np.isnan(y)
    X_clean = X[mask]
    y_clean = y[mask]
    
    if len(X_clean) < 2:  # Skip if insufficient data
        print(f"⚠️ Warning: Skipping protein '{protein}' due to insufficient non-NaN data.")
        continue
    
    # Fit linear regression model
    model = LinearRegression()
    model.fit(X_clean, y_clean)
    
    # Predict protein level at reference age (55)
    adjustment = model.coef_[0] * (merged_df['stage'] - reference_age)
    regressed_proteins[protein] = merged_df[protein] - adjustment

# --- Calculate standard deviation for each individual across proteins ---
merged_df['std_dev'] = merged_df[protein_columns].std(axis=1)

merged_df[['eid', 'std_dev']].to_csv(output_dir + '/protein_intra_variance.csv', index=False)