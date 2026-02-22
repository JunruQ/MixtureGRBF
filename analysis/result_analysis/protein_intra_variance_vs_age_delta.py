# ==============================================================================
# 0. SETUP: LIBRARIES AND CONFIGURATION
# ==============================================================================

import pandas as pd
import numpy as np
import warnings
import sys
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import os
import utils.utils as utils

# --- [!] 用户配置: 文件路径 ---
nsubtype = 5
result_folder = 'ukb_MixtureGRBF_cv_nsubtype_biom17'
output_dir = f'./output/result_analysis/{result_folder}/{nsubtype}_subtypes'

# --- 输入文件路径 ---
SUBTYPE_FILE_PATH = f'./output/{result_folder}/{nsubtype}_subtypes/subtype_stage.csv'
PROTEIN_FILE_PATH = 'input/ukb/ukb_covreg2_trans1_nanf1_biom17.csv'
PREDICTED_STAGE_PATH = 'analysis/result_analysis/stage_pred/ukb_covreg1_trans1_nanf1_biom0_stage_pred.csv'

# --- 输出文件路径 ---
OUTPUT_IMAGE_PATH = output_dir + '/protein_intra_variance_vs_age_delta.png'

# --- 定义非蛋白质数据列 (元数据列) ---
NON_PROTEIN_COLUMNS = ['eid', 'stage', 'sex', 'education', 'centre', 'Ethnic', 'years']

# ==============================================================================
# 1. DATA LOADING AND PREPARATION
# ==============================================================================

print("--- Step 1: Loading and Preparing Data ---")

# --- 加载亚型数据 ---
try:
    subtype_df = pd.read_csv(SUBTYPE_FILE_PATH)
    print(f"✅ Successfully loaded subtype data from: {SUBTYPE_FILE_PATH}")
except FileNotFoundError:
    print(f"❌ ERROR: Subtype file not found at '{SUBTYPE_FILE_PATH}'.")
    sys.exit()

# --- 加载蛋白质数据 ---
try:
    protein_df = pd.read_csv(PROTEIN_FILE_PATH)
    print(f"✅ Successfully loaded protein data from: {PROTEIN_FILE_PATH}")
except FileNotFoundError:
    print(f"❌ ERROR: Protein data file not found at '{PROTEIN_FILE_PATH}'.")
    sys.exit()

# --- 统一ID列名以便合并 ---
if 'PTID' in subtype_df.columns and 'eid' not in subtype_df.columns:
    subtype_df.rename(columns={'PTID': 'eid'}, inplace=True)
    print("Info: Renamed 'PTID' column to 'eid' in subtype data for merging.")
if 'RID' in protein_df.columns and 'eid' not in protein_df.columns:
    protein_df.rename(columns={'RID': 'eid'}, inplace=True)
    print("Info: Renamed 'RID' column to 'eid' in protein data for merging.")

subtype_df = utils.subtype_order_map(subtype_df, result_folder, nsubtype=nsubtype)
subtype_df['subtype_ordered'] = subtype_df['subtype'].astype(int)

# --- 合并数据 ---
merged_df_protein = pd.merge(subtype_df[['eid', 'subtype_ordered']], protein_df, on='eid', how='inner')

if merged_df_protein.empty:
    print("❌ ERROR: Merged DataFrame for protein data is empty.")
    sys.exit()
else:
    print(f"✅ Protein data merged successfully. Found {len(merged_df_protein)} samples.")

# --- 加载预测阶段数据并合并 ---
try:
    df = pd.read_csv(PREDICTED_STAGE_PATH)
    df['stage_delta'] = df['stage_pred'] - df['stage']
    merged_df_age = pd.merge(df, subtype_df[['eid', 'subtype']], left_on='RID', right_on='eid', how='left')
    print(f"✅ Successfully loaded and merged predicted stage data from: {PREDICTED_STAGE_PATH}")
except FileNotFoundError:
    print(f"❌ ERROR: Predicted stage file not found at '{PREDICTED_STAGE_PATH}'.")
    sys.exit()

# ==============================================================================
# 2. REGRESS PROTEINS TO AGE 55 AND CALCULATE STANDARD DEVIATIONS
# ==============================================================================

print("--- Step 2: Regressing Proteins to Age 55 and Calculating Standard Deviations ---")

# --- Identify protein columns ---
protein_columns = [col for col in merged_df_protein.columns if col not in NON_PROTEIN_COLUMNS + ['eid', 'subtype_ordered']]
print(f"Info: Found {len(protein_columns)} protein columns.")

# --- Initialize DataFrame to store regressed protein values ---
regressed_proteins = merged_df_protein[['eid', 'subtype_ordered']].copy()

# --- Regress each protein to age 55 ---
reference_age = 55
for protein in protein_columns:
    # Prepare data for regression
    X = merged_df_protein[['stage']].values
    y = merged_df_protein[protein].values
    
    # Remove rows with NaN values for this protein
    mask = ~np.isnan(y)
    X_clean = X[mask]
    y_clean = y[mask]
    
    if len(X_clean) < 2:
        print(f"⚠️ Warning: Skipping protein '{protein}' due to insufficient non-NaN data.")
        continue
    
    # Fit linear regression model
    model = LinearRegression()
    model.fit(X_clean, y_clean)
    
    # Predict protein level at reference age (55)
    adjustment = model.coef_[0] * (merged_df_protein['stage'] - reference_age)
    regressed_proteins[protein] = merged_df_protein[protein] - adjustment

# --- Calculate standard deviation for each individual across proteins ---
regressed_proteins['std_dev'] = regressed_proteins[protein_columns].std(axis=1)

# --- Check for any issues in standard deviation calculation ---
if regressed_proteins['std_dev'].isna().all():
    print("❌ ERROR: All standard deviations are NaN. Check protein data.")
    sys.exit()
else:
    print(f"✅ Calculated standard deviations for {len(regressed_proteins)} individuals.")

# ==============================================================================
# 3. CREATE COMBINED SUBPLOT WITH TWO BOX PLOTS
# ==============================================================================

print("--- Step 3: Creating Combined Subplot ---")

# --- Ensure output directory exists ---
os.makedirs(output_dir, exist_ok=True)

# --- Set aesthetic parameters ---
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.labelsize': 10,
    'axes.titlesize': 12,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'axes.spines.top': False,
    'axes.spines.right': False
})

# --- Create figure with two subplots ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), sharey=False)

# --- First subplot: Protein Variability Across Subtypes ---
sns.boxplot(
    x='subtype_ordered',
    y='std_dev',
    data=regressed_proteins,
    width=0.35,
    notch=True,
    palette=utils.subtype_colors[:nsubtype],  # Use custom colors for subtypes
    boxprops={'edgecolor': 'black', 'linewidth': 1},
    medianprops={'color': 'black', 'linewidth': 1.5},
    whiskerprops={'color': 'black', 'linewidth': 1},
    capprops={'color': 'black', 'linewidth': 1},
    flierprops={'marker': '+', 'markerfacecolor': 'black', 'markersize': 4, 'alpha': 0.5},
    ax=ax1
)

ax1.set_title('Protein Variability Across Subtypes\n(Adjusted to age 55)', fontsize=12, pad=15)
ax1.set_xlabel('Subtype', fontsize=10, labelpad=8)
ax1.set_ylabel('Standard Deviation of Protein Levels', fontsize=10, labelpad=8)
ax1.grid(True, linestyle='--', alpha=0.2)

# --- Second subplot: Age Delta by Subtype ---
sns.boxplot(
    x='subtype',
    y='stage_delta',
    data=merged_df_age,
    width=0.35,
    notch=True,
    palette=utils.subtype_colors[:nsubtype],
    boxprops={'edgecolor': 'black', 'linewidth': 1},
    medianprops={'color': 'black', 'linewidth': 1.5},
    whiskerprops={'color': 'black', 'linewidth': 1},
    capprops={'color': 'black', 'linewidth': 1},
    flierprops={'marker': '+', 'markerfacecolor': 'black', 'markersize': 4, 'alpha': 0.5},
    ax=ax2
)

# --- Add reference line ---
ax2.axhline(y=0, color='k', linestyle='--', linewidth=0.5)

ax2.set_title('Age Delta by Subtype\n(Pred from Linear Lasso)', fontsize=12, pad=15)
ax2.set_xlabel('Subtype', fontsize=10, labelpad=8)
ax2.set_ylabel('Age Delta (Pred - True)', fontsize=10, labelpad=8)
ax2.grid(True, linestyle='--', alpha=0.2)

# --- Adjust layout ---
plt.tight_layout()
fig.subplots_adjust(bottom=0.1)  # Reduced bottom margin since no significance text

# --- Save the plot ---
plt.savefig(OUTPUT_IMAGE_PATH, dpi=300, bbox_inches='tight')
plt.close()
print(f"✅ Combined box plot saved to: {OUTPUT_IMAGE_PATH}")