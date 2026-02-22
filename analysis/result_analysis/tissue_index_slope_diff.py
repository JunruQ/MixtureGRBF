import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
import textwrap
from sklearn.preprocessing import StandardScaler
import utils.utils as utils

# 文件路径和参数保持不变
nsubtype = 5
BLOOD_CHEM_PATH = 'data/ClinicalLabData.csv'
OTHER_IND_PATH = 'data/prot_Modifiable_bl_data.csv'
INPUT_TABLE_PATH = 'input/ukb/ukb_covreg1_trans1_nanf1_biom17.csv'
exp_name = 'ukb_MixtureGRBF_cv_nsubtype_biom17'
CLASSIFICATION_PATH = 'analysis/result_analysis/clinical_lab_data_dict.csv'
OUTPUT_DIR = f'output/result_analysis/{exp_name}/{nsubtype}_subtypes'

palette = ['#0072BD', '#D95319', '#EDB120', '#7E2F8E', '#77AC30']
gray = '#777777'

# df = pd.read_csv(INPUT_TABLE_PATH)
df = utils.get_subtype_stage_with_cov(exp_name, nsubtype)
# df = pd.merge(df, subtype_stage[['PTID', 'subtype']], left_on='RID', right_on='PTID', how='left')

classification = pd.read_csv(CLASSIFICATION_PATH)
classification = classification[['Field ID', 'Category', 'Abbreviation']]
classification['Field ID'] = classification['Field ID'].astype(str)
abbr_map = dict(zip(classification['Field ID'], classification['Abbreviation']))

blood_chem = pd.read_csv(BLOOD_CHEM_PATH)
blood_chem_cols = ['eid'] + [abbr_map[col.replace('-0.0', '')] for col in blood_chem.columns if col != 'eid']
blood_chem.columns = blood_chem_cols

common_cols = df.columns.intersection(blood_chem.columns).difference(['PTID', 'eid'])
blood_chem_for_merge = blood_chem.drop(columns=common_cols)
df = pd.merge(df, blood_chem_for_merge, how='left', left_on='PTID', right_on='eid')

other_ind = pd.read_csv(OTHER_IND_PATH)
common_cols = df.columns.intersection(other_ind.columns).difference(['PTID', 'eid'])
other_ind_for_merge = other_ind.drop(columns=common_cols)
df = pd.merge(df, other_ind_for_merge, how='left', left_on='PTID', right_on='eid')

# Original indicators list with mixed abbreviations and full names
indicators = ['WBC', 'RBC', 'MPV', 'CRP', 
              'ALP', 'ALT', 'AST', 'GGT', 'Total bilirubin', 'Albumin',
              'Pulse rate', 'Systolic blood pressure', 'Diastolic blood pressure', 
              'Forced vital capacity', 'Forced expiratory volume in 1-second', 'Peak expiratory flow',
              'Creatinine', 'Cystatin C', 'Sodium in urine',
              'Glucose', 'Cholesterol', 'HDL-C', 'LDL-C', 'Triglycerides']

# Define a mapping from abbreviations to full names (based on common clinical terms)
abbr_to_full_name = {
    'WBC': 'White blood cell count',
    'RBC': 'Red blood cell count',
    'MPV': 'Mean platelet volume',
    'CRP': 'C-reactive protein',
    'ALP': 'Alkaline phosphatase',
    'ALT': 'Alanine aminotransferase',
    'AST': 'Aspartate aminotransferase',
    'GGT': 'Gamma-glutamyl transferase',
    'HDL-C': 'High-density lipoprotein cholesterol',
    'LDL-C': 'Low-density lipoprotein cholesterol'
}

# Create a list of full names for plotting, keeping existing full names as-is
full_name_indicators = [abbr_to_full_name.get(ind, ind) for ind in indicators]

n_inds = len(indicators)

# def perform_regression(df, indicator, subtype_col='subtype', stage_col='stage'):
#     results = {}
#     for subtype in sorted(df[subtype_col].dropna().unique()):
#         subset = df[df[subtype_col] == subtype].dropna(subset=[indicator, stage_col])
#         if len(subset) > 1:
#             slope, intercept, r_value, p_value, std_err = stats.linregress(
#                 subset[stage_col], subset[indicator]
#             )
#             results[subtype] = {
#                 'slope': slope,
#                 'intercept': intercept  # Store intercept for plotting
#             }
#     return results

# # Perform regression for each indicator
# all_results = {}
# for indicator in indicators:
#     if indicator in df.columns:
#         reg_results = perform_regression(df, indicator)
#         all_results[indicator] = reg_results

# # Create grid of subplots: rows = indicators, columns = subtypes
# fig, axes = plt.subplots(nrows=n_inds, ncols=nsubtype, figsize=(nsubtype * 3, n_inds * 3), 
#                          sharex='col', sharey='row')

# # Ensure axes is 2D even if n_inds or nsubtype is 1
# if n_inds == 1:
#     axes = axes.reshape(1, -1)
# elif nsubtype == 1:
#     axes = axes.reshape(-1, 1)

unique_subtypes = sorted(df['subtype'].dropna().unique())

# # Plot scatter and regression lines
# for i, (indicator, full_name) in enumerate(zip(indicators, full_name_indicators)):
#     if indicator in df.columns:
#         for j, subtype in enumerate(unique_subtypes):
#             ax = axes[i, j]
#             subset = df[df['subtype'] == subtype].dropna(subset=[indicator, 'stage'])
            
#             if len(subset) > 1 and subtype in all_results.get(indicator, {}):
#                 # Scatter plot
#                 ax.scatter(subset['stage'], subset[indicator], 
#                           color=palette[j % len(palette)], alpha=0.5, s=10)
                
#                 # Regression line
#                 slope = all_results[indicator][subtype]['slope']
#                 intercept = all_results[indicator][subtype]['intercept']
#                 x = np.array([subset['stage'].min(), subset['stage'].max()])
#                 ax.plot(x, intercept + slope * x, '-', 
#                         color=palette[j % len(palette)], linewidth=2)
            
#             # Set labels with full names
#             if i == n_inds - 1:  # Bottom row
#                 ax.set_xlabel('Stage')
#             if j == 0:  # Leftmost column
#                 ax.set_ylabel(full_name)  # Use full name here
#             if i == 0:  # Top row
#                 ax.set_title(f'S{subtype}')

# plt.tight_layout()
# plt.savefig(f'{OUTPUT_DIR}/scatter_regression_grid.png', dpi=300, bbox_inches='tight')
# plt.close()

print(df[indicators].dropna(how='all').shape)

# Z-score normalization function
def zscore_normalize(df, column):
    scaler = StandardScaler()
    return scaler.fit_transform(df[[column]].values.reshape(-1, 1)).flatten()

# Perform regression with sex, stage, and group as explanatory variables
def perform_group_regression(df, indicator, subtype, subtype_col='subtype', 
                             stage_col='stage', sex_col='sex', center_col='centre'):
    # 1. Z-score 标准化
    df[indicator + '_z'] = zscore_normalize(df, indicator)
    
    # 2. 创建组别变量 (当前亚型 vs 其他)
    df['group'] = (df[subtype_col] == subtype).astype(int)
    
    # 3. 准备基础协变量列表
    base_covariates = [sex_col, stage_col, 'group']
    
    # 4. 处理 center 变量：转换为 Dummy 变量
    # drop_first=True 会自动去掉一个参照组，防止多重共线性
    center_dummies = pd.get_dummies(df[center_col], prefix='center', drop_first=True).astype(int)
    
    # 5. 合并数据并清洗缺失值
    # 注意：这里将原始列和 dummy 列合并后再统一 dropna
    data_for_reg = pd.concat([df[[indicator + '_z'] + base_covariates], center_dummies], axis=1)
    subset = data_for_reg.dropna()
    
    # 6. 检查有效样本量 (样本量需大于特征数)
    if len(subset) <= (len(base_covariates) + center_dummies.shape[1] + 1):
        return None
    
    # 7. 定义自变量 X 和因变量 y
    # X 包含：常量、性别、分期、组别、以及所有的 center dummy 变量
    X = subset.drop(columns=[indicator + '_z'])
    X = sm.add_constant(X)
    y = subset[indicator + '_z']
    
    # 8. 拟合模型
    try:
        model = sm.OLS(y, X).fit()
        
        # 提取 'group' 的系数和 P 值
        beta = model.params['group']
        p_value = model.pvalues['group']
        
        return {
            'beta': beta, 
            'p_value': p_value,
            'n_obs': len(subset) # 建议记录样本量
        }
    except Exception as e:
        print(f"Regression failed: {e}")
        return None

# Run regressions for all indicators and subtypes
regression_results = {}
for indicator in indicators:
    if indicator in df.columns:
        regression_results[indicator] = {}
        for subtype in unique_subtypes:
            result = perform_group_regression(df, indicator, subtype)
            if result:
                regression_results[indicator][subtype] = result

# Prepare data for dot plot
beta_values = np.zeros((n_inds, nsubtype))
p_values = np.zeros((n_inds, nsubtype))
for i, indicator in enumerate(indicators):
    if indicator in regression_results:
        for j, subtype in enumerate(unique_subtypes):
            if subtype in regression_results[indicator]:
                beta_values[i, j] = regression_results[indicator][subtype]['beta']
                p_values[i, j] = regression_results[indicator][subtype]['p_value']
            else:
                beta_values[i, j] = np.nan
                p_values[i, j] = 1  # Non-significant if no result

# Calculate -log10(p-value) for size
log_p_values = -np.log10(p_values)
log_p_values = np.clip(log_p_values, 0, 200)  # Cap at 5 for visualization

# Create a DataFrame for the dot plot
plot_data = pd.DataFrame({
    'Subtype': np.tile([f'S{subtype}' for subtype in unique_subtypes], n_inds),
    'Indicator': np.repeat(full_name_indicators, nsubtype),
    'Beta': beta_values.flatten(),
    'LogP': log_p_values.flatten()
})

# Define the category mapping based on the provided groupings
category_mapping = {
    'White blood cell count': 'Immune',
    'Red blood cell count': 'Immune',
    'Mean platelet volume': 'Immune',
    'C-reactive protein': 'Immune',
    'Alkaline phosphatase': 'Hepatic',
    'Alanine aminotransferase': 'Hepatic',
    'Aspartate aminotransferase': 'Hepatic',
    'Gamma-glutamyl transferase': 'Hepatic',
    'Total bilirubin': 'Hepatic',
    'Albumin': 'Hepatic',
    'Pulse rate': 'Cardiovascular',
    'Systolic blood pressure': 'Cardiovascular',
    'Diastolic blood pressure': 'Cardiovascular',
    'Forced vital capacity': 'Pulmonary',
    'Forced expiratory volume in 1-second': 'Pulmonary',
    'Peak expiratory flow': 'Pulmonary',
    'Creatinine': 'Renal',
    'Cystatin C': 'Renal',
    'Sodium in urine': 'Renal',
    'Glucose': 'Metabolic',
    'Cholesterol': 'Metabolic',
    'High-density lipoprotein cholesterol': 'Metabolic',
    'Low-density lipoprotein cholesterol': 'Metabolic',
    'Triglycerides': 'Metabolic'
}

# Create a DataFrame for the dot plot
plot_data = pd.DataFrame({
    'Subtype': np.tile([f'{subtype}' for subtype in unique_subtypes], n_inds),
    'Indicator': np.repeat(full_name_indicators, nsubtype),
    'Beta': beta_values.flatten(),
    'LogP': log_p_values.flatten()
})

# Remove rows with NaN beta values
plot_data = plot_data.dropna(subset=['Beta'])

# Add the Category column based on the mapping
plot_data['Category'] = plot_data['Indicator'].map(category_mapping)

# Save the data to a CSV file
csv_path = f'{OUTPUT_DIR}/tissue_index_slope_diff.csv'
plot_data.to_csv(csv_path, index=False)
print(f"Data saved to {csv_path}")
