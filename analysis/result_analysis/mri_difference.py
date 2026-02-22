import pandas as pd
import utils.utils as utils
from statsmodels.api import OLS
import statsmodels.api as sm
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests
import numpy as np

exp_name = 'ukb_MixtureGRBF_cv_nsubtype_biom17'
nsubtype = 5

subtype_stage = utils.get_subtype_stage_with_cov(exp_name, nsubtype).rename(columns={'PTID': 'eid'})

subset_field = pd.read_csv('data/brain_mri_PRS_2025_9_29/UKB_Tabular_merged_10/UKB_FieldID_Subset.csv')

def merge_info(df, info_path):
    field = pd.read_csv(info_path)

    fields = field['Value'].tolist()
    subset = subset_field[subset_field['Field_ID'].isin(fields)]
    field_name_map = dict(zip(field['Value'], field['Name']))
    
    for subset_idx in subset['Subset_ID'].unique():
        subset_fields = subset[subset['Subset_ID'] == subset_idx]['Field_ID'].tolist()
        data = pd.read_csv(f'data/brain_mri_PRS_2025_9_29/UKB_Tabular_merged_10/UKB_subset_{subset_idx}.csv', usecols=['eid'] + subset_fields).rename(columns=field_name_map)
        df = pd.merge(df, data, on='eid', how='left')

    return df

subtype_stage = merge_info(subtype_stage, 'data/brain_mri_PRS_2025_9_29/icv_field.csv')
subtype_stage = merge_info(subtype_stage, 'data/brain_mri_PRS_2025_9_29/subcortical_field.csv')
subtype_stage = merge_info(subtype_stage, 'data/brain_mri_PRS_2025_9_29/cortical_field.csv')

subtype_stage = merge_info(subtype_stage, 'data/brain_mri_PRS_2025_9_29/time_field.csv')

# fa md diffenrence are all collected on instance 2
subtype_stage['stage'] = subtype_stage['stage'] + (pd.to_datetime(subtype_stage['Instance 2']) - pd.to_datetime(subtype_stage['Instance 0'])).dt.days / 365.25

subtype_stage.drop(columns=['Instance 0', 'Instance 1', 'Instance 2', 'Instance 3'], inplace=True)

# remove na
subtype_stage.dropna(inplace=True)

cortical_sort = pd.read_csv('data/brain_mri_PRS_2025_9_29/cortical_field.csv')['Name'].tolist()
subcortical_sort = pd.read_csv('data/brain_mri_PRS_2025_9_29/subcortical_field.csv')['Name'].tolist()

mean_df = subtype_stage[['subtype'] + cortical_sort + subcortical_sort].groupby('subtype').mean().T
std_df  = subtype_stage[['subtype'] + cortical_sort + subcortical_sort].groupby('subtype').std().T
mean_df.columns = [f'{c}_mean' for c in mean_df.columns]
std_df.columns  = [f'{c}_std'  for c in std_df.columns]
result = pd.concat([mean_df, std_df], axis=1)
result = result[sorted(result.columns, key=lambda x: (x.split('_')[0], x.split('_')[1]))]

result.to_csv(f'output/result_analysis/{exp_name}/{nsubtype}_subtypes/mri_summary.csv')

print(subtype_stage.shape[0])

continuous_var = ['stage', 'sex', 'ICV']

categorical_var = ['centre']

X_original = subtype_stage.iloc[:, 8:].copy()

# Prepare covariates: one-hot encode categorical variables
covariates = subtype_stage[continuous_var + categorical_var].copy()
categorical_encoded = pd.get_dummies(covariates[categorical_var], drop_first=True)
covariates_encoded = pd.concat([covariates[continuous_var], categorical_encoded], axis=1)

# # Residualize each column in X
# residualized_X = pd.DataFrame(index=subtype_stage.index, columns=X_original.columns, dtype=float)

# for col in X_original.columns:
#     y = X_original[col]
#     common_idx = y.index.intersection(covariates_encoded.index)
#     if len(common_idx) < len(covariates_encoded.columns) + 1:  # Rough check for degrees of freedom
#         print(f"Warning: Insufficient data for {col}, skipping.")
#         continue
    
#     y_aligned = y.loc[common_idx]
#     cov_aligned = covariates_encoded.loc[common_idx]
    
#     # Add constant for intercept
#     X_model = sm.add_constant(cov_aligned)
    
#     # Fit model
#     model = OLS(y_aligned, X_model).fit()
#     residuals = model.resid
#     residualized_X.loc[common_idx, col] = residuals

# # Replace original columns
# subtype_stage.iloc[:, 8:] = residualized_X

print(subtype_stage.shape[0])
print(subtype_stage["subtype"].value_counts())


# Assume 'subtype' is a column in subtype_stage
subtypes = range(1, nsubtype + 1)
features = subtype_stage.columns[8:]

# List to hold results
signed_log10fdr = []

results = []
# for col in features:
#     pvals = []
#     means_diff = []
#     sub_list = []
#     for sub in subtypes:
#         mask1 = subtype_stage['subtype'] == sub
#         mask2 = ~mask1
#         group1 = subtype_stage.loc[mask1, col].dropna()
#         group2 = subtype_stage.loc[mask2, col].dropna()
#         if len(group1) < 2 or len(group2) < 2:
#             pvals.append(1.0)
#             means_diff.append(0.0)
#             sub_list.append(sub)
#             continue
#         stat, p = ttest_ind(group1, group2, equal_var=False)  # Welch's t-test
#         mean_diff = group1.mean() - group2.mean()
#         results.append({
#                 'region': col, 
#                 'subtype': sub, 
#                 'p_value': p,
#                 'sign': np.sign(mean_diff)
#             })

for col in features:
    y = subtype_stage[col].astype(float)
    for sub in subtypes:
        subtype_indicator = (subtype_stage['subtype'] == sub).astype(int)
        X = pd.concat([subtype_indicator.rename('sub_ind'), covariates_encoded], axis=1)
        X = pd.concat([subtype_indicator.rename('sub_ind')], axis=1)
        X = sm.add_constant(X)

        # drop missing
        mask = y.notna() & X.notna().all(axis=1)
        y_clean = y[mask]
        X_clean = X[mask]

        # Fit model
        model = sm.OLS(y_clean, X_clean).fit()

        beta = model.params['sub_ind']
        p = model.pvalues['sub_ind']
        sign = np.sign(beta)

        results.append({
            'region': col,
            'subtype': sub,
            'p_value': p,
            'sign': sign
        })


result_df = pd.DataFrame(results)

print(f"Number of tests: {len(result_df['p_value'])}")

reject, p_corrected, _, _ = multipletests(result_df['p_value'], method='fdr_bh', alpha=0.05)
log10fdr = -np.log10(p_corrected)
result_df['signed_log10FDR'] = log10fdr * result_df['sign']

# 筛选出 signed_log10FDR < log10(0.05) 的行

# 按 subtype 统计个数
print('Significant count:')
print(result_df[result_df['signed_log10FDR'].abs() > -np.log10(0.05)].groupby('subtype').size())

result_df.to_csv(f'output/result_analysis/{exp_name}/{nsubtype}_subtypes/mri_difference.csv', index=False)