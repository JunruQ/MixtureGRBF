import pandas as pd
import utils.utils as utils
from statsmodels.api import OLS
import statsmodels.api as sm
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests
import numpy as np
import textwrap

exp_name = 'ukb_MixtureGRBF_cv_nsubtype_biom17'
nsubtype = 5

subtype_stage = utils.get_subtype_stage_with_cov(exp_name, nsubtype).rename(columns={'PTID': 'eid'})

subset_field = pd.read_csv('data/brain_mri_PRS_2025_9_29/UKB_Tabular_merged_10/UKB_FieldID_Subset.csv')

def merge_info(df, info_path):
    field = pd.read_csv(info_path)

    fields = field['Value'].tolist()
    subset = subset_field[subset_field['Field_ID'].isin(fields)]
    field_name_map = dict(zip(field['Value'], field['Name']))
    print(f"Missing fields: {[field_name_map[field] for field in fields if field not in subset['Field_ID'].tolist()]}")
    
    for subset_idx in subset['Subset_ID'].unique():
        subset_fields = subset[subset['Subset_ID'] == subset_idx]['Field_ID'].tolist()
        data = pd.read_csv(f'data/brain_mri_PRS_2025_9_29/UKB_Tabular_merged_10/UKB_subset_{subset_idx}.csv', usecols=['eid'] + subset_fields).rename(columns=field_name_map)
        df = pd.merge(df, data, on='eid', how='left')

    return df

subtype_stage = merge_info(subtype_stage, 'data/brain_mri_PRS_2025_9_29/prs_field.csv')

# remove na
subtype_stage.dropna(inplace=True)

REGRESSION_FLAG = True

if REGRESSION_FLAG:
    continuous_var = ['stage', 'sex']

    categorical_var = ['centre']

    X_original = subtype_stage.iloc[:, 7:].copy()

    # Prepare covariates: one-hot encode categorical variables
    covariates = subtype_stage[continuous_var + categorical_var].copy()
    categorical_encoded = pd.get_dummies(covariates[categorical_var], drop_first=True) if len(categorical_var) > 0 else pd.DataFrame()
    covariates_encoded = pd.concat([covariates[continuous_var], categorical_encoded], axis=1)

    # Residualize each column in X
    # residualized_X = pd.DataFrame(index=subtype_stage.index, columns=X_original.columns, dtype=float)

    # for col in X_original.columns:
    #     y = X_original[col]
    #     common_idx = y.index.intersection(covariates_encoded.index)
        
    #     y_aligned = y.loc[common_idx]
    #     cov_aligned = covariates_encoded.loc[common_idx]
        
    #     # Add constant for intercept
    #     X_model = sm.add_constant(cov_aligned)
        
    #     # Fit model
    #     model = OLS(y_aligned, X_model).fit()
    #     residuals = model.resid
    #     residualized_X.loc[common_idx, col] = residuals

    # # Replace original columns
    # subtype_stage.iloc[:, 7:] = residualized_X

# Assume 'subtype' is a column in subtype_stage
subtypes = [i + 1 for i in range(nsubtype)]
features = subtype_stage.columns[7:]

# results = []

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
#                 'feature': col, 
#                 'subtype': sub, 
#                 'p_value': p,
#                 'sign': np.sign(mean_diff)
#             })
    
results = []

# Prepare covariates again (continuous + one-hot categorical)
continuous_var = ['stage', 'sex']
categorical_var = ['centre']

covariates = subtype_stage[continuous_var + categorical_var].copy()
categorical_encoded = pd.get_dummies(covariates[categorical_var], drop_first=True) \
                       if len(categorical_var) > 0 else pd.DataFrame()
covariates_encoded = pd.concat([covariates[continuous_var], categorical_encoded], axis=1)

for col in features:
    y = subtype_stage[col].astype(float)

    for sub in subtypes:
        # one indicator: this subtype vs others
        subtype_indicator = (subtype_stage['subtype'] == sub).astype(int)

        # design matrix: indicator + covariates
        X = pd.concat([subtype_indicator.rename('sub_ind'), covariates_encoded], axis=1)
        X = sm.add_constant(X)

        # drop missing
        mask = y.notna() & X.notna().all(axis=1)
        y_clean = y[mask]
        X_clean = X[mask]

        if len(y_clean) < X_clean.shape[1] + 2:
            results.append({
                'feature': col,
                'subtype': sub,
                'p_value': 1.0,
                'sign': 0.0
            })
            continue

        # Fit model
        model = sm.OLS(y_clean, X_clean).fit()

        beta = model.params['sub_ind']
        p = model.pvalues['sub_ind']
        sign = np.sign(beta)

        results.append({
            'feature': col,
            'subtype': sub,
            'p_value': p,
            'sign': sign
        })

result_df = pd.DataFrame(results)


# FDR correction
reject, p_corrected, _, _ = multipletests(result_df['p_value'], method='fdr_bh', alpha=0.05)
log10fdr = -np.log10(p_corrected)
result_df['signed_log10FDR'] = log10fdr * result_df['sign']

result_df.to_csv(f'output/result_analysis/{exp_name}/{nsubtype}_subtypes/prs_difference.csv', index=False)


# import matplotlib.pyplot as plt

# # 定义显著性阈值
# threshold_star = np.log10(1 / 0.05)    # ≈1.3010, *
# threshold_dstar = np.log10(1 / 0.01)   # ≈2.0000, **
# threshold_tstar = np.log10(1 / 0.001)  # ≈3.0000, ***

# # Pivot the DataFrame to create a matrix with features as rows and subtypes as columns
# pivot_df = result_df.pivot(index='feature', columns='subtype', values='signed_log10FDR')

# plt.rcParams.update({
#     'font.family': 'Arial',  # 更清晰的字体
#     'axes.labelsize': 12,
#     'axes.titlesize': 12,
#     'xtick.labelsize': 10,
#     'ytick.labelsize': 10,
#     'legend.fontsize': 8,
#     # 'axes.spines.top': False,
#     # 'axes.spines.right': False,
#     # 'axes.spines.bottom': False,
#     # 'axes.spines.left': False,
# })

# # Create the heatmap with full data (no masking)
# fig, ax = plt.subplots(figsize=(10/2.54, 26.7/2.54))  # Adjust figsize based on number of features
# # im = ax.imshow(pivot_df.values, cmap='RdBu_r', aspect='auto', vmin=-threshold_tstar*1.2, vmax=threshold_tstar*1.2)  # 扩展范围以显示所有值
# # im = ax.imshow(pivot_df.values, cmap='RdBu_r', aspect='auto', vmin=-np.max(pivot_df), vmax=np.max(pivot_df))
# im = ax.imshow(pivot_df.values, cmap=utils.custom_rdbu_r, aspect='auto', vmin=-np.max(pivot_df), vmax=np.max(pivot_df))
# # Add significance annotations
# for i in range(len(pivot_df.index)):
#     for j in range(len(pivot_df.columns)):
#         val = np.abs(pivot_df.iloc[i, j])
#         if val > threshold_tstar:
#             symbol = '***'
#         elif val > threshold_dstar:
#             symbol = '**'
#         elif val > threshold_star:
#             symbol = '*'
#         else:
#             continue
#         ax.text(j, i+0.16, symbol, ha='center', va='center', color='white' if val > np.max(np.abs(pivot_df))/2 else 'black', fontsize=12, weight='bold', transform=ax.transData)

# # Add colorbar
# cbar = ax.figure.colorbar(im, ax=ax, shrink=0.5)
# cbar.set_label('Signed -log10(FDR)', rotation=270, labelpad=13)
# cbar.outline.set_visible(False)
# # cbar.ax.tick_params(length=0.2)


# # 取得当前位置 [x0, y0, width, height]
# pos = cbar.ax.get_position()

# # 手动调整高度 (缩短一半并向下平移一些)
# cbar.ax.set_position([pos.x0, pos.y0 + 0.1, pos.width, pos.height * 0.25])

# # Set labels
# ax.set_xticks(np.arange(len(pivot_df.columns)))
# ax.set_yticks(np.arange(len(pivot_df.index)))
# feature_labels = ['\n'.join(textwrap.wrap(i, width=25)) for i in pivot_df.index]
# ax.set_xticklabels(pivot_df.columns, ha='right')
# ax.set_yticklabels(feature_labels)
# ax.set_xlabel('Subtype')
# # ax.tick_params(axis='y', which='both', length=0)
# # ax.set_ylabel('Features')
# # ax.set_title('Polygenic risk scores difference by subtype')

# plt.tight_layout()
# output_path = f'output/result_analysis/{exp_name}/{nsubtype}_subtypes/prs_difference.png'
# plt.savefig(output_path, dpi=300)

# print(f"Figure saved to: {output_path}")