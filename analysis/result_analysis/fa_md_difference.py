import pandas as pd
import utils.utils as utils
from statsmodels.api import OLS
import statsmodels.api as sm
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests
import numpy as np
import matplotlib.pyplot as plt

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

subtype_stage = merge_info(subtype_stage, 'data/brain_mri_PRS_2025_9_29/time_field.csv')

# fa md diffenrence are all collected on instance 2
subtype_stage['stage'] = subtype_stage['stage'] + (pd.to_datetime(subtype_stage['Instance 2']) - pd.to_datetime(subtype_stage['Instance 0'])).dt.days / 365.25

subtype_stage.drop(columns=['Instance 0', 'Instance 1', 'Instance 2', 'Instance 3'], inplace=True)

subtype_stage_template = subtype_stage.copy()

results = []
for idx, index in enumerate(['fa', 'md']):
    subtype_stage = merge_info(subtype_stage_template, f'data/brain_mri_PRS_2025_9_29/{index}_field.csv')
    index_sort = pd.read_csv(f'data/brain_mri_PRS_2025_9_29/{index}_field.csv')['Name'].tolist()
    mean_df = subtype_stage[['subtype'] + index_sort].groupby('subtype').mean().T
    std_df  = subtype_stage[['subtype'] + index_sort].groupby('subtype').std().T
    mean_df.columns = [f'{c}_mean' for c in mean_df.columns]
    std_df.columns  = [f'{c}_std'  for c in std_df.columns]
    result = pd.concat([mean_df, std_df], axis=1)
    result = result[sorted(result.columns, key=lambda x: (x.split('_')[0], x.split('_')[1]))]
    result.to_csv(f'output/result_analysis/{exp_name}/{nsubtype}_subtypes/{index}_summary.csv')
    # remove na
    subtype_stage.dropna(inplace=True)
    print(f"Number of samples for {index}: {len(subtype_stage)}")

    REGRESSION_FLAG = True

    if REGRESSION_FLAG:
        continuous_var = ['stage', 'sex']

        categorical_var = ['centre']

        X_original = subtype_stage.iloc[:, 7:].copy()

        # Prepare covariates: one-hot encode categorical variables
        covariates = subtype_stage[continuous_var + categorical_var].copy()
        categorical_encoded = pd.get_dummies(covariates[categorical_var], drop_first=True) if len(categorical_var) > 0 else pd.DataFrame()
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
        # subtype_stage.iloc[:, 7:] = residualized_X

    # Assume 'subtype' is a column in subtype_stage
    subtypes = [i + 1 for i in range(nsubtype)]
    features = subtype_stage.columns[7:]

    # List to hold results
    signed_log10fdr = []

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
    #         pvals.append(p)
    #         mean_diff = group1.mean() - group2.mean()
    #         results.append({
    #             'index_name': index,
    #             'feature': col, 
    #             'subtype': sub, 
    #             'p_value': p,
    #             'sign': np.sign(mean_diff)
    #         })

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
                'index_name': index,
                'feature': col,
                'subtype': sub,
                'p_value': p,
                'sign': sign
            })

result_df = pd.DataFrame(results)

for index in ['fa', 'md']:
    reject, p_corrected, _, _ = multipletests(result_df.loc[result_df['index_name'] == index, 'p_value'], method='fdr_bh', alpha=0.05)
    print(f"{index} p_corrected: {len(p_corrected)}")
    log10fdr = -np.log10(p_corrected)
    result_df.loc[result_df['index_name'] == index, 'signed_log10FDR'] = log10fdr * result_df.loc[result_df['index_name'] == index, 'sign']

result_df.to_csv(f'output/result_analysis/{exp_name}/{nsubtype}_subtypes/fa_md_difference.csv', index=False)



# import matplotlib.gridspec as gridspec

# fig = plt.figure(figsize=(8/2.54, 20/2.54))
# # gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 0.05], wspace=0.1)
# gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1], wspace=0.1)

# axes = [fig.add_subplot(gs[0]), fig.add_subplot(gs[1])]
# # cbar_ax = fig.add_subplot(gs[2])


# plt.rcParams.update({
#     'font.family': 'sans-serif',  # 指定字体家族为无衬线
#     'font.sans-serif': ['Arial'], # 在无衬线字体列表中首选 Arial
#     'axes.labelsize': 12,
#     'axes.titlesize': 12,
#     'xtick.labelsize': 10,
#     'ytick.labelsize': 10,
#     'legend.fontsize': 8
# })

# for idx, index in enumerate(['fa', 'md']):
#     result_df = results[idx]
#     threshold_star = np.log10(1 / 0.05)
#     threshold_dstar = np.log10(1 / 0.01)
#     threshold_tstar = np.log10(1 / 0.001)

#     pivot_df = result_df.pivot(index='feature', columns='subtype', values='signed_log10FDR')

#     ax = axes[idx]
#     im = ax.imshow(pivot_df.values, cmap=utils.custom_rdbu_r, aspect='auto', vmin=-vmax, vmax=vmax)
    
#     # Add significance annotations
#     for i in range(len(pivot_df.index)):
#         for j in range(len(pivot_df.columns)):
#             val = np.abs(pivot_df.iloc[i, j])
#             if val > threshold_tstar:
#                 symbol = '***'
#             elif val > threshold_dstar:
#                 symbol = '**'
#             elif val > threshold_star:
#                 symbol = '*'
#             else:
#                 continue
#             ax.text(j, i+0.16, symbol, ha='center', va='center', 
#                    color='white' if val > np.max(pivot_df)/2 else 'black', 
#                    fontsize=12, weight='bold')

#     # Add colorbar using divider
#     # if idx == 1:
#     #     divider = make_axes_locatable(ax)
#     #     cax = divider.append_axes("right", size="5%", pad=0.1)
#     #     cbar = fig.colorbar(im, cax=cax)
#     #     cbar.set_label('Signed -log10(FDR)', rotation=270, labelpad=13)

#     # Set labels
#     ax.set_xticks(np.arange(len(pivot_df.columns)))
#     ax.set_yticks(np.arange(len(pivot_df.index)))
#     ax.set_xticklabels(pivot_df.columns, ha='right')
#     if idx == 0:
#         ax.set_yticklabels(pivot_df.index)
#     else:
#         ax.set_yticklabels([])
        
#     ax.set_xlabel('Subtype')

# # Add colorbar to the dedicated axis
# # cbar = fig.colorbar(im, cax=cbar_ax, shrink=0.5)
# # cbar.set_label('Signed -log10(FDR)', rotation=270, labelpad=13)

# output_path = f'output/result_analysis/{exp_name}/{nsubtype}_subtypes/fa_md_difference.png'
# plt.savefig(output_path, dpi=300, bbox_inches='tight')
# print(f'Figure saved to {output_path}')
# plt.close()
