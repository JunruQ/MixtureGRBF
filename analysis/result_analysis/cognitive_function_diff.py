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

subtype_stage = merge_info(subtype_stage, 'data/brain_mri_PRS_2025_9_29/time_field.csv')

subtype_stage['age_instance_2'] = subtype_stage['stage'] + (pd.to_datetime(subtype_stage['Instance 2']) - pd.to_datetime(subtype_stage['Instance 0'])).dt.days / 365.25

subtype_stage = merge_info(subtype_stage, 'data/brain_mri_PRS_2025_9_29/cf_field.csv')

cf_df = pd.read_csv('data/brain_mri_PRS_2025_9_29/cf_field.csv')

results = []

for row in cf_df.iterrows():
    cf_name, cf_value, cf_instance, cf_inversed = row[1]['Name'], row[1]['Value'], row[1]['Instance'], row[1]['IsInversed']
    age_col = 'stage' if cf_instance == 0 else 'age_instance_2'
    sub_cols = ['eid',  'subtype', age_col, 'sex', 'centre', cf_name]
    sub_df = subtype_stage[sub_cols].copy()

    # remove na
    sub_df.dropna(inplace=True)

    print(f"{cf_name}, {sub_df.shape[0]}")

    if cf_name == 'Prospective memory':
        sub_df[cf_name] = sub_df[cf_name].map({0:0, 2:1, 1:2})

    if cf_inversed:
        # minmax缩放
        data = sub_df[cf_name]
        sub_df[cf_name] = 1 - (data - data.min()) / (data.max() - data.min())
    else:
        data = sub_df[cf_name]
        sub_df[cf_name] = (data - data.min()) / (data.max() - data.min())
    

    REGRESSION_FLAG = True

    if REGRESSION_FLAG:
        continuous_var = [age_col, 'sex']

        categorical_var = ['centre']

        y = sub_df[[cf_name]]

        # Prepare covariates: one-hot encode categorical variables
        covariates = sub_df[continuous_var + categorical_var].copy()
        categorical_encoded = pd.get_dummies(covariates[categorical_var], drop_first=True) if len(categorical_var) > 0 else pd.DataFrame()
        covariates_encoded = pd.concat([covariates[continuous_var], categorical_encoded], axis=1)

        # Residualize each column in X
        # residualized_y = pd.DataFrame(index=sub_df.index, columns=y.columns, dtype=float)

        # common_idx = y.index.intersection(covariates_encoded.index)
        
        # y_aligned = y.loc[common_idx]
        # cov_aligned = covariates_encoded.loc[common_idx]
        
        # # Add constant for intercept
        # X_model = sm.add_constant(cov_aligned)
        
        # # Fit model
        # model = OLS(y_aligned, X_model).fit()
        # residuals = model.resid
        # residualized_y.loc[common_idx, cf_name] = residuals

        # sub_df[cf_name] = residualized_y

    subtypes = [i + 1 for i in range(nsubtype)]
    features = [cf_name]

    # for col in features:
    #     y = subtype_stage[col].astype(float)
    #     for sub in subtypes:
    #         mask1 = sub_df['subtype'] == sub
    #         mask2 = ~mask1
    #         group1 = sub_df.loc[mask1, col].dropna()
    #         group2 = sub_df.loc[mask2, col].dropna()
    #         stat, p = ttest_ind(group1, group2, equal_var=False)  # Welch's t-test
    #         mean_diff = group1.mean() - group2.mean()
    #         results.append({
    #             'feature': col, 
    #             'subtype': sub, 
    #             'p_value': p,
    #             'sign': np.sign(mean_diff)
    #         })

    for col in features:
        y = sub_df[col].astype(float)
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
                'feature': col,
                'subtype': sub,
                'p_value': p,
                'sign': sign
            })


result_df = pd.DataFrame(results)

# FDR correction
reject, p_corrected, _, _ = multipletests(result_df['p_value'], method='fdr_bh', alpha=0.05)
print(len(p_corrected))
log10fdr = -np.log10(p_corrected)
result_df['signed_log10FDR'] = log10fdr * result_df['sign']

result_df.to_csv(f'output/result_analysis/{exp_name}/{nsubtype}_subtypes/cf_difference.csv', index=False)