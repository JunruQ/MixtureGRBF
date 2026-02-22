import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import os

def bonferroni_correction(p_values):
    n = len(p_values)
    corrected_p_values = np.array(p_values) * n
    return np.minimum(corrected_p_values, 1.0)  # Cap at 1.0 as p-values can't exceed 1

nsubtype = 5

df_input = pd.read_csv('input/ukb/ukb_covreg1_trans1_nanf1_biom0.csv')
df_input.rename(columns={'RID':'PTID'}, inplace=True)
biomarker_names = df_input.iloc[:, 7:].columns.tolist()

INPUT_TABLE_PATH = 'input/ukb/ukb_covreg1_trans1_nanf1_biom0.csv'
OUTPUT_DIR = f'output/result_analysis/ukb_MixtureGRBF_cv_nsubtype_biom17/{nsubtype}_subtypes/site_val'
site_map_path = 'input/ukb/ukb_site_map.csv'

site_map = pd.read_csv(site_map_path, header=None, index_col=1)

for site in ['Northern England', 'Southern England', 'Midlands', 'Scotland', 'Wales']:
    # for set_type in ['train', 'val']:
    for set_type in ['val']:
        subtype_stage_path = f'output/ukb_MixtureGRBF_site_validation/5_subtypes/{site}/subtype_stage.csv'
        subtype_order = pd.read_csv(f'output/result_analysis/ukb_MixtureGRBF_cv_nsubtype_biom17/5_subtypes/site_val/{site}_order.csv', header=None)
        subtype_stage = pd.read_csv(subtype_stage_path)
        subtype_order_map = {int(subtype_order.iloc[i]): i+1 for i in range(nsubtype)}
        subtype_stage['subtype'] = subtype_stage['subtype'].map(subtype_order_map)

        df = df_input.copy()
        df = pd.merge(df, subtype_stage, how='inner', on=['PTID', 'stage'])
        if set_type == 'train':
            df = df[~df['centre'].isin(site_map.loc[site, 0].tolist())]
        else:
            df = df[df['centre'].isin(site_map.loc[site, 0].tolist())]

        alpha = 0.05

        # Create output directory if it doesn't exist
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        # Create list to store all data
        results = []

        for i in range(1, nsubtype + 1):
            # Create binary indicator for the subtype
            df['is_subtype'] = (df['subtype'] == i).astype(int)
            
            for biom in biomarker_names:
                # Check for missing values in the biomarker, stage, columns
                current_df = df[[biom, 'is_subtype', 'stage']].dropna()
                if len(current_df) == 0:
                    print(f"Warning: No valid data for {biom} after dropping NA, skipping...")
                    continue
                
                # Define X (predictors) and y (response)
                X = current_df[['is_subtype', 'stage']]
                y = current_df[biom]
                
                # Add constant term for intercept
                X = sm.add_constant(X)
                
                # Fit linear regression model
                try:
                    model = sm.OLS(y, X).fit()
                    
                    # Extract t-statistic and p-value for the subtype indicator
                    t_value = model.tvalues['is_subtype']
                    p_value = model.pvalues['is_subtype']
                    
                    # Add to results list
                    results.append({
                        'Biomarker': biom,
                        'Subtype': i,
                        't': t_value,
                        'p': p_value
                    })
                except Exception as e:
                    print(f"Error fitting model for {biom}, subtype {i}: {e}")
                    continue

        # Create DataFrame
        t_stats_df = pd.DataFrame(results)

        # Apply Bonferroni correction
        t_stats_df['corrected_p_value'] = bonferroni_correction(t_stats_df['p'])
        t_stats_df['significant'] = t_stats_df['corrected_p_value'] < alpha

        # Sort by biom and subtype
        t_stats_df = t_stats_df.sort_values(['Biomarker', 'Subtype'])

        # Save to CSV file
        output_csv_path = f'{OUTPUT_DIR}/t_stats_{site}_{set_type}.csv'
        t_stats_df.to_csv(output_csv_path, index=False)

        # Print confirmation
        print(f"t-statistics saved to: {output_csv_path}")
        print(f"Total rows: {len(t_stats_df)}")
        print("First few rows of the data:")
        print(t_stats_df.head())