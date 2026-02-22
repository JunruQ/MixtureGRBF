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
exp_name = 'ukb_MixtureGRBF_cv_nsubtype_biom17'
INPUT_TABLE_PATH = 'input/ukb/ukb_covreg1_trans1_nanf1_biom0.csv'
SUBTYPE_STAGE_PATH = f'output/{exp_name}/{nsubtype}_subtypes/subtype_stage.csv'
OUTPUT_DIR = f'output/result_analysis/{exp_name}/{nsubtype}_subtypes'
SUBTYPE_ORDER_PATH = f'output/result_analysis/{exp_name}/{nsubtype}_subtypes/all_cause_mortality_order.csv'

df = pd.read_csv(INPUT_TABLE_PATH)
df.rename(columns={'RID':'PTID'}, inplace=True)
subtype_stage = pd.read_csv(SUBTYPE_STAGE_PATH)
try:
    subtype_order = pd.read_csv(SUBTYPE_ORDER_PATH, header=None)
except:
    subtype_order = pd.DataFrame(np.array(list(range(1, nsubtype+1))))
subtype_order_map = {int(subtype_order.iloc[i]): i+1 for i in range(nsubtype)}
subtype_stage['subtype'] = subtype_stage['subtype'].map(subtype_order_map)
biomarker_names = df.iloc[:, 7:].columns.tolist()
df = pd.merge(df, subtype_stage, how='inner', on=['PTID', 'stage'])

alpha = 0.05

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Create list to store all data
results = []

for i in range(1, nsubtype + 1):
    # Create binary indicator for the subtype
    df['is_subtype'] = (df['subtype'] == i).astype(int)
    
    for biom in biomarker_names:
        # Check for missing values in the biomarker, stage, and sex columns
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
                'biom': biom,
                'subtype': i,
                't_statistic': t_value,
                'p_value': p_value
            })
        except Exception as e:
            print(f"Error fitting model for {biom}, subtype {i}: {e}")
            continue

# Create DataFrame
t_stats_df = pd.DataFrame(results)

# Apply Bonferroni correction
t_stats_df['corrected_p_value'] = bonferroni_correction(t_stats_df['p_value'])
t_stats_df['significant'] = t_stats_df['corrected_p_value'] < alpha

# Sort by biom and subtype
t_stats_df = t_stats_df.sort_values(['biom', 'subtype'])

# Save to CSV file
output_csv_path = f'{OUTPUT_DIR}/t_stats_by_subtype.csv'
t_stats_df.to_csv(output_csv_path, index=False)

# Print confirmation
print(f"t-statistics (adjusted for stage and sex) saved to: {output_csv_path}")
print(f"Total rows: {len(t_stats_df)}")
print("First few rows of the data:")
print(t_stats_df.head())