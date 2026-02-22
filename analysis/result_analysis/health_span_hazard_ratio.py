import json
import pandas as pd
import os
from lifelines import CoxPHFitter, KaplanMeierFitter
import matplotlib.pyplot as plt
import numpy as np
import warnings
import textwrap

def parse_target_field(df: pd.DataFrame) -> pd.DataFrame:
    df_result = pd.DataFrame()
    df_result['eid'] = df['eid']
    if 'Field' in df.columns:
        df_result['field'] = df['Field'].apply(lambda x: x.split(' ')[1] if not pd.isna(x) else x)
    elif 'target_cancer' in df.columns:
        df_result['field'] = df['target_cancer']
    elif 'target_death' in df.columns:
        df_result['field'] = df['target_death']
    else:
        raise ValueError('Field not found in dataframe')
    df_result['bl2t'] = df['BL2Target_yrs']
    return df_result

def calculate_hazard_ratios(df_s, disease_name, disease_code, output_dir):
    # Validate data
    if df_s['bl2t'].isnull().any() or df_s['stage'].isnull().any():
        warnings.warn(f"Missing values in 'bl2t' or 'stage' for {disease_name}, skipping HR calculation.")
        return None
    # if (df_s['bl2t'] + df_s['stage'] <= 0).any():
    #     warnings.warn(f"Non-positive survival times for {disease_name}, skipping HR calculation.")
    #     return None
    if df_s['subtype'].isnull().any():
        warnings.warn(f"Missing subtype values for {disease_name}, skipping HR calculation.")
        return None

    # Check event counts by subtype before one-hot encoding
    event_counts = df_s.groupby('subtype')['field'].apply(lambda x: x.apply(lambda y: y in disease_code if not pd.isna(y) else False).sum())
    if event_counts.eq(0).any():
        warnings.warn(f"No events for some subtypes in {disease_name}, skipping HR calculation.")
        return None

    cph = CoxPHFitter(penalizer=0.1)  # Add L2 regularization to handle collinearity
    # Prepare data for Cox model: one-hot encode subtypes, drop first to avoid collinearity
    df_cox = pd.get_dummies(df_s, columns=['subtype'], prefix='subtype', drop_first=True)
    subtype_cols = [col for col in df_cox.columns if col.startswith('subtype_')]
    
    # Event is True if the disease code is in the field, False otherwise
    df_cox['event'] = df_s['field'].apply(lambda x: x in disease_code if not pd.isna(x) else False)
    df_cox['time'] = df_s['bl2t'] + df_s['stage']
    
    # Fit Cox model with exception handling
    try:
        cph.fit(df_cox[['time'] + subtype_cols + ['event']], duration_col='time', event_col='event')
        
        # Extract hazard ratios and confidence intervals
        hr_summary = cph.summary[['exp(coef)', 'exp(coef) lower 95%', 'exp(coef) upper 95%']]
        hr_summary = hr_summary.rename(columns={'exp(coef)': 'HR', 'exp(coef) lower 95%': 'Lower_CI', 'exp(coef) upper 95%': 'Upper_CI'})
        hr_summary['Disease'] = disease_name
        hr_summary['Subtype'] = [col.replace('subtype_', '') for col in hr_summary.index]
        return hr_summary
    except Exception as e:
        warnings.warn(f"Failed to fit Cox model for {disease_name}: {str(e)}")
        return None

def calculate_lifespan(df_s, disease_name, disease_code, output_dir):
    # Validate data
    if df_s['bl2t'].isnull().any() or df_s['stage'].isnull().any():
        warnings.warn(f"Missing values in 'bl2t' or 'stage' for {disease_name}, skipping lifespan calculation.")
        return None
    if df_s['subtype'].isnull().any():
        warnings.warn(f"Missing subtype values for {disease_name}, skipping lifespan calculation.")
        return None

    # Event is True if the disease code is in the field, False otherwise
    df_s['event'] = df_s['field'].apply(lambda x: x in disease_code if not pd.isna(x) else False)
    df_s['time'] = df_s['bl2t'] + df_s['stage']

    # Fit Kaplan-Meier estimator for each subtype
    kmf = KaplanMeierFitter()

    lifespan_results = []

    # Group by subtype and fit Kaplan-Meier estimator
    for subtype in df_s['subtype'].unique():
        df_subtype = df_s[df_s['subtype'] == subtype]
        
        # Fit Kaplan-Meier estimator
        kmf.fit(df_subtype['time'], event_observed=df_subtype['event'])
        
        # Calculate expected lifespan (mean survival time)
        expected_lifespan = kmf.median_survival_time_  # This gives the median survival time
        lifespan_results.append({'Subtype': subtype, 'Lifespan': expected_lifespan})
    
    lifespan_df = pd.DataFrame(lifespan_results)
    lifespan_df['Disease'] = disease_name
    
    return lifespan_df

def calculate_healthspan_hazard_ratio(subtype_stage, important_disease, output_dir):
    cph = CoxPHFitter(penalizer=0.1)  # Add L2 regularization

    df_healthspan = subtype_stage.copy()
    df_healthspan['time'] = df_healthspan['stage']
    df_healthspan['event'] = False
    
    # Validate stage
    if df_healthspan['stage'].isnull().any() or (df_healthspan['stage'] <= 0).any():
        warnings.warn("Invalid or missing stage values in healthspan calculation, skipping.")
        return None
    if df_healthspan['subtype'].isnull().any():
        warnings.warn("Missing subtype values in healthspan calculation, skipping.")
        return None
    
    # For each disease, find the earliest event time
    earliest_times = []
    for disease_name, disease_code in important_disease.items():
        disease_upper_level_code = disease_code[0][0]
        try:
            disease_info = pd.read_csv(f'./input/disease_info/{disease_upper_level_code}0.csv')
            df_s = pd.merge(subtype_stage, parse_target_field(disease_info), left_on='PTID', right_on='eid', how='left')
            df_s['event'] = df_s['field'].apply(lambda x: x in disease_code if not pd.isna(x) else False)
            df_s['event_time'] = df_s['bl2t'] + df_s['stage']
            earliest_times.append(df_s[['PTID', 'event', 'event_time']])
        except FileNotFoundError:
            warnings.warn(f"Disease info file for {disease_name} not found, skipping in healthspan calculation.")
            continue
    
    if not earliest_times:
        warnings.warn("No valid disease data for healthspan calculation, skipping.")
        return None
    
    # Combine all disease events, take the earliest event time per participant
    df_events = pd.concat(earliest_times)
    df_events = df_events[df_events['event'] == True].groupby('PTID').agg({'event_time': 'min', 'event': 'max'}).reset_index()
    
    # Merge with subtype_stage to include all participants
    df_healthspan = pd.merge(subtype_stage, df_events, on='PTID', how='left')
    df_healthspan['event'] = df_healthspan['event'].fillna(False)
    df_healthspan['time'] = df_healthspan.apply(
        lambda row: row['event_time'] if row['event'] else row['stage'], axis=1
    )
    
    # Check event counts
    event_counts = df_healthspan.groupby('subtype')['event'].sum()
    if event_counts.eq(0).any():
        warnings.warn("No events for some subtypes in healthspan, skipping HR calculation.")
        return None
    
    # One-hot encode subtypes, drop first to avoid collinearity
    df_cox = pd.get_dummies(df_healthspan, columns=['subtype'], prefix='subtype', drop_first=True)
    subtype_cols = [col for col in df_cox.columns if col.startswith('subtype_')]
    
    # Fit Cox model
    try:
        cph.fit(df_cox[['time'] + subtype_cols + ['event']], duration_col='time', event_col='event')
        
        # Extract hazard ratios and confidence intervals
        hr_summary = cph.summary[['exp(coef)', 'exp(coef) lower 95%', 'exp(coef) upper 95%']]
        hr_summary = hr_summary.rename(columns={'exp(coef)': 'HR', 'exp(coef) lower 95%': 'Lower_CI', 'exp(coef) upper 95%': 'Upper_CI'})
        hr_summary['Disease'] = 'Healthspan'
        hr_summary['Subtype'] = [col.replace('subtype_', '') for col in hr_summary.index]
        return hr_summary
    except Exception as e:
        warnings.warn(f"Failed to fit Cox model for healthspan: {str(e)}")
        return None

def plot_forest_plot(hr_dfs, output_dir):
    # Filter out None results
    hr_dfs = [hr for hr in hr_dfs if hr is not None]
    if not hr_dfs:
        warnings.warn("No valid hazard ratios to plot.")
        return
    
    # Combine all hazard ratio dataframes
    all_hr = pd.concat(hr_dfs)
    
    # Get unique diseases and assign colors
    diseases = list(all_hr['Disease'].unique().astype(str))  # Convert to list of strings
    colors = plt.cm.Set2(np.linspace(0, 1, len(diseases)))
    disease_colors = dict(zip(diseases, colors))
    
    # Create figure
    plt.figure(figsize=(10, len(all_hr) * 0.4 + 1))
    
    # Group by disease
    grouped = all_hr.groupby('Disease')
    y_pos = len(all_hr) - 1
    y_ticks = []
    y_labels = []
    legend_handles = {}
    
    for disease, group in grouped:
        for _, row in group.iterrows():
            # Plot errorbar
            line = plt.errorbar(
                row['HR'], 
                y_pos, 
                xerr=[[row['HR'] - row['Lower_CI']], [row['Upper_CI'] - row['HR']]],
                fmt='o', 
                capsize=3,
                color=disease_colors[str(disease)],  # Ensure disease is string
                label=str(disease) if str(disease) not in legend_handles else ""
            )
            # Store legend handle for each disease
            if str(disease) not in legend_handles:
                legend_handles[str(disease)] = line
            # Store position and label for y-ticks
            y_ticks.append(y_pos)
            y_labels.append(str(row['Subtype']))
            y_pos -= 1
        # Add spacing between disease groups
        y_pos -= 0.5
    
    # Customize plot
    plt.axvline(x=1, color='gray', linestyle='--')
    plt.xlabel('Hazard Ratio (95% CI)')
    plt.ylabel('Subtype')
    plt.xscale('log')
    
    # Set y-ticks and labels
    plt.yticks(y_ticks, y_labels)
    
    # Customize grid: only show horizontal grid lines at y-tick positions
    plt.grid(True, which="both", ls="--", axis='x')
    plt.grid(True, which="major", ls="--", axis='y', alpha=0.5)
    
    plt.title('Hazard Ratios by Disease and Subtype')
    
    # Wrap legend labels to a maximum width (e.g., 20 characters)
    wrapped_labels = ['\n'.join(textwrap.wrap(disease, width=20)) for disease in diseases]
    
    # Add legend with wrapped labels
    plt.legend(handles=[legend_handles[disease] for disease in diseases], 
               labels=wrapped_labels, 
               bbox_to_anchor=(1.05, 1), 
               loc='upper left')
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(f'{output_dir}/hazard_ratios_forest_plot.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_forest_plot_lifespan(lifespan_dfs, output_dir):
    # Filter out None results
    lifespan_dfs = [lf for lf in lifespan_dfs if lf is not None]
    if not lifespan_dfs:
        warnings.warn("No valid lifespan data to plot.")
        return
    
    # Combine all lifespan dataframes
    all_lifespan = pd.concat(lifespan_dfs)
    
    # Create figure
    plt.figure(figsize=(10, len(all_lifespan) * 0.4 + 1))
    
    # Group by disease
    grouped = all_lifespan.groupby('Disease')
    y_pos = len(all_lifespan) - 1
    y_ticks = []
    y_labels = []
    legend_handles = {}
    
    for disease, group in grouped:
        # Find subtype1 (assumed to be the first subtype in the list)
        subtype1 = group['Subtype'].min()  # assuming subtype1 is the lowest subtype number
        
        for _, row in group.iterrows():
            # Plot errorbar for comparing to subtype1
            line = plt.errorbar(
                row['Lifespan'], 
                y_pos, 
                xerr=[[row['Lifespan'] - row['Lifespan']], [row['Lifespan'] - row['Lifespan']]],  # Can replace with CI if needed
                fmt='o', 
                capsize=3,
                color='blue',  # You can use different colors here
                label=f'{disease} Subtype{row["Subtype"]}' if f'{disease} Subtype{row["Subtype"]}' not in legend_handles else ""
            )
            # Store legend handle for each disease
            if f'{disease} Subtype{row["Subtype"]}' not in legend_handles:
                legend_handles[f'{disease} Subtype{row["Subtype"]}'] = line
            # Store position and label for y-ticks
            y_ticks.append(y_pos)
            y_labels.append(f'Subtype{row["Subtype"]}')
            y_pos -= 1
        # Add spacing between disease groups
        y_pos -= 0.5
    
    # Customize plot
    plt.axvline(x=1, color='gray', linestyle='--')
    plt.xlabel('Expected Lifespan')
    plt.ylabel('Subtype')
    
    # Set y-ticks and labels
    plt.yticks(y_ticks, y_labels)
    
    plt.title('Expected Lifespan by Disease and Subtype')
    
    # Add legend
    plt.legend(handles=[legend_handles[disease] for disease in legend_handles], loc='upper left')
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(f'{output_dir}/lifespan_forest_plot.png', dpi=300, bbox_inches='tight')
    plt.close()


# IO
with open('preprocess/data/important_disease_healthspan.json', 'r') as f:
    important_disease = json.load(f)

nsubtype = 5
result_folder = 'ukb_MixtureGRBF_cv_nsubtype_biom17'
subtype_stage = pd.read_csv(f'./output/{result_folder}/{nsubtype}_subtypes/subtype_stage.csv')
output_dir = f'./output/result_analysis/{result_folder}/{nsubtype}_subtypes'
SUBTYPE_ORDER_PATH = f'output/result_analysis/{result_folder}/{nsubtype}_subtypes/all_cause_mortality_order.csv'

try:
    subtype_order = pd.read_csv(SUBTYPE_ORDER_PATH, header=None)
except:
    subtype_order = pd.DataFrame(np.array(list(range(1, nsubtype+1))))

# Create subtype mapping, exclude invalid subtypes
subtype_mapping = {subtype: i + 1 for i, subtype in enumerate(subtype_order.iloc[:, 0].tolist())}
subtype_stage['subtype'] = subtype_stage['subtype'].map(subtype_mapping).fillna(np.nan)
if subtype_stage['subtype'].isnull().any():
    warnings.warn("Invalid subtype values found, removing affected rows.")
    subtype_stage = subtype_stage.dropna(subset=['subtype'])
subtype_stage['subtype'] = subtype_stage['subtype'].astype(int)

os.makedirs(output_dir, exist_ok=True)

# Calculate hazard ratios for each disease
# hr_dfs = []
# for disease_name, disease_code in important_disease.items():
#     disease_upper_level_code = disease_code[0][0]
#     try:
#         disease_info = pd.read_csv(f'./input/disease_info/{disease_upper_level_code}0.csv')
#         df_s = pd.merge(subtype_stage, parse_target_field(disease_info), left_on='PTID', right_on='eid', how='left')
        
#         # Calculate hazard ratios
#         hr_df = calculate_hazard_ratios(df_s, disease_name, disease_code, output_dir)
#         if hr_df is not None:
#             hr_dfs.append(hr_df)
#     except FileNotFoundError:
#         warnings.warn(f"Disease info file for {disease_name} not found, skipping.")
#         continue

# # Calculate healthspan hazard ratio
# hr_healthspan = calculate_healthspan_hazard_ratio(subtype_stage, important_disease, output_dir)
# if hr_healthspan is not None:
#     hr_dfs.append(hr_healthspan)

# # Save hazard ratios to CSV
# if hr_dfs:
#     all_hr = pd.concat(hr_dfs)
#     all_hr.to_csv(f'{output_dir}/hazard_ratios.csv', index=False)
# else:
#     warnings.warn("No hazard ratios calculated, CSV not saved.")

# # Plot and save forest plot
# plot_forest_plot(hr_dfs, output_dir)

# Calculate lifespan for each disease
lifespan_dfs = []
for disease_name, disease_code in important_disease.items():
    disease_upper_level_code = disease_code[0][0]
    try:
        disease_info = pd.read_csv(f'./input/disease_info/{disease_upper_level_code}0.csv')
        df_s = pd.merge(subtype_stage, parse_target_field(disease_info), left_on='PTID', right_on='eid', how='left')
        
        # Calculate lifespan
        lifespan_df = calculate_lifespan(df_s, disease_name, disease_code, output_dir)
        if lifespan_df is not None:
            lifespan_dfs.append(lifespan_df)
    except FileNotFoundError:
        warnings.warn(f"Disease info file for {disease_name} not found, skipping.")
        continue

# Save lifespan results to CSV
if lifespan_dfs:
    all_lifespan = pd.concat(lifespan_dfs)
    all_lifespan.to_csv(f'{output_dir}/lifespan.csv', index=False)
else:
    warnings.warn("No lifespan data calculated, CSV not saved.")

# Plot and save forest plot for lifespan
plot_forest_plot_lifespan(lifespan_dfs, output_dir)
