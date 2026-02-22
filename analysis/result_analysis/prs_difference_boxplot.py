import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import utils.utils as utils

# ==========================================
# 1. Configuration & Styles
# ==========================================
nsubtype = 5
exp_name = 'ukb_MixtureGRBF_cv_nsubtype_biom17'
OUTPUT_DIR = f'output/result_analysis/{exp_name}/{nsubtype}_subtypes'

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# --- Visual Style Constants (Matched to your reference) ---
MAIN_COLOR = '#0072BD'
BOX_PROPS = {'facecolor': 'none', 'edgecolor': '#333333', 'linewidth': 1.2}
WHISKER_PROPS = {'color': '#333333', 'linewidth': 1.2}
CAP_PROPS = {'color': '#333333', 'linewidth': 1.2}
MEDIAN_PROPS = {'color': 'red', 'linewidth': 1.5}
POINTS_SIZE = 2
POINTS_ALPHA = 0.4
MAX_POINTS_DISPLAY = 500  # Downsample for speed if N is huge

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 2. Data Loading Helpers
# ==========================================
# Load the field IDs for PRS
subset_field = pd.read_csv('data/brain_mri_PRS_2025_9_29/UKB_Tabular_merged_10/UKB_FieldID_Subset.csv')

def merge_info(df, info_path):
    """
    Loads PRS fields, renames them using the name map, and merges into the main dataframe.
    """
    field = pd.read_csv(info_path)
    fields = field['Value'].tolist()
    
    # Create mapping: Field ID -> Nice Name
    field_name_map = dict(zip(field['Value'], field['Name']))
    
    # Check for missing fields (optional print)
    subset = subset_field[subset_field['Field_ID'].isin(fields)]
    # missing = [field_name_map[f] for f in fields if f not in subset['Field_ID'].tolist()]
    # if missing: print(f"Missing fields: {missing}")
    
    # Merge data
    for subset_idx in subset['Subset_ID'].unique():
        subset_fields = subset[subset['Subset_ID'] == subset_idx]['Field_ID'].tolist()
        
        # Load specific subset file
        subset_path = f'data/brain_mri_PRS_2025_9_29/UKB_Tabular_merged_10/UKB_subset_{subset_idx}.csv'
        if os.path.exists(subset_path):
            data = pd.read_csv(subset_path, usecols=['eid'] + subset_fields)
            data = data.rename(columns=field_name_map)
            df = pd.merge(df, data, on='eid', how='left')
            
    return df

def wrap_label(label: str, threshold: int = 20) -> str:
    """Wraps long titles for better plot display."""
    if not label or len(str(label)) <= threshold:
        return str(label)
    s = str(label)
    middle = len(s) // 2
    # Find nearest space to middle
    spaces = [i for i, c in enumerate(s) if c == ' ']
    if not spaces:
        return s[:middle] + '\n' + s[middle:]
    
    best_space = min(spaces, key=lambda x: abs(x - middle))
    return s[:best_space] + '\n' + s[best_space+1:]

# ==========================================
# 3. Load Data
# ==========================================
print("Loading PRS data...")

# 1. Get Base Subtype Data
subtype_stage = utils.get_subtype_stage_with_cov(exp_name, nsubtype).rename(columns={'PTID': 'eid'})

# 2. Merge PRS Features
subtype_stage = merge_info(subtype_stage, 'data/brain_mri_PRS_2025_9_29/prs_field.csv')

# 3. Clean Data
subtype_stage.dropna(inplace=True)

# 4. Define Features to Plot
# Assuming the first 7 columns are ID/Subtype info, and the rest are PRS features
prs_features = subtype_stage.columns[7:] 
print(f"Found {len(prs_features)} PRS features to plot.")

# ==========================================
# 4. Plotting Logic
# ==========================================
# Calculate Grid Size
N_COLS = 6
n_rows = int(np.ceil(len(prs_features) / N_COLS))
FIG_WIDTH = 33/2.54
FIG_HEIGHT = 23/2.54

fig, axes = plt.subplots(n_rows, N_COLS, figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=150)
axes = axes.flatten()

for i, feature_name in enumerate(prs_features):
    ax = axes[i]
    
    # Extract data for this feature
    plot_data = subtype_stage[['subtype', feature_name]].dropna()
    
    if plot_data.empty:
        ax.text(0.5, 0.5, 'No Data', ha='center', va='center')
        continue

    # --- 1. Stripplot (Background Scatter) ---
    # Sampling for performance if dataset is large
    unique_subtypes = sorted(plot_data['subtype'].unique())
    sampled_dfs = []
    for st in unique_subtypes:
        sub_df = plot_data[plot_data['subtype'] == st]
        if len(sub_df) > MAX_POINTS_DISPLAY:
            sub_df = sub_df.sample(n=MAX_POINTS_DISPLAY, random_state=42)
        sampled_dfs.append(sub_df)
    
    if sampled_dfs:
        scatter_data = pd.concat(sampled_dfs)
        sns.stripplot(
            data=scatter_data, x='subtype', y=feature_name, 
            color=MAIN_COLOR,
            size=POINTS_SIZE, 
            alpha=POINTS_ALPHA,
            jitter=0.25, 
            zorder=0,
            ax=ax
        )
    
    # --- 2. Boxplot (Foreground Distribution) ---
    sns.boxplot(
        data=plot_data, x='subtype', y=feature_name,
        showfliers=False,
        width=0.5,
        boxprops=BOX_PROPS, 
        whiskerprops=WHISKER_PROPS,
        capprops=CAP_PROPS, 
        medianprops=MEDIAN_PROPS,
        zorder=10,
        ax=ax
    )

    # --- 3. Aesthetics ---
    # Title
    n_obs = len(plot_data)
    ax.set_title(f"{wrap_label(feature_name, 50)}\n(n={n_obs:,})", fontsize=9, pad=10)
    
    # Labels
    if i % N_COLS == 0:
        ax.set_ylabel('PRS', fontsize=9)
    else:
        ax.set_ylabel('') 
    if i >= len(prs_features) - N_COLS:
        ax.set_xlabel('Subtype', fontsize=9)
    else:
        ax.set_xlabel('')

    # Clean up spines and ticks
    sns.despine(ax=ax)
    ax.tick_params(axis='y', labelsize=9)
    ax.grid(axis='y', linestyle='--', alpha=0.3)

# Hide empty subplots
for j in range(len(prs_features), len(axes)):
    axes[j].axis('off')

plt.tight_layout()
plt.subplots_adjust(hspace=1, wspace=0.5)

# Save
out_file = os.path.join(OUTPUT_DIR, 'prs_difference_boxplot.png')
plt.savefig(out_file, bbox_inches='tight', dpi=500)

print(f"Figure saved successfully to: {out_file}")