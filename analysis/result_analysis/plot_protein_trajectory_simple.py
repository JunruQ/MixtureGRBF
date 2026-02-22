import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# Reset style to default
plt.style.use('default')

# Set font and other parameters before figure creation
mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['axes.labelsize'] = 12
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12
mpl.rcParams['xtick.color'] = 'black'
mpl.rcParams['ytick.color'] = 'black'
mpl.rcParams['axes.titlesize'] = 12
mpl.rcParams['axes.labelcolor'] = 'black'
mpl.rcParams['axes.edgecolor'] = 'black'
mpl.rcParams['axes.linewidth'] = 0.5

# File paths and parameters
nsubtype = 5
exp_name = 'ukb_MixtureGRBF_cv_nsubtype_biom17'
INPUT_TABLE_PATH = 'input/ukb/ukb_covreg1_trans1_nanf1_biom17.csv'
SUBTYPE_STAGE_PATH = f'output/{exp_name}/{nsubtype}_subtypes/subtype_stage.csv'
OUTPUT_DIR = f'output/result_analysis/{exp_name}/{nsubtype}_subtypes'
SUBTYPE_ORDER_PATH = f'output/result_analysis/{exp_name}/{nsubtype}_subtypes/all_cause_mortality_order.csv'
CLASSIFICATION_PATH = 'analysis/result_analysis/protein_classification_form.csv'

# Read data
df = pd.read_csv(INPUT_TABLE_PATH)
subtype_order = pd.read_csv(SUBTYPE_ORDER_PATH, header=None)
classification = pd.read_csv(CLASSIFICATION_PATH)
biomarker_names = df.iloc[:, 7:].columns.tolist()

import utils.utils as utils
subtype_stage = utils.get_subtype_stage(exp_name, nsubtype)
# Get classification and protein mapping
classification_dict = {}
xtick_labels_legend = classification.iloc[:, 0].tolist()
for j in range(classification.shape[0]):
    c = eval(classification.iloc[j, 1])
    c = sorted(list(set(c).intersection(biomarker_names)))
    for biom in c:
        classification_dict[biom] = j

# Figure parameters
cm_to_inch = 1 / 2.54
fig_width = 3 * cm_to_inch
fig_height_per_subplot = 1.2 * cm_to_inch  # Increased to accommodate histograms
fig_height = fig_height_per_subplot * nsubtype + 1.5 * cm_to_inch

# Star proteins and colors
star_proteins = ['CXCL17', 'SCARF2', 'PODXL2', 'FBLN2']
star_protein_colors = {
    'CXCL17': '#c0392b',
    'SCARF2': '#f1c40f',
    'PODXL2': '#2ecc71',
    'FBLN2': '#2980b9'
}

# Create figure and subplots
fig, axes = plt.subplots(nrows=nsubtype, ncols=1, figsize=(fig_width, fig_height), dpi=300)
fig.subplots_adjust(hspace=0.15, top=0.88, bottom=0.08, left=0.2, right=0.95)

legend_handles = []

for i in range(nsubtype):
    ax = axes[i]
    k = int(subtype_order.iloc[i, 0])
    TRAJECTORY_PATH = f'output/{exp_name}/{nsubtype}_subtypes/trajectory{k}.csv'

    try:
        trajectory_df = pd.read_csv(TRAJECTORY_PATH)
    except FileNotFoundError:
        print(f"Warning: {TRAJECTORY_PATH} not found, skipping Subtype {i+1}.")
        continue

    # Plot trajectories
    max_y = max(trajectory_df[biomarker_names].abs().max()) * 1.1

    for biom in biomarker_names:
        if biom not in star_proteins and biom in trajectory_df.columns:
            ax.plot(range(39, 71), trajectory_df[biom], color='gray', linewidth=0.5, alpha=0.1)
    
    # ax.set_xticks([40, 50, 60, 70])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    # Set labels and ticks
    # if i == nsubtype - 1:
    #     ax.set_xlabel('Age', fontsize=12, fontfamily='Arial', labelpad=5)
        
    # else:
    #     ax.set_xlabel('')
    #     ax.set_xticklabels([])
    ax.set_xlabel('')
    ax.set_xticklabels([])
    ax.tick_params(axis='x', labelsize=12, which='both', labelcolor='black')
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily('Arial')
    # dashline
    ax.axhline(0, linestyle='--', color='black', linewidth=0.5) # dashline
    ax.set_ylim(-2.5, 2.5)
    ax.tick_params(direction='in', width=0.5)
    ax.grid(False)


# 关闭四周的框
for ax in axes:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
# Save
plt.savefig(f'{OUTPUT_DIR}/protein_trajectories_simple.png',
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close()