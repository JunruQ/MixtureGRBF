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
fig_width = 6 * cm_to_inch
fig_height_per_subplot = 3.6 * cm_to_inch  # Increased to accommodate histograms
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

    for biom in star_proteins:
        if biom not in trajectory_df.columns:
            print(f"Warning: {biom} not found in trajectory data for Subtype {i+1}.")
            continue
        ax.plot(range(39, 71), trajectory_df[biom], color='white', linewidth=2.0, alpha=0.5)
        ax.plot(range(39, 71), trajectory_df[biom], color=star_protein_colors[biom],
                linewidth=1.5, alpha=0.5, label=biom if i == 0 else None)
        if i == 0:
            legend_handles.append(plt.Line2D([0], [0], color=star_protein_colors[biom],
                                             linewidth=1.5, alpha=0.8, label=biom))

    # Plot age distribution histogram
    subtype_data = subtype_stage[subtype_stage['subtype'] == i+1]
    if 'stage' not in subtype_data.columns:
        print(f"Warning: 'age' column not found in subtype_stage for Subtype {i+1}.")
    else:
        # Create inset axes for histogram at the bottom
        hist_height = 0.3 # Fraction of subplot height for histogram
        hist_ax = ax.inset_axes([0, 0, 1, hist_height])  # Position: [left, bottom, width, height]
        # hist_ax.hist(subtype_data['stage'], bins=32, range=(39, 71), color='lightblue', alpha=0.7)
        # 绘制频数直方图
        counts, _, _ = hist_ax.hist(subtype_data['stage'], bins=np.arange(38.5, 71.5, 1), color='lightblue', alpha=0.7)

        # 设置 y 轴为频数
        # hist_max_y = np.ceil(max(counts) / 10) * 10
        hist_max_y = 500
        hist_ax.set_ylim(0, hist_max_y)
        hist_ax.set_xlim(37.5, 71.5)

        # 设置 y 轴为整十数刻度
        hist_ax.set_yticks(np.arange(0, hist_max_y + 1, 200))
        hist_ax.yaxis.set_ticks_position('right')  # Place ticks on right
        hist_ax.yaxis.set_label_position('right')
        hist_ax.tick_params(axis='y', labelsize=12, labelcolor='black', direction='in', width=0.5)
        for label in hist_ax.get_yticklabels():
            label.set_fontfamily('Arial')
        hist_ax.set_xticks([])  # Hide x-axis ticks to avoid overlap with main plot
        hist_ax.patch.set_alpha(0.0)  # Transparent background
        hist_ax.spines['top'].set_visible(False)
        hist_ax.spines['right'].set_visible(True)  # Show right spine for y-axis
        hist_ax.spines['left'].set_visible(False)
        hist_ax.spines['bottom'].set_color('black')
        hist_ax.spines['bottom'].set_linewidth(0.5)
        hist_ax.spines['right'].set_color('black')
        hist_ax.spines['right'].set_linewidth(0.5)

    # Add subtype annotation
    ax.text(0.04, 0.95, f'Subtype {i+1}: N = {subtype_data.shape[0]}', transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='left', fontfamily='Arial')

    ax.set_xticks([40, 50, 60, 70])
    # Set labels and ticks
    if i == nsubtype - 1:
        ax.set_xlabel('Age', fontsize=12, fontfamily='Arial', labelpad=5)
        
    else:
        ax.set_xlabel('')
        ax.set_xticklabels([])
    ax.tick_params(axis='x', labelsize=12, which='both', labelcolor='black')
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily('Arial')
    # dashline
    ax.axhline(0, linestyle='--', color='black', linewidth=1.0) # dashline
    ax.set_ylim(-2.5, 2.5)
    ax.tick_params(direction='in', width=0.5)
    ax.grid(False)

# Add legend
fig.legend(
    handles=legend_handles,
    loc='lower left',
    bbox_to_anchor=(0.2, 0.88),
    ncol=2,
    frameon=True,
    framealpha=1,
    edgecolor='black',
    title='Star Proteins',
    title_fontsize=12,
    fontsize=12,
    columnspacing=1.0,
    handletextpad=0.5
)

# Save
plt.savefig(f'{OUTPUT_DIR}/star_protein_trajectories_with_histograms.png',
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print(f"Trajectory plots with star proteins and age distribution histograms saved to {OUTPUT_DIR}/star_protein_trajectories_with_histograms.png")