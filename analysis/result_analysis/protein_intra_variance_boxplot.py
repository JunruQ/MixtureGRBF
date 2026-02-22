# ==============================================================================
# 0. SETUP: LIBRARIES AND CONFIGURATION
# ==============================================================================
import pandas as pd
import seaborn as sns
import os
import matplotlib.pyplot as plt
import utils.utils as utils

# --- [!] 用户配置: 文件路径 ---
nsubtype = 5
result_folder = 'ukb_MixtureGRBF_cv_nsubtype_biom17'
output_dir = f'./output/result_analysis/{result_folder}/{nsubtype}_subtypes'

OUTPUT_IMAGE_PATH = output_dir + '/protein_intra_variance_boxplot.png'


# --- Ensure output directory exists ---
os.makedirs(output_dir, exist_ok=True)

# --- Set aesthetic parameters ---
plt.rcParams.update({
    'font.family': 'Arial',  # 更清晰的字体
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'axes.spines.top': False,
    'axes.spines.right': False
})

prot_var_path = f'./output/result_analysis/{result_folder}/{nsubtype}_subtypes/protein_intra_variance.csv'

prot_var_table = pd.read_csv(prot_var_path)

subtype_stage = utils.get_subtype_stage(result_folder, nsubtype)

prot_var_table = prot_var_table.merge(subtype_stage, left_on='eid', right_on='PTID', how='left')

# --- Create styled box plot ---
plt.figure(figsize=(6/2.54, 6/2.54))
ax = sns.boxplot(
    x='subtype',
    y='std_dev',
    data=prot_var_table,
    palette=utils.subtype_colors,  # 更现代的渐变色
    width=0.3,
    linewidth=1.5,
    flierprops=dict(
        marker='+',
        markersize=4,
        alpha=0.8
    )
)

# --- Add annotations ---
plt.xlabel('Subtype')
plt.ylabel('Protein Variability')

# --- Improve grid and layout ---
# plt.grid(True, axis='y', linestyle='-', alpha=0.3)
# plt.tight_layout(pad=2.5)

# --- Save the plot ---
plt.savefig(OUTPUT_IMAGE_PATH, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to: {OUTPUT_IMAGE_PATH}")