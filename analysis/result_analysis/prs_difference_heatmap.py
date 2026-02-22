import utils.utils as utils
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.stats.multitest import multipletests
import textwrap

exp_name = 'ukb_MixtureGRBF_cv_nsubtype_biom17'
nsubtype = 5

result_df = pd.read_csv(f'output/result_analysis/{exp_name}/{nsubtype}_subtypes/prs_difference.csv')

# 定义显著性阈值
threshold_star = np.log10(1 / 0.05)    # ≈1.3010, *
threshold_dstar = np.log10(1 / 0.01)   # ≈2.0000, **
threshold_tstar = np.log10(1 / 0.001)  # ≈3.0000, ***

# Pivot the DataFrame to create a matrix with features as rows and subtypes as columns
pivot_df = result_df.pivot(index='feature', columns='subtype', values='signed_log10FDR')

plt.rcParams.update({
    'font.family': 'Arial', 
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 8,
})

# Create the heatmap with full data (no masking)
fig, ax = plt.subplots(figsize=(10/2.54, 26.7/2.54))  # Adjust figsize based on number of features
# im = ax.imshow(pivot_df.values, cmap='RdBu_r', aspect='auto', vmin=-threshold_tstar*1.2, vmax=threshold_tstar*1.2)  # 扩展范围以显示所有值
# im = ax.imshow(pivot_df.values, cmap='RdBu_r', aspect='auto', vmin=-np.max(pivot_df), vmax=np.max(pivot_df))
im = ax.imshow(pivot_df.values, cmap=utils.custom_rdbu_r, aspect='auto', vmin=-np.max(pivot_df), vmax=np.max(pivot_df))
# Add significance annotations
for i in range(len(pivot_df.index)):
    for j in range(len(pivot_df.columns)):
        val = np.abs(pivot_df.iloc[i, j])
        if val > threshold_tstar:
            symbol = '***'
        elif val > threshold_dstar:
            symbol = '**'
        elif val > threshold_star:
            symbol = '*'
        else:
            continue
        ax.text(j, i+0.16, symbol, ha='center', va='center', color='white' if val > np.max(np.abs(pivot_df))/2 else 'black', fontsize=12, weight='bold', transform=ax.transData)

# Add colorbar
cbar = ax.figure.colorbar(im, ax=ax, shrink=0.5)
cbar.set_label('Signed -log10(FDR)', rotation=270, labelpad=13)
cbar.outline.set_visible(False)
# cbar.ax.tick_params(length=0.2)


# 取得当前位置 [x0, y0, width, height]
pos = cbar.ax.get_position()

# 手动调整高度 (缩短一半并向下平移一些)
cbar.ax.set_position([pos.x0, pos.y0 + 0.1, pos.width, pos.height * 0.25])

# Set labels
ax.set_xticks(np.arange(len(pivot_df.columns)))
ax.set_yticks(np.arange(len(pivot_df.index)))
feature_labels = ['\n'.join(textwrap.wrap(i, width=25)) for i in pivot_df.index]
ax.set_xticklabels(pivot_df.columns, ha='right')
ax.set_yticklabels(feature_labels)
ax.set_xlabel('Subtype')
# ax.tick_params(axis='y', which='both', length=0)
# ax.set_ylabel('Features')
# ax.set_title('Polygenic risk scores difference by subtype')

plt.tight_layout()
output_path = f'output/result_analysis/{exp_name}/{nsubtype}_subtypes/prs_difference.png'
plt.savefig(output_path, dpi=300)

print(f"Figure saved to: {output_path}")