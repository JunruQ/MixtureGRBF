import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

# 读取数据

protein_table_path = 'input/ukb/ukb_covreg1_trans1_nanf1_biom0.csv'
df = pd.read_csv(protein_table_path)
df.sort_values(by='stage', inplace=True)
df = df[df['stage'] >= 40]
# 7:end 的蛋白数据根据stage取mean
df_mean = df.iloc[:, 7:].groupby(df['stage']).mean()
X = df_mean.T

colors = [(0.53, 0.81, 0.98), (0, 0, 0), (1, 1, 0)]  # 天蓝色 -> 黑色 -> 黄色
cmap_name = 'skyblue_black_yellow'
cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=256)

# 设置全局字体为 Arial，加粗
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.weight'] = 'bold'

# 绘制 heatmap
fig, ax = plt.subplots(figsize=(15, 10))
vmax = 0.5

# 显示主图像
im = ax.imshow(X, aspect='auto', interpolation='nearest', cmap=cm, origin='upper',
               vmin=-vmax, vmax=vmax)

# # 添加 colorbar，并设置字体加粗、变大
# cbar = plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
# cbar.set_label('z-score', fontsize=28, weight='bold', family='Arial')
# cbar.ax.tick_params(labelsize=24)  # colorbar 刻度字体大小

# 添加 stage 标签
n_stage = X.shape[1]
# if 'stage' in df.columns:
#     ax.set_xticks(np.arange(0, n_stage, 5))
#     ax.set_xticklabels(np.arange(40, 71, 5), fontsize=24, fontweight='bold', family='Arial')
#     ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=True)
#     ax.axis('on')
# ax.set_yticks([])

ax.axis('off')
# 紧凑布局并保存
plt.tight_layout()
plt.savefig('output/result_analysis/ukb_MixtureGRBF_cv_nsubtype_biom17/5_subtypes/protein_stage_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

# Define the colormap: skyblue -> black -> yellow
colors = [(0.53, 0.81, 0.98), (0, 0, 0), (1, 1, 0)]
cmap_name = 'skyblue_black_yellow'
cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=256)

# Create a figure for the colorbar
fig, ax = plt.subplots(figsize=(1, 8))  # Narrow width, tall height for vertical colorbar

# Create a gradient array for the colorbar
gradient = np.linspace(0, 1, 256).reshape(-1, 1)

# Display the colorbar as an image
ax.imshow(gradient, aspect='auto', cmap=cm, origin='upper')

# Remove all axes, ticks, and labels
ax.axis('off')

# Save the colorbar
plt.tight_layout()
plt.savefig('output/result_analysis/ukb_MixtureGRBF_cv_nsubtype_biom17/5_subtypes/protein_stage_heatmap_colorbar.png', dpi=300, bbox_inches='tight')
plt.close()