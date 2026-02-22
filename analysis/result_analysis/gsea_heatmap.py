import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib as mpl
import pandas as pd
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

nsubtype = 5
exp_name = 'ukb_MixtureGRBF_cv_nsubtype_biom17'
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12  # 减小y轴标签字体以适应更多指标
OUTPUT_DIR = f'output/result_analysis/{exp_name}/{nsubtype}_subtypes'

df = pd.read_csv(f'{OUTPUT_DIR}/gsea_results.csv')
df['neg_log10_p'] = -np.log10(df['p.adjust'])  # convert p-values to -log10 scale

top_n = 5
top_df = df.groupby('Subtype', group_keys=False).apply(
    lambda g: g.nlargest(top_n, 'neg_log10_p')
)

top_pathways = top_df['Description'].unique().tolist()

df_selected = df[df['Description'].isin(top_pathways)]
df_selected['Description'] = df_selected['Description'].astype(str)
desc_order = sorted(df_selected['Description'].unique(), key=lambda x: x.lower())
subtype_order = sorted(df_selected['Subtype'].unique())

print(df_selected)
# 建立映射
desc_to_y = {desc: i for i, desc in enumerate(desc_order)}
subtype_to_x = {subtype: i for i, subtype in enumerate(subtype_order)}

# 自定义 suppressed 的颜色映射：从 #453697 到 #26a9d3
cmap_suppressed = mpl.colors.LinearSegmentedColormap.from_list(
    "suppressed_custom", ["#453697", "#26a9d3"]
)

# 自定义 activated 的颜色映射：从 #bd2f21 到 #f8e25d
cmap_activated = mpl.colors.LinearSegmentedColormap.from_list(
    "activated_custom", ["#bd2f21", "#f8e25d"]
)

# neg_log10_p 范围
p_min, p_max = -np.log10(0.05), df_selected['neg_log10_p'].max()

# 定义归一化，使用 TwoSlopeNorm，中心点为 0
norm = mpl.colors.TwoSlopeNorm(vmin=-p_max, vcenter=0, vmax=p_max)

# 获取反向的 suppressed 颜色映射
cmap_suppressed_r = cmap_suppressed.reversed()

# 创建分段颜色映射
colors = []
n_bins = 256  # 颜色条的分辨率
colors.extend(cmap_suppressed_r(np.linspace(0, 1, n_bins//2)))
colors.extend([(0, 0, 0)] * (n_bins//4))  # 黑色区域
colors.extend(cmap_activated(np.linspace(0, 1, n_bins//2)))

# 创建自定义颜色映射
custom_cmap = mpl.colors.LinearSegmentedColormap.from_list("custom_cmap", colors, N=n_bins)

# ===== 开始画图 =====
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as mpl_cm
from matplotlib.gridspec import GridSpec

# 创建GridSpec
n_cols = len(subtype_order)
fig = plt.figure(figsize=(10, 6))
gs = GridSpec(1, n_cols + 2, width_ratios=[0.1] * n_cols + [3.5, 0.15])  # 最后一个为colorbar

axes = [fig.add_subplot(gs[0, i]) for i in range(n_cols)]

# 为每个子图设置背景色
for ax in axes:
    ax.set_facecolor("black")

# 在每个子图中绘制对应的数据
for i, subtype in enumerate(subtype_order):
    ax = axes[i]
    
    # 获取该 subtype 的数据
    subtype_data = df_selected[df_selected['Subtype'] == subtype]
    
    for _, row in subtype_data.iterrows():
        y = desc_to_y[row['Description']]
        
        # 决定颜色
        if row['.sign'] == 'suppressed':
            color = custom_cmap(norm(-row['neg_log10_p']))
        else:
            color = custom_cmap(norm(row['neg_log10_p']))
        
        # GeneRatio 决定填充宽度
        width = row['GeneRatio']
        
        # 在格子内部画一个矩形（左对齐）
        rect = patches.Rectangle(
            (0, y - 0.5),  # 左下角（x从0开始）
            width,         # 宽度（GeneRatio）
            1,             # 高度
            linewidth=0,
            facecolor=color
        )
        ax.add_patch(rect)
    
    # 设置每个子图的上方x轴
    ax.set_xlim(0, 1)
    if i == 0:
        ax.set_xticks([0])  # 显示x轴刻度
    elif i == 4:
        ax.set_xticks([1])
    else:
        ax.set_xticks([])
    ax.xaxis.set_ticks_position('top')  # x轴刻度显示在顶部
    ax.invert_yaxis()
    ax.set_title(subtype, color='white')
    secax = ax.secondary_xaxis("bottom")
    secax.set_xticks([0.5])
    secax.set_xticklabels([i+1])

    
    # 设置y轴（只在最后一个子图显示）
    if i == 4:
        ax.set_yticks(range(len(desc_order)))
        ax.set_yticklabels(desc_order)
        ax.yaxis.set_ticks_position('right')
        ax.yaxis.set_label_position('right')
    else:
        ax.set_yticks([])
    
    # 设置y轴范围
    ax.set_ylim(-0.5, len(desc_order) - 0.5)

# 添加颜色条
# cax = fig.add_subplot(gs[0, -1])
# sm = mpl_cm.ScalarMappable(cmap=custom_cmap, norm=norm)
# cbar = fig.colorbar(sm, cax=cax)
# cbar.set_ticks([-4, 0, 4])
# # 清空数字标签，让它只画线不显示数字
# cbar.set_ticklabels(["", "", ""])
# pos = cbar.ax.get_position()
# # cbar.set_label("Signed log10(FDR)")
# new_height = pos.height * 0.1
# new_y0 = pos.y0 + (pos.height - new_height) / 2
# cbar.ax.set_position([pos.x0, new_y0, pos.width, new_height])
# 手动在 figure 上创建一个小轴作为 colorbar
# [x0, y0, width, height] 的范围都是 figure 的归一化坐标 (0~1)
cax = fig.add_axes([0.92, 0.45, 0.02, 0.1])  # 这里手动指定位置和高度

sm = mpl_cm.ScalarMappable(cmap=custom_cmap, norm=norm)
cbar = fig.colorbar(sm, cax=cax)

# 只保留 0 和 ±4 的线，不显示数字
cbar.set_ticks([-4, 0, 4])
cbar.set_ticklabels(["", "", ""])
# cbar.set_label("Signed log10(FDR)")


# 调整布局
fig.subplots_adjust(wspace=0.08, bottom=0.15)  # 为下方x轴预留空间
plt.savefig('output/result_analysis/ukb_MixtureGRBF_cv_nsubtype_biom17/5_subtypes/gsea_heatmap.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print(f'Figure saved to {"output/result_analysis/ukb_MixtureGRBF_cv_nsubtype_biom17/5_subtypes/gsea_heatmap.png"}')