import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib as mpl
import pandas as pd
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

nsubtype = 5
exp_name = 'ukb_MixtureGRBF_cv_nsubtype_biom17'
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.labelsize'] = 9
plt.rcParams['axes.titlesize'] = 9
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9  # 减小y轴标签字体以适应更多指标
OUTPUT_DIR = f'output/result_analysis/{exp_name}/{nsubtype}_subtypes'

df = pd.read_csv(f'{OUTPUT_DIR}/gsea_results.csv')
df['neg_log10_p'] = -np.log10(df['p.adjust'])  # convert p-values to -log10 scale

top_n = 10
top_df = df.groupby('subtype', group_keys=False).apply(
    lambda g: g.nlargest(top_n, 'neg_log10_p')
)

top_pathways = top_df['Description'].unique().tolist()

df_selected = df[df['Description'].isin(top_pathways)]
df_selected['Description'] = df_selected['Description'].astype(str)
desc_order = sorted(df_selected['Description'].unique(), key=lambda x: x.lower())
subtype_order = sorted(df_selected['subtype'].unique())

print(df_selected)
# 建立映射
desc_to_y = {desc: i for i, desc in enumerate(desc_order)}
subtype_to_x = {subtype: i for i, subtype in enumerate(subtype_order)}

cmap_suppressed = mpl.colors.LinearSegmentedColormap.from_list("blue_style", ["#76b7e2", "#ffffff"])
cmap_activated = mpl.colors.LinearSegmentedColormap.from_list("red_style", ["#ffffff", "#dd786d"])

# 重新构建自定义颜色条
n_bins = 256
colors = []
colors.extend(cmap_suppressed(np.linspace(0, 1, n_bins//2)))
colors.extend(cmap_activated(np.linspace(0, 1, n_bins//2)))
custom_cmap = mpl.colors.LinearSegmentedColormap.from_list("custom_rd_bu", colors, N=n_bins)

# custom_cmap = plt.get_cmap('RdBu_r')
# neg_log10_p 范围
p_min, p_max = -np.log10(0.05), df_selected['neg_log10_p'].max()

# 定义归一化，使用 TwoSlopeNorm，中心点为 0
norm = mpl.colors.TwoSlopeNorm(vmin=-p_max, vcenter=0, vmax=p_max)

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib as mpl
import pandas as pd
import numpy as np
from matplotlib.gridspec import GridSpec
import matplotlib.cm as mpl_cm

# 建立全局通路顺序（可选，但如果你希望每个子图内部独立排序，则在循环内处理）
subtype_order = sorted(df_selected['subtype'].unique())
n_subtypes = len(subtype_order)

fig = plt.figure(figsize=(8/2.54, 15))
gs = GridSpec(n_subtypes, 1, hspace=0.15)

max_ratio = df_selected['GeneRatio'].max() * 1.1

for i, subtype in enumerate(subtype_order):
    ax = fig.add_subplot(gs[i, 0])
    ax.set_facecolor("white") # 3. 修改背景为白色
    
    # --- 核心修改点：在循环内部进行该 Subtype 的 Top N 筛选 ---
    # 1. 只取当前 subtype 的数据
    # 2. 按显著性排序并取前 top_n 个
    subtype_data = df[df['subtype'] == subtype].copy()
    subtype_data = subtype_data.nlargest(top_n, 'neg_log10_p')
    
    # 3. 为了让条形图从上到下排列（大的在上），这里需要 reverse 排序以便 ax.barh 绘制
    subtype_data = subtype_data.sort_values('GeneRatio', ascending=True)
    
    # 准备颜色
    plot_colors = []
    for _, row in subtype_data.iterrows():
        # 根据 sign 和显著性决定映射位置
        val = row['neg_log10_p'] if row['.sign'] == 'activated' else -row['neg_log10_p']
        plot_colors.append(custom_cmap(norm(val)))
    
    # 绘制横向柱状图
    bars = ax.barh(
        subtype_data['Description'], 
        subtype_data['GeneRatio'], 
        color=plot_colors,
        linewidth=0.5
    )
    # 修改 ylabel 朝向内部（文字向右偏移进入绘图区）
    ax.tick_params(axis='y', direction='in', pad=-10) # 负的 pad 会让文字进入轴内部
    plt.setp(ax.get_yticklabels(), ha="left", color="black")

    # 让刻度线长度为0，避免遮挡文字
    ax.tick_params(axis='y', length=0)

    # 样式设置
    # ax.set_title(f"Subtype {subtype}", loc='center', fontsize=10, fontweight='bold')
    # ax.set_ylabel(f"Subtype {subtype}")
    ax.set_xlim(0, max_ratio)
    # ax.set_xlabel("GeneRatio")

    # 移除顶部和右侧边框，并确保左侧轴线可见
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)
    
# 调整整体布局
plt.subplots_adjust(right=0.9, left=0.1, bottom=0.15, top=0.85) 

# 保存
output_path = f'{OUTPUT_DIR}/gsea_barplot.png'
plt.savefig(output_path, dpi=500, bbox_inches='tight')

def generate_gsea_colorbar(custom_cmap, norm, output_dir):
    """
    专门为 GSEA 柱状图生成对应的颜色条 Legend
    """
    # 1. 创建画布，尺寸可以根据需要调整
    fig, ax = plt.subplots(figsize=(0.8/2.54, 4/2.54)) 
    
    # 2. 使用 mpl.colorbar.ColorbarBase 创建颜色条
    # orientation='vertical' 为纵向，'horizontal' 为横向
    cb = mpl.colorbar.ColorbarBase(
        ax, 
        cmap=custom_cmap,
        norm=norm,
        orientation='vertical',
    )

    # 5. 美化设置
    ax.tick_params(labelsize=9)
    ax.spines['outline'].set_linewidth(0.8)

    # 6. 保存
    legend_path = f'{output_dir}/gsea_colorbar_legend.png'
    plt.savefig(legend_path, dpi=500, bbox_inches='tight', transparent=True)
    plt.show()
    print(f"Legend saved to: {legend_path}")

# 调用示例 (请确保在你的主程序中 custom_cmap 和 norm 已经定义)
generate_gsea_colorbar(custom_cmap, norm, OUTPUT_DIR)