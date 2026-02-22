import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import ttest_ind
import utils.utils as utils

# 读取数据
PREDICTED_STAGE_PATH = 'analysis/result_analysis/stage_pred/ukb_covreg1_trans1_nanf1_biom0_stage_pred.csv'
nsubtype = 5
exp_name = 'ukb_MixtureGRBF_cv_nsubtype_biom17'
SUBTYPE_STAGE_PATH = f'output/{exp_name}/{nsubtype}_subtypes/subtype_stage.csv'
OUTPUT_DIR = f'output/result_analysis/{exp_name}/{nsubtype}_subtypes'

df = pd.read_csv(PREDICTED_STAGE_PATH)
df['stage_delta'] = df['stage_pred'] - df['stage'] 
subtype_stage = utils.get_subtype_stage(exp_name, nsubtype)
merged_df = pd.merge(df, subtype_stage[['PTID', 'subtype']], 
                    left_on='RID', right_on='PTID', how='left')

# 计算显著性差异（与Subtype 1比较）
p_values = []
subtype_1_data = merged_df[merged_df['subtype'] == 1]['stage_delta'].dropna()
for i in range(2, nsubtype + 1):
    subtype_data = merged_df[merged_df['subtype'] == i]['stage_delta'].dropna()
    stat, p = ttest_ind(subtype_1_data, subtype_data)
    p_values.append(p)

# Bonferroni校正
p_values_corrected = np.minimum(np.array(p_values) * (nsubtype - 1), 1.0)

# 创建图形
plt.figure(figsize=(4, 4))
plt.style.use('classic')
colors = utils.subtype_colors[:nsubtype]

# 绘制箱线图
ax = sns.boxplot(
    x='subtype',
    y='stage_delta',
    data=merged_df,
    width=0.3,
    notch=True,
    palette=colors,
    boxprops={'edgecolor': 'black', 'linewidth': 1},
    medianprops={'color': 'black', 'linewidth': 1.5},
    whiskerprops={'color': 'black', 'linewidth': 1},
    capprops={'color': 'black', 'linewidth': 1},
    flierprops={'marker': '+', 'markerfacecolor': 'black', 'markersize': 4, 'alpha': 0.5}
)

# 添加参考线
plt.axhline(y=0, color='k', linestyle='--', linewidth=0.5)

# 添加显著性标注
max_y = merged_df['stage_delta'].max() * 1.1
alpha = 0.05
for i, p in enumerate(p_values_corrected, start=1):  # 从Subtype 2开始
    if p <= alpha:
        stars = '*' if p > 0.01 else '**' if p > 0.001 else '***'
        plt.text(i, max_y, stars, ha='center', va='bottom', fontsize=10)

# 设置标题和标签
plt.title('Age Delta by Subtype\n(Pred from Linear Lasso)', fontsize=10, pad=20)
plt.xlabel('Subtype', fontsize=9)
plt.ylabel('Age Delta (Pred - True)', fontsize=9)
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)
plt.grid(True, linestyle='--', alpha=0.2)

# 添加显著性说明（移到最下侧）
sig_text = 'Significance (vs Subtype 1): * p<0.05, ** p<0.01, *** p<0.001'

plt.text(0.5, -0.15, sig_text, transform=plt.gca().transAxes, 
         fontsize=8, va='top', ha='center')

# 调整布局
plt.subplots_adjust(left=0.15, right=0.95, top=0.85, bottom=0.2)  # 增加bottom以容纳下方文字

# 保存图形
plt.savefig(OUTPUT_DIR + '/age_delta.png', dpi=300, bbox_inches='tight')  # 使用bbox_inches确保文字完整保存
plt.close()

# 打印p值
print("P-values (Bonferroni corrected) for Subtype 1 vs others:")
for i, p in enumerate(p_values_corrected, start=2):
    print(f"Subtype 1 vs Subtype {i}: p = {p:.4f}")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 假设已有数据加载部分保持不变直到 merged_df 创建
# 在你的代码基础上添加以下部分：

# 设置子图数量和布局
nsubtype = 5  # 根据你的数据
fig, axes = plt.subplots(nrows=1, ncols=nsubtype, figsize=(nsubtype * 3, 4), sharey=True)
plt.style.use('classic')

# 颜色方案
colors = ['#0072BD', '#D95319', '#EDB120', '#7E2F8E', '#77AC30']  # 为每个 subtype 指定颜色

# 为每个 subtype 绘制子图
for i in range(nsubtype):
    subtype_num = i + 1
    subtype_data = merged_df[merged_df['subtype'] == subtype_num]
    
    # 绘制散点图
    sns.scatterplot(
        x='stage',  # 真实年龄
        y='stage_delta',  # Pred - True
        data=subtype_data,
        ax=axes[i],
        color=colors[i],  # 每个 subtype 用不同颜色
        alpha=0.5,
        s=30
    )
    
    # 添加平滑趋势线
    sns.regplot(
        x='stage',
        y='stage_delta',
        data=subtype_data,
        ax=axes[i],
        scatter=False,  # 不重复绘制散点
        color='black',
        line_kws={'linewidth': 1, 'linestyle': '--'}
    )
    
    # 添加参考线
    axes[i].axhline(y=0, color='k', linestyle='--', linewidth=0.5)
    
    # 设置标题和标签
    axes[i].set_title(f'Subtype {subtype_num}', fontsize=10)
    axes[i].set_xlabel('True Age', fontsize=9)
    if i == 0:  # 只在最左侧子图设置 y 轴标签
        axes[i].set_ylabel('Age Delta (Pred - True)', fontsize=9)
    else:
        axes[i].set_ylabel('')
    
    # 设置刻度字体
    axes[i].tick_params(axis='both', labelsize=8)
    axes[i].grid(True, linestyle='--', alpha=0.2)

# 调整布局
plt.tight_layout()

# 保存图形
output_path = OUTPUT_DIR + '/age_delta_vs_true_age_subplots.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"Figure saved to: {output_path}")