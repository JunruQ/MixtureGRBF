import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import utils.utils as utils

# 设置字体为 Arial Bold，字号 9pt（约等于 matplotlib 中的 9）
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12

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

# 创建图形（7cm × 7cm，单位为英寸）
cm_to_inch = 1 / 2.54
fig, ax = plt.subplots(figsize=(7 * cm_to_inch, 7 * cm_to_inch))

# 设置颜色
colors = utils.subtype_colors[:nsubtype]

# # 绘制小提琴图
# sns.violinplot(
#     x='subtype',
#     y='stage_delta',
#     data=merged_df,
#     width=0.6,  # Adjust width to leave space for boxplot
#     palette=colors,
#     inner=None,  # Remove inner annotations to avoid clutter
#     alpha=0.6,
#     ax=ax
# )

# 绘制箱线图
# sns.boxplot(
#     x='subtype',
#     y='stage_delta',
#     data=merged_df,
#     width=0.1,  # Narrow boxplot to fit within violin
#     palette=colors,
#     # flierprops={'marker': 'None'},  # No outliers (handled by jitter)
#     boxprops={'edgecolor': 'black', 'linewidth': 0.5},
#     medianprops={'color': 'black', 'linewidth': 1},
#     whiskerprops={'color': 'black', 'linewidth': 0.5},
#     capprops={'color': 'black', 'linewidth': 0.5},
#     ax=ax
# )

sns.boxplot(
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
    flierprops={'marker': '+', 'markerfacecolor': 'black', 'markersize': 4, 'alpha': 0.5},
    ax=ax
)

ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5)

# 设置标题和标签
ax.set_title('Protein age gap', fontsize=12, pad=10)
ax.set_xlabel('Subtype')
ax.set_ylabel('Age gap (years)')

# 设置坐标轴刻度字体
ax.tick_params(axis='both', labelsize=12)

# 设置边框和刻度线粗细
for spine in ax.spines.values():
    spine.set_linewidth(0.5)  # Match original code
ax.tick_params(axis='both', width=0.5)  # Match original code

# 调整布局，确保字体不被裁剪
plt.subplots_adjust(left=0.2, right=0.95, top=0.85, bottom=0.2)

# 保存图像
plt.savefig('output/result_analysis/ukb_MixtureGRBF_cv_nsubtype_biom17/5_subtypes/protein_predicted_age_delta_boxplot.png', dpi=300, bbox_inches='tight')
plt.close()

print("✅ Box plot saved to: 'output/result_analysis/ukb_MixtureGRBF_cv_nsubtype_biom17/5_subtypes/protein_predicted_age_delta_boxplot.png'")