import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import ttest_ind, chi2_contingency

# 基本设置
PREDICTED_STAGE_PATH = 'analysis/result_analysis/output/predicted_stage.csv'
COV_INFO_PATH = 'data/ukb_cov_info.csv'
nsubtype = 5
exp_name = 'ukb_MixtureGRBF_cv_nsubtype_biom17'
SUBTYPE_STAGE_PATH = f'output/{exp_name}/{nsubtype}_subtypes/subtype_stage.csv'
SUBTYPE_ORDER_PATH = f'output/result_analysis/{exp_name}/{nsubtype}_subtypes/all_cause_mortality_order.csv'
OUTPUT_DIR = f'output/result_analysis/{exp_name}/{nsubtype}_subtypes'

# 读取基础数据
try:
    subtype_order = pd.read_csv(SUBTYPE_ORDER_PATH, header=None)
except:
    subtype_order = pd.DataFrame(np.array(list(range(1, nsubtype+1))))
df = pd.read_csv(PREDICTED_STAGE_PATH)
cov_df = pd.read_csv(COV_INFO_PATH)
subtype_stage = pd.read_csv(SUBTYPE_STAGE_PATH)
subtype_order_map = {int(subtype_order.iloc[i]): i+1 for i in range(nsubtype)}
subtype_stage['subtype'] = subtype_stage['subtype'].map(subtype_order_map)

# 合并数据
merged_df = pd.merge(df, subtype_stage[['PTID', 'subtype']], 
                    left_on='RID', right_on='PTID', how='left')
merged_df = pd.merge(merged_df, cov_df[['eid', 'Townsend', 'education', 'BMI', 'Smoking', 'Alcohol']], 
                    left_on='RID', right_on='eid', how='left')

# 将Alcohol和Smoking限制为0,1,2，其他转换为NA
for var in ['Alcohol', 'Smoking']:
    merged_df[var] = merged_df[var].apply(lambda x: x if x in [0, 1, 2] else np.nan)

# 定义变量类型
numeric_vars = ['Townsend', 'BMI']
categorical_vars = ['Smoking', 'Alcohol', 'education']

# 使用更清晰的色卡（例如 'tab10'）
colors = sns.color_palette('colorblind', n_colors=6)  # tab10 提供10种清晰颜色，足够覆盖类别

# 创建大图
fig = plt.figure(figsize=(15, 10))
plt.style.use('classic')

# 处理数值变量
for i, var in enumerate(numeric_vars, 1):
    ax = plt.subplot(2, 3, i)
    
    # 计算显著性
    p_values = []
    subtype_1_data = merged_df[merged_df['subtype'] == 1][var].dropna()
    for j in range(2, nsubtype + 1):
        subtype_data = merged_df[merged_df['subtype'] == j][var].dropna()
        stat, p = ttest_ind(subtype_1_data, subtype_data)
        p_values.append(p)
    p_values_corrected = np.minimum(np.array(p_values) * (nsubtype - 1), 1.0)
    
    # 绘制箱线图（保留原来的颜色方案）
    sns.boxplot(x='subtype', y=var, data=merged_df, ax=ax, width=0.3, notch=True,
                palette=['#0072BD', '#D95319', '#EDB120', '#7E2F8E', '#77AC30'],  # 数值变量用原色
                boxprops={'edgecolor': 'black', 'linewidth': 1},
                medianprops={'color': 'black', 'linewidth': 1.5},
                whiskerprops={'color': 'black', 'linewidth': 1},
                capprops={'color': 'black', 'linewidth': 1},
                flierprops={'marker': '+', 'markerfacecolor': 'black', 'markersize': 4, 'alpha': 0.5})
    
    ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
    
    # 添加显著性标注
    max_y = merged_df[var].max()
    min_y = merged_df[var].min()
    delta = max_y - min_y
    for j, p in enumerate(p_values_corrected, start=1):
        if p <= 0.05:
            stars = '*' if p > 0.01 else '**' if p > 0.001 else '***'
            ax.text(j, max_y + 0.05 * delta, stars, ha='center', va='bottom', fontsize=8)
    
    ax.set_ylim(min_y - 0.1 * delta, max_y + 0.1 * delta)
    ax.set_title(f'{var} by Subtype', fontsize=10, pad=10)
    ax.set_xlabel('Subtype', fontsize=9)
    ax.set_ylabel(f'{var}', fontsize=9)
    ax.tick_params(axis='both', labelsize=8)
    ax.grid(True, linestyle='--', alpha=0.2)
    
    print(f"\nP-values (Bonferroni corrected) for {var} - Subtype 1 vs others:")
    for j, p in enumerate(p_values_corrected, start=2):
        print(f"Subtype 1 vs Subtype {j}: p = {p:.4f}")

# 处理分类变量
for i, var in enumerate(categorical_vars, 3):  # 从第3个子图开始（因为只有2个数值变量）
    ax = plt.subplot(2, 3, i)
    
    # 创建交叉表并归一化
    crosstab = pd.crosstab(merged_df['subtype'], merged_df[var], normalize='index')
    
    # 与Subtype 1比较的卡方检验
    p_values = []
    for j in range(2, nsubtype + 1):
        pair_data = merged_df[merged_df['subtype'].isin([1, j])][[var, 'subtype']].dropna()
        contingency_table = pd.crosstab(pair_data['subtype'], pair_data[var])
        if contingency_table.shape[0] > 1 and contingency_table.shape[1] > 0:
            _, p, _, _ = chi2_contingency(contingency_table)
            p_values.append(p)
        else:
            p_values.append(1.0)
    p_values_corrected = np.minimum(np.array(p_values) * (nsubtype - 1), 1.0)
    
    # 绘制堆叠条形图，使用新色卡
    crosstab.plot(kind='bar', stacked=True, color=colors[:len(crosstab.columns)],
                  width=0.8, edgecolor='black', linewidth=0.5, ax=ax)
    
    # 添加显著性标注
    for j, p in enumerate(p_values_corrected, start=1):
        if p <= 0.05:
            stars = '*' if p > 0.01 else '**' if p > 0.001 else '***'
            ax.text(j, 1.05, stars, ha='center', va='bottom', fontsize=8)
    
    # 设置图表属性
    ax.set_title(f'{var} Distribution by Subtype', fontsize=10, pad=10)
    ax.set_xlabel('Subtype', fontsize=9)
    ax.set_ylabel('Proportion', fontsize=9)
    ax.tick_params(axis='x', rotation=0, labelsize=8)
    ax.tick_params(axis='y', labelsize=8)
    
    # 动态生成图例（因为 education 的类别可能与 Smoking/Alcohol 不同）
    if var in ['Smoking', 'Alcohol']:
        ax.legend(title=var, labels=['Never (0)', 'Previous (1)', 'Current (2)'],
                 bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    else:  # education
        unique_vals = sorted(merged_df[var].dropna().unique())
        ax.legend(title=var, labels=[str(int(val)) for val in unique_vals],
                 bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    ax.set_ylim(0, 1.15)
    
    print(f"\nChi-square p-values (Bonferroni corrected) for {var} - Subtype 1 vs others:")
    for j, p in enumerate(p_values_corrected, start=2):
        print(f"Subtype 1 vs Subtype {j}: p = {p:.4f}")

# 添加总体显著性说明
fig.text(0.5, 0.01, 'Significance (vs Subtype 1): * p<0.05, ** p<0.01, *** p<0.001',
         ha='center', fontsize=10)

# 调整布局
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(f'{OUTPUT_DIR}/combined_plots.png', dpi=300, bbox_inches='tight')
plt.close()