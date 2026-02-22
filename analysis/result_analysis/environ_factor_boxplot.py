import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.multitest import multipletests

# MATLAB 风格设置
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.labelsize'] = 9
plt.rcParams['axes.titlesize'] = 9
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['axes.labelcolor'] = 'black'
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['axes.linewidth'] = 1.0

# 文件路径和参数
nsubtype = 4
INPUT_TABLE_PATH = 'data/prot_Modifiable_bl_data.csv'
exp_name = 'ukb_MixtureGRBF_test_biom17'
SUBTYPE_STAGE_PATH = f'output/{exp_name}/{nsubtype}_subtypes/subtype_stage.csv'
SUBTYPE_ORDER_PATH = f'output/result_analysis/{exp_name}/{nsubtype}_subtypes/all_cause_mortality_order.csv'

# 定义颜色（使用 MATLAB 风格单一颜色）
box_color = '#206491'  # MATLAB 蓝色

# 读取数据
df = pd.read_csv(INPUT_TABLE_PATH)
subtype_stage = pd.read_csv(SUBTYPE_STAGE_PATH)
df = pd.merge(df, subtype_stage, how='left', left_on='eid', right_on='PTID')
df = df.dropna(subset=['subtype'])  # 移除 subtype 为空的行
df['subtype'] = df['subtype'].astype(int)

try:
    subtype_order = pd.read_csv(SUBTYPE_ORDER_PATH, header=None)
except:
    subtype_order = pd.DataFrame([list(range(1, nsubtype+1))])

# 获取协变量列
covariate_names = [col for col in df.columns if col not in ['eid', 'PTID', 'stage', 'subtype']]

# 参数
top_n = 10
alpha = 0.05

# 对所有 subtype 进行 Kruskal-Wallis 检验
results = []
for cov in covariate_names:
    groups = [df[df['subtype'] == s][cov].dropna() for s in range(1, nsubtype + 1)]
    try:
        h_stat, p_value = stats.kruskal(*groups)
        results.append({
            'covariate': cov,
            'h_stat': h_stat,
            'p_value': p_value
        })
    except:
        results.append({
            'covariate': cov,
            'h_stat': np.nan,
            'p_value': np.nan
        })

# 转换为 DataFrame 并进行 FDR 校正
results_df = pd.DataFrame(results)
results_df['p_adjusted'] = multipletests(results_df['p_value'].fillna(1.0), method='fdr_bh')[1]

# 按 H 统计量排序并取前 5
top_vars = results_df.sort_values(by='h_stat', ascending=False).head(top_n)['covariate'].tolist()

# 绘制箱线图在一张图中
fig = plt.figure(figsize=(8, 16), dpi=300, facecolor='white')
fig.subplots_adjust(hspace=0.4, wspace=0.3, left=0.05, bottom=0.05, right=0.95, top=0.9)

for idx, var in enumerate(top_vars, 1):
    ax = fig.add_subplot(5, 2, idx)  # 假设 top_n=5，使用 2x3 布局（最后一格为空）
    
    # 绘制箱线图
    box = ax.violinplot([df[df['subtype'] == s][var].dropna() for s in range(1, nsubtype + 1)],
                        showmedians=True,
                        showextrema=False,
                        showmeans=False)
    # box = ax.boxplot([df[df['subtype'] == s][var].dropna() for s in range(1, nsubtype + 1)],
    #                 patch_artist=True,
    #                 widths=0.6)
    
    # 设置箱体颜色
    # for patch in box['boxes']:
    #     patch.set_facecolor(box_color)
    #     patch.set_edgecolor('black')
    #     patch.set_linewidth(1.0)
    
    # # 设置其他元素（须、边界等）为黑色
    # for element in ['whiskers', 'caps', 'medians']:
    #     plt.setp(box[element], color='black', linewidth=1.0)
    
    # # 设置中位线颜色
    # plt.setp(box['medians'], color='black')
    
    # 设置图形属性
    ax.set_xlabel('Subtype', fontsize=9, labelpad=5, color='black')
    ax.set_ylabel(var, fontsize=9, labelpad=5, color='black')
    h_val = results_df[results_df['covariate'] == var]['h_stat'].values[0]
    p_adj = results_df[results_df['covariate'] == var]['p_adjusted'].values[0]
    ax.set_title(f'{var}\nH-statistic: {h_val:.2f}, Adjusted p: {p_adj:.2e}',
                 fontsize=9, pad=10, color='black')
    
    ax.set_xticks(range(1, nsubtype + 1))
    ax.set_xticklabels([str(s) for s in range(1, nsubtype + 1)], color='black')
    
    # 设置刻度朝内，无网格
    ax.tick_params(direction='in')
    ax.grid(False)

# 添加整体说明
fig.text(0.5, 0.02, f'Top {top_n} covariates with largest Kruskal-Wallis H-statistics across {nsubtype} subtypes',
         ha='center', fontsize=9, color='black')

# 保存图像
plt.savefig('boxplot_top_kruskal.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print(f"Boxplots generated for top {top_n} covariates with largest H-statistics across all subtypes.")