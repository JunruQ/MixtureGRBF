import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

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
nsubtype = 5
exp_name = 'ukb_MixtureGRBF_cv_nsubtype_biom17'
INPUT_TABLE_PATH = 'input/ukb/ukb_covreg1_trans1_nanf1_biom17.csv'
SUBTYPE_STAGE_PATH = f'output/{exp_name}/{nsubtype}_subtypes/subtype_stage.csv'
OUTPUT_DIR = f'output/result_analysis/{exp_name}/{nsubtype}_subtypes'
SUBTYPE_ORDER_PATH = f'output/result_analysis/{exp_name}/{nsubtype}_subtypes/all_cause_mortality_order.csv'

# 定义颜色（为 top_n 个蛋白分配不同颜色）
palette = ['#206491', '#038db2', '#f9637c', '#fe7966', '#fbb45c']  # top_n=5 的颜色
top_n = 5
alpha = 0.05

# Bonferroni 校正函数
def bonferroni_correction(p_values):
    n = len(p_values)
    corrected_p_values = np.minimum(np.array(p_values) * n, 1.0)
    return corrected_p_values

# 读取数据
df = pd.read_csv(INPUT_TABLE_PATH)
biomarker_names = df.iloc[:, 7:].columns.tolist()
df.rename(columns={'RID': 'PTID'}, inplace=True)
subtype_stage = pd.read_csv(SUBTYPE_STAGE_PATH)
df = pd.merge(df, subtype_stage, how='inner', on=['PTID', 'stage'])
df = df.dropna(subset=['subtype'])
df['subtype'] = df['subtype'].astype(int)

# 读取 subtype 顺序
try:
    subtype_order = pd.read_csv(SUBTYPE_ORDER_PATH, header=None)
except:
    subtype_order = pd.DataFrame(np.array(list(range(1, nsubtype+1))))


# 对第 nsubtype 个 subtype 计算差异
# k = int(subtype_order.iloc[nsubtype-1, 0])  # 第 nsubtype 个 subtype
k = 1
case_group = df['subtype'] == k
control_group = df['subtype'] != k

results = []
for biom in biomarker_names:
    case = df.loc[case_group, biom].dropna()
    control = df.loc[control_group, biom].dropna()
    t_value, p_value = stats.ttest_ind(case, control, equal_var=False, nan_policy='omit')
    results.append({
        'biomarker': biom,
        't_stat': t_value,
        'p_value': p_value
    })

# 转换为 DataFrame 并校正 p 值
results_df = pd.DataFrame(results)
results_df['p_adjusted'] = bonferroni_correction(results_df['p_value'])

# 按 t 值绝对值排序，取 top_n
top_proteins = results_df.sort_values(by='t_stat', key=abs, ascending=False).head(top_n)
top_protein_names = top_proteins['biomarker'].tolist()
top_protein_idx = [biomarker_names.index(prot) for prot in top_protein_names]

print(f"Top {top_n} proteins for Subtype {nsubtype} vs rest:")
print(top_proteins[['biomarker', 't_stat', 'p_adjusted']])
print(f"Indices in original data: {top_protein_idx}")

# 绘制轨迹图
fig = plt.figure(figsize=(9.5, 9), dpi=300, facecolor='white')
fig.subplots_adjust(hspace=0.3, wspace=0.2, left=0.05, bottom=0.1, right=0.95, top=0.9)

for i in range(1, nsubtype + 1):
    # 根据 subtype_order 获取对应的 subtype 编号和轨迹文件
    k = int(subtype_order.iloc[i-1, 0])
    TRAJECTORY_PATH = f'output/{exp_name}/{nsubtype}_subtypes/trajectory{k}.csv'
    
    # 读取轨迹数据
    try:
        trajectory_df = pd.read_csv(TRAJECTORY_PATH)  # 假设第一列是 x（index）
    except FileNotFoundError:
        print(f"Warning: {TRAJECTORY_PATH} not found, skipping Subtype {i}.")
        continue
    
    ax = plt.subplot(3, 2, i)
    max_y = -float('inf')  # 初始化为负无穷大
    for protein in top_protein_names:
        if protein in trajectory_df.columns:
            max_y = max(max_y, trajectory_df[protein].abs().max())
    max_y += 0.1 * max_y  # 留出 10% 的空间
    
    # 绘制 top_n 个蛋白的轨迹
    for idx, protein in enumerate(top_protein_names):
        if protein not in trajectory_df.columns:
            print(f"Warning: {protein} not found in trajectory data for Subtype {i}.")
            continue
        
        # ax.plot(trajectory_df.index, trajectory_df[protein], 
        #         color=palette[idx % len(palette)], linewidth=1.5, 
        #         label=f'{protein}' if i == 1 else None)  # 仅在第一个 subplot 加标签
        ax.plot(range(39,71), trajectory_df[protein], 
                color=palette[idx % len(palette)], linewidth=1.5, 
                label=f'{protein}' if i == 1 else None)  # 仅在第一个 subplot 加标签
    
    # 设置图形属性
    ax.set_xlabel('Age', fontsize=9, labelpad=5, color='black')
    ax.set_ylabel('Protein Value', fontsize=9, labelpad=5, color='black')
    ax.set_title(f'Subtype {i}', fontsize=9, pad=10, color='black')
    ax.set_ylim(-max_y, max_y)

    # 在0处绘制参考线，为虚线
    ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
    
    ax.tick_params(direction='in')
    ax.grid(False)
    
    # 在第一个 subplot 添加图例
    if i == 1:
        ax.legend(loc='upper left', bbox_to_anchor=(-0.55, 1.1), frameon=True, framealpha=1, 
                  edgecolor='black', fontsize=8, title='Proteins', title_fontsize=9)

# 添加整体说明
fig.text(0.5, 0.02, f'Trajectories of top {top_n} proteins (Subtype {k} vs rest) across {nsubtype} subtypes',
         ha='center', fontsize=9, color='black')

# 保存图像
plt.savefig(f'{OUTPUT_DIR}/top_{top_n}_protein_trajectories_all_subtypes.png', 
           dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print(f"Trajectory plots saved to {OUTPUT_DIR}/top_{top_n}_protein_trajectories_all_subtypes.png")