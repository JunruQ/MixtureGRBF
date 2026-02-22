import pandas as pd
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib.lines import Line2D
from concurrent.futures import ProcessPoolExecutor

def count_significant_biom_fast(subtype_data, age_min, age_max, window_size=10, alpha=0.05, correction_method='bonf'):
    """使用线性回归进行显著性检验，加入sex作为协变量"""
    import numpy as np
    import statsmodels.api as sm
    
    stage_array = subtype_data['stage']
    biom_arrays = subtype_data['biomarkers']
    sex_array = subtype_data['sex']  # 假设sex数据在subtype_data中
    ages = range(age_min + 2, age_max - 2 + 1)
    counts = []

    def fdr_correction(p_values, alpha=0.05):
        """Benjamini-Hochberg FDR校正"""
        p_values = np.array(p_values)
        n = len(p_values)
        sorted_idx = np.argsort(p_values)
        sorted_p = p_values[sorted_idx]
        
        # 计算FDR临界值
        critical_values = alpha * (np.arange(n) + 1) / n
        
        # 找到最后一个显著的p值位置
        significant = sorted_p <= critical_values
        if np.any(significant):
            max_significant_idx = np.max(np.where(significant))
            threshold = sorted_p[max_significant_idx]
        else:
            threshold = -1  # 如果没有显著结果
        
        # 返回校正后的结果
        return p_values <= threshold

    
    def bonferroni_correction(p_values, alpha=0.05):
        n = len(p_values)
        return np.array(p_values) * n <= alpha
    
    for age in ages:
        lower1, upper1 = age - window_size, age
        lower2, upper2 = age, age + window_size
        
        # 获取窗口内的数据索引
        start1 = np.searchsorted(stage_array, lower1, side='left')
        end1 = np.searchsorted(stage_array, upper1, side='right')
        start2 = np.searchsorted(stage_array, lower2, side='left')
        end2 = np.searchsorted(stage_array, upper2, side='right')
        
        # 合并两个窗口的数据
        indices = np.concatenate([np.arange(start1, end1), np.arange(start2, end2)])
        age_binary = np.concatenate([
            np.zeros(end1 - start1),  # 第一段窗口标记为0
            np.ones(end2 - start2)    # 第二段窗口标记为1
        ])
        sex_data = sex_array[indices]
        
        p_values = []
        for biom in biom_arrays:
            biom_data = biom_arrays[biom][indices]
            mask = ~np.isnan(biom_data)  # 去除NaN值
            if np.sum(mask) > 2:  # 确保有足够的数据点进行回归
                X = np.column_stack([age_binary[mask], sex_data[mask]])
                X = sm.add_constant(X)  # 添加常数项
                y = biom_data[mask]
                
                # 线性回归
                model = sm.OLS(y, X)
                results = model.fit()
                p_value = results.pvalues[1]  # 获取age_binary的回归系数p值
                p_values.append(p_value)
            else:
                p_values.append(1.0)  # 数据不足时设为非显著
        
        # 使用Bonferroni校正
        if correction_method == 'bonf':
            significant = bonferroni_correction(p_values, alpha)
        elif correction_method == 'fdr':
            significant = fdr_correction(p_values, alpha)
        counts.append(np.sum(significant))
    
    return counts, ages

# 数据准备和预处理
nsubtype = 5
matlab_colors = ['#0072BD', '#D95319', '#EDB120', '#7E2F8E', '#77AC30']

exp_name = 'ukb_MixtureGRBF_cv_nsubtype_biom17'
SUBTYPE_ORDER_PATH = f'output/result_analysis/{exp_name}/{nsubtype}_subtypes/all_cause_mortality_order.csv'
OUTPUT_DIR = f'output/result_analysis/{exp_name}/{nsubtype}_subtypes'
try:
    subtype_order = pd.read_csv(SUBTYPE_ORDER_PATH, header=None)
except:
    subtype_order = pd.DataFrame(np.array(list(range(1, nsubtype+1))))
df = pd.read_csv('input/ukb/ukb_covreg1_trans1_nanf1_biom0.csv').rename(columns={'RID':'PTID'})
subtype_stage = pd.read_csv(f'output/{exp_name}/{nsubtype}_subtypes/subtype_stage.csv')
subtype_order_map = {int(subtype_order.iloc[i]): i+1 for i in range(nsubtype)}
subtype_stage['subtype'] = subtype_stage['subtype'].map(subtype_order_map)
df = pd.merge(df, subtype_stage[['PTID', 'subtype']], how='inner', on=['PTID'])

# 预处理每个亚型的数据
subtype_dfs = {}
for i in range(1, nsubtype+1):
    df_sub = df[df['subtype'] == i].sort_values(by='stage')
    subtype_dfs[i] = {
        'stage': df_sub['stage'].values,
        'biomarkers': {biom: df_sub[biom].values for biom in df.columns[7:-1]},
        'sex': df_sub['sex'].values  # 假设数据集中有'sex'列
    }

age_min, age_max = int(df['stage'].min()), int(df['stage'].max())


# 使用多进程并行计算
def compute_counts(args):
    i, window, correction_method = args
    counts, ages = count_significant_biom_fast(subtype_dfs[i], age_min, age_max, window_size=window, correction_method=correction_method)
    return (i, window, counts, ages)

# windows = [2, 3, 4, 5, 10]
windows = [5, 10]
counts_results = {w: {} for w in windows}


correction_method = 'bonf'
with ProcessPoolExecutor() as executor:
    args_list = [(i, w, correction_method) for i in range(1, nsubtype+1) for w in windows]
    for result in executor.map(compute_counts, args_list):
        i, window, counts, ages = result
        counts_results[window][i] = (counts, ages)

# # 可视化：所有subtype放在一张图中
# fig, ax = plt.subplots(figsize=(8, 6), dpi=300, facecolor='white')
# linestyles = {2: '-', 3: '--', 4: ':', 5: '-.', 10:'-'}

# # 绘制每个subtype的显著性计数
# for i in range(1, nsubtype+1):
#     color = matlab_colors[i-1]
#     for window in windows:
#         counts, ages = counts_results[window][i]
#         ax.plot(ages, np.array(counts) + 1,
#                 color=color,
#                 linestyle=linestyles[window],
#                 linewidth=1.5,
#                 label=f'Subtype {i}, Window={window}')

# ax.set_xlabel('Age')
# ax.set_ylabel('# Significant Biomarkers')
# ax.set_title('Significant Biomarkers Comparison')
# ax.grid(True, linestyle='--', alpha=0.7)
# ax.legend(fontsize=8)

# # 调整布局并保存
# plt.tight_layout()
# plt.savefig(OUTPUT_DIR + '/significant_protein_peak_linear_regression_10.png', dpi=300, bbox_inches='tight')
# plt.close()
# Visualization: Subplots with 2 columns
n_windows = len(windows)
n_cols = 2
n_rows = (n_windows + 1) // 2  # Ceiling division to get number of rows

fig, axes = plt.subplots(n_rows, n_cols, figsize=(28/2.54, 4*n_rows), dpi=300, facecolor='white')
axes = axes.flatten()  # Flatten 2D array of axes for easier indexing

# Plot each window size in its own subplot
for idx, window in enumerate(windows):
    ax = axes[idx]
    for i in range(1, nsubtype+1):
        color = matlab_colors[i-1]
        counts, ages = counts_results[window][i]
        ax.plot(ages, np.array(counts) + 1,
                color=color,
                linewidth=1.5,
                label=f'Subtype {i}')
    
    ax.set_xlabel('Age')
    ax.set_ylabel('# Significant Biomarkers')
    ax.set_title(f'Window Size = {window}')
    ax.grid(True, linestyle='--', alpha=0.7)
    if idx == 0:  # Only add legend to first subplot
        ax.legend(fontsize=8)

# Remove empty subplots if any
for idx in range(n_windows, len(axes)):
    fig.delaxes(axes[idx])

# Add Title
fig.suptitle('Significant Biomarkers Comparison', fontsize=12)
# Adjust layout and save
plt.tight_layout()
plt.savefig(OUTPUT_DIR + f'/significant_protein_peak_linear_regression_{correction_method}.png', 
            dpi=300, bbox_inches='tight')
plt.close()