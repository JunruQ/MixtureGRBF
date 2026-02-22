# import pandas as pd
# import statsmodels.api as sm
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy import stats
# from matplotlib.lines import Line2D
# from concurrent.futures import ProcessPoolExecutor

# def bonferroni_correction(p_values):
#     n = len(p_values)
#     corrected_p_values = np.array(p_values) * n
#     return np.minimum(corrected_p_values, 1.0)


# # def count_significant_biom_fast(subtype_data, age_min, age_max, window_size=10, alpha=0.05):
# #     """优化后的快速版本，使用预处理数据和numpy进行加速"""
# #     stage_array = subtype_data['stage']
# #     biom_arrays = subtype_data['biomarkers']
# #     ages = range(age_min + window_size, age_max - window_size + 1)
# #     counts = []
    
# #     for age in ages:
# #         lower1, upper1 = age - window_size, age
# #         lower2, upper2 = age, age + window_size
        
# #         start1 = np.searchsorted(stage_array, lower1, side='left')
# #         end1 = np.searchsorted(stage_array, upper1, side='right')
# #         start2 = np.searchsorted(stage_array, lower2, side='left')
# #         end2 = np.searchsorted(stage_array, upper2, side='right')
        
# #         p_values = []
# #         for biom in biom_arrays:
# #             data1 = biom_arrays[biom][start1:end1]
# #             data1 = data1[~np.isnan(data1)]
# #             data2 = biom_arrays[biom][start2:end2]
# #             data2 = data2[~np.isnan(data2)]
            
# #             if len(data1) > 1 and len(data2) > 1:
# #                 _, p = stats.ttest_ind(data1, data2, equal_var=False)
# #                 p_values.append(p)
# #             else:
# #                 p_values.append(1.0)
        
# #         corrected_p = bonferroni_correction(p_values)
# #         counts.append(np.sum(corrected_p <= alpha))
    
# #     return counts, ages

# def count_significant_biom_fast(subtype_data, age_min, age_max, window_size=10, alpha=0.05):
#     """优化后的快速版本，使用预处理数据和numpy进行加速"""
#     import numpy as np
#     from scipy import stats
    
#     stage_array = subtype_data['stage']
#     biom_arrays = subtype_data['biomarkers']
#     ages = range(age_min + window_size, age_max - window_size + 1)
#     counts = []
    
#     def fdr_correction(p_values, alpha=0.05):
#         """Benjamini-Hochberg FDR校正"""
#         p_values = np.array(p_values)
#         n = len(p_values)
#         sorted_idx = np.argsort(p_values)
#         sorted_p = p_values[sorted_idx]
        
#         # 计算FDR临界值
#         critical_values = alpha * (np.arange(n) + 1) / n
        
#         # 找到最后一个显著的p值位置
#         significant = sorted_p <= critical_values
#         if np.any(significant):
#             max_significant_idx = np.max(np.where(significant))
#             threshold = sorted_p[max_significant_idx]
#         else:
#             threshold = -1  # 如果没有显著结果
        
#         # 返回校正后的结果
#         return p_values <= threshold

#     def bonferroni_correction(p_values, alpha=0.05):
#         n = len(p_values)
#         return np.array(p_values) * n <= alpha
    
#     for age in ages:
#         lower1, upper1 = age - window_size, age
#         lower2, upper2 = age, age + window_size
        
#         start1 = np.searchsorted(stage_array, lower1, side='left')
#         end1 = np.searchsorted(stage_array, upper1, side='right')
#         start2 = np.searchsorted(stage_array, lower2, side='left')
#         end2 = np.searchsorted(stage_array, upper2, side='right')
        
#         p_values = []
#         for biom in biom_arrays:
#             data1 = biom_arrays[biom][start1:end1]
#             data1 = data1[~np.isnan(data1)]
#             data2 = biom_arrays[biom][start2:end2]
#             data2 = data2[~np.isnan(data2)]
            
#             if len(data1) > 1 and len(data2) > 1:
#                 _, p = stats.ttest_ind(data1, data2, equal_var=False)
#                 p_values.append(p)
#             else:
#                 p_values.append(1.0)
        
#         # 使用FDR校正替代Bonferroni
#         # significant = fdr_correction(p_values, alpha)
#         significant = bonferroni_correction(p_values, alpha)
#         counts.append(np.sum(significant))
    
#     return counts, ages

# # 计算一阶导数并返回L2-norm
# def compute_trajectory_gradient(trajectory_df):
#     traj_array = trajectory_df.values  # ages * dimensions
#     # 使用中心差分计算梯度: (y[i+1] - y[i-1]) / 2
#     gradient = (traj_array[2:] - traj_array[:-2]) / 2
#     # 计算L2-norm
#     # l2_norm = np.sum(np.abs(gradient), axis=1)
#     l2_norm = np.sqrt(np.sum(gradient**2, axis=1))
#     # 对应的年龄范围（去掉首尾各一个点）
#     ages = np.arange(traj_array.shape[0])[1:-1]
#     return l2_norm, ages

# # 数据准备和预处理
# nsubtype = 5
# matlab_colors = ['#0072BD', '#D95319', '#EDB120', '#7E2F8E', '#77AC30']

# exp_name = 'ukb_MixtureGRBF_cv_nsubtype_biom17'
# SUBTYPE_ORDER_PATH = f'output/result_analysis/{exp_name}/{nsubtype}_subtypes/all_cause_mortality_order.csv'
# OUTPUT_DIR = f'output/result_analysis/{exp_name}/{nsubtype}_subtypes'
# try:
#     subtype_order = pd.read_csv(SUBTYPE_ORDER_PATH, header=None)
# except:
#     subtype_order = pd.DataFrame(np.array(list(range(1, nsubtype+1))))
# df = pd.read_csv('input/ukb/ukb_covreg1_trans3_nanf1_biom0.csv').rename(columns={'RID':'PTID'})
# subtype_stage = pd.read_csv(f'output/{exp_name}/{nsubtype}_subtypes/subtype_stage.csv')
# subtype_order_map = {int(subtype_order.iloc[i]): i+1 for i in range(nsubtype)}
# subtype_stage['subtype'] = subtype_stage['subtype'].map(subtype_order_map)
# df = pd.merge(df, subtype_stage, how='inner', on=['PTID', 'stage'])

# # 预处理每个亚型的数据
# subtype_dfs = {}
# for i in range(1, nsubtype+1):
#     df_sub = df[df['subtype'] == i].sort_values(by='stage')
#     subtype_dfs[i] = {
#         'stage': df_sub['stage'].values,
#         'biomarkers': {biom: df_sub[biom].values for biom in df.columns[7:-1]}
#     }

# age_min, age_max = int(df['stage'].min()), int(df['stage'].max())

# # 加载并计算每个subtype的trajectory梯度
# trajectory_gradients = {}
# for k in range(1, nsubtype+1):
#     TRAJECTORY_PATH = f'output/{exp_name}/{nsubtype}_subtypes/trajectory{k}.csv'
#     trajectory_df = pd.read_csv(TRAJECTORY_PATH)
#     l2_norm, traj_ages = compute_trajectory_gradient(trajectory_df)
#     trajectory_gradients[k] = (l2_norm, traj_ages)

# # 使用多进程并行计算
# def compute_counts(args):
#     i, window = args
#     counts, ages = count_significant_biom_fast(subtype_dfs[i], age_min, age_max, window_size=window)
#     return (i, window, counts, ages)

# # windows = [2, 3, 4, 5]
# windows = [5]
# counts_results = {w: {} for w in windows}
# with ProcessPoolExecutor() as executor:
#     args_list = [(i, w) for i in range(1, nsubtype+1) for w in windows]
#     for result in executor.map(compute_counts, args_list):
#         i, window, counts, ages = result
#         counts_results[window][i] = (counts, ages)

# # 可视化：4个子图
# fig = plt.figure(figsize=(12, 12), dpi=300, facecolor='white')
# linestyles = {2: '-', 3: '--', 4: ':', 5: '-.'}

# # 绘制每个subtype的子图
# for i in range(1, nsubtype+1):
#     ax = plt.subplot((nsubtype + 1) // 2 , 2, i)
#     color = matlab_colors[i-1]
    
#     # 绘制显著性计数
#     for window in windows:
#         counts, ages = counts_results[window][i]
#         ax.plot(ages, np.array(counts) + 1, 
#                 color=color,
#                 linestyle=linestyles[window],
#                 linewidth=1.5,
#                 label=f'Window={window}')
    
#     # # 添加trajectory梯度的L2-norm（使用次坐标轴）
#     # ax2 = ax.twinx()
#     # l2_norm, traj_ages = trajectory_gradients[i]
#     # ax2.plot(np.array(range(age_min, age_max+1))[traj_ages], l2_norm, 
#     #          color='black', 
#     #          linestyle='-.', 
#     #          linewidth=1.5, 
#     #          label='Traj Gradient L2')
#     # ax2.set_ylabel('Gradient L2-norm', fontsize=9, color='black')
#     # ax2.tick_params(axis='y', labelcolor='black')

#     # # ax.set_yscale('log')
#     # ax.set_title(f'Subtype {i}', fontsize=10)
#     # ax.set_xlabel('Age', fontsize=9)
#     # ax.set_ylabel('# Significant Biomarkers', fontsize=9)
#     # ax.grid(True, linestyle='--', alpha=0.3)

# # 创建自定义图例
# legend_elements = [
#     *[Line2D([0], [0], color=c, lw=2) for c in matlab_colors],
#     *[Line2D([0], [0], color='k', linestyle=linestyles[w], lw=1.5) for w in windows],
#     Line2D([0], [0], color='k', linestyle='-.', lw=1.5)
# ]
# legend_labels = (
#     [f'Subtype {i}' for i in range(1, nsubtype+1)] +
#     [f'Window={w}' for w in windows] +
#     ['Traj Gradient L2']
# )

# # 将图例放在图表下方
# fig.legend(legend_elements, legend_labels, 
#            loc='upper center', 
#            bbox_to_anchor=(0.5, 0.0),
#            ncol=nsubtype+1,  # 调整列数以适应新增的图例项
#            fontsize=8)

# # 调整布局
# plt.suptitle('Significant Biomarkers and Trajectory Gradient Comparison', fontsize=12, y=1.02)
# plt.tight_layout()
# plt.savefig(OUTPUT_DIR+'/significant_protein_peak.png', dpi=300, bbox_inches='tight')
# plt.close()

# import pandas as pd
# import statsmodels.api as sm
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy import stats
# from matplotlib.lines import Line2D
# from concurrent.futures import ProcessPoolExecutor

# def bonferroni_correction(p_values):
#     n = len(p_values)
#     corrected_p_values = np.array(p_values) * n
#     return np.minimum(corrected_p_values, 1.0)

# def count_significant_biom_fast(subtype_data, age_min, age_max, window_size=10, alpha=0.05):
#     """优化后的快速版本，使用预处理数据和numpy进行加速"""
#     import numpy as np
#     from scipy import stats
    
#     stage_array = subtype_data['stage']
#     biom_arrays = subtype_data['biomarkers']
#     ages = range(age_min, age_max + 1)
#     counts = []
    
#     def fdr_correction(p_values, alpha=0.05):
#         """Benjamini-Hochberg FDR校正"""
#         p_values = np.array(p_values)
#         n = len(p_values)
#         sorted_idx = np.argsort(p_values)
#         sorted_p = p_values[sorted_idx]
        
#         # 计算FDR临界值
#         critical_values = alpha * (np.arange(n) + 1) / n
        
#         # 找到最后一个显著的p值位置
#         significant = sorted_p <= critical_values
#         if np.any(significant):
#             max_significant_idx = np.max(np.where(significant))
#             threshold = sorted_p[max_significant_idx]
#         else:
#             threshold = -1  # 如果没有显著结果
        
#         # 返回校正后的结果
#         return p_values <= threshold

#     def bonferroni_correction(p_values, alpha=0.05):
#         n = len(p_values)
#         return np.array(p_values) * n <= alpha
    
#     for age in ages:
#         lower1, upper1 = age - window_size, age
#         lower2, upper2 = age, age + window_size
        
#         start1 = np.searchsorted(stage_array, lower1, side='left')
#         end1 = np.searchsorted(stage_array, upper1, side='right')
#         start2 = np.searchsorted(stage_array, lower2, side='left')
#         end2 = np.searchsorted(stage_array, upper2, side='right')
        
#         p_values = []
#         for biom in biom_arrays:
#             data1 = biom_arrays[biom][start1:end1]
#             data1 = data1[~np.isnan(data1)]
#             data2 = biom_arrays[biom][start2:end2]
#             data2 = data2[~np.isnan(data2)]
            
#             if len(data1) > 1 and len(data2) > 1:
#                 _, p = stats.ttest_ind(data1, data2, equal_var=False)
#                 p_values.append(p)
#             else:
#                 p_values.append(1.0)
        
#         # 使用FDR校正替代Bonferroni
#         # significant = fdr_correction(p_values, alpha)
#         significant = bonferroni_correction(p_values, alpha)
#         counts.append(np.sum(significant))
    
#     return counts, ages

# # 计算一阶导数并返回L2-norm
# def compute_trajectory_gradient(trajectory_df):
#     traj_array = trajectory_df.values  # ages * dimensions
#     # 使用中心差分计算梯度: (y[i+1] - y[i-1]) / 2
#     gradient = (traj_array[2:] - traj_array[:-2]) / 2
#     # 计算L2-norm
#     # l2_norm = np.sum(np.abs(gradient), axis=1)
#     l2_norm = np.sqrt(np.sum(gradient**2, axis=1))
#     # 对应的年龄范围（去掉首尾各一个点）
#     ages = np.arange(traj_array.shape[0])[1:-1]
#     return l2_norm, ages

# # 数据准备和预处理
# nsubtype = 5
# matlab_colors = ['#0072BD', '#D95319', '#EDB120', '#7E2F8E', '#77AC30']

# exp_name = 'ukb_MixtureGRBF_cv_nsubtype_biom17'
# SUBTYPE_ORDER_PATH = f'output/result_analysis/{exp_name}/{nsubtype}_subtypes/all_cause_mortality_order.csv'
# OUTPUT_DIR = f'output/result_analysis/{exp_name}/{nsubtype}_subtypes'
# try:
#     subtype_order = pd.read_csv(SUBTYPE_ORDER_PATH, header=None)
# except:
#     subtype_order = pd.DataFrame(np.array(list(range(1, nsubtype+1))))
# df = pd.read_csv('input/ukb/ukb_covreg1_trans3_nanf1_biom0.csv').rename(columns={'RID':'PTID'})
# subtype_stage = pd.read_csv(f'output/{exp_name}/{nsubtype}_subtypes/subtype_stage.csv')
# subtype_order_map = {int(subtype_order.iloc[i]): i+1 for i in range(nsubtype)}
# subtype_stage['subtype'] = subtype_stage['subtype'].map(subtype_order_map)
# df = pd.merge(df, subtype_stage[['PTID', 'subtype']], how='inner', on=['PTID'])

# # 预处理每个亚型的数据
# subtype_dfs = {}
# for i in range(1, nsubtype+1):
#     df_sub = df[df['subtype'] == i].sort_values(by='stage')
#     subtype_dfs[i] = {
#         'stage': df_sub['stage'].values,
#         'biomarkers': {biom: df_sub[biom].values for biom in df.columns[7:-1]}
#     }

# age_min, age_max = int(df['stage'].min()), int(df['stage'].max())

# # 加载并计算每个subtype的trajectory梯度
# trajectory_gradients = {}
# for k in range(1, nsubtype+1):
#     TRAJECTORY_PATH = f'output/{exp_name}/{nsubtype}_subtypes/trajectory{k}.csv'
#     trajectory_df = pd.read_csv(TRAJECTORY_PATH)
#     l2_norm, traj_ages = compute_trajectory_gradient(trajectory_df)
#     trajectory_gradients[k] = (l2_norm, traj_ages)

# # 使用多进程并行计算
# def compute_counts(args):
#     i, window = args
#     counts, ages = count_significant_biom_fast(subtype_dfs[i], age_min, age_max, window_size=window)
#     return (i, window, counts, ages)

# # windows = [2, 3, 4, 5]
# windows = [10]
# counts_results = {w: {} for w in windows}
# with ProcessPoolExecutor() as executor:
#     args_list = [(i, w) for i in range(1, nsubtype+1) for w in windows]
#     for result in executor.map(compute_counts, args_list):
#         i, window, counts, ages = result
#         counts_results[window][i] = (counts, ages)

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
# plt.savefig(OUTPUT_DIR + '/significant_protein_peak_combined.png', dpi=300, bbox_inches='tight')
# plt.close()


import pandas as pd
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib.lines import Line2D
from concurrent.futures import ProcessPoolExecutor

def bonferroni_correction(p_values):
    n = len(p_values)
    corrected_p_values = np.array(p_values) * n
    return np.minimum(corrected_p_values, 1.0)

def count_significant_biom_fast(subtype_data, age_min, age_max, window_size=10, alpha=0.05):
    """优化后的快速版本，使用预处理数据和numpy进行加速"""
    import numpy as np
    from scipy import stats
    
    stage_array = subtype_data['stage']
    biom_arrays = subtype_data['biomarkers']
    ages = range(age_min + 2, age_max - 2 + 1)
    counts = []
    
    def fdr_correction(p_values, alpha=0.05):
        """Benjamini-Hochberg FDR校正"""
        p_values = np.array(p_values)
        n = len(p_values)
        sorted_idx = np.argsort(p_values)
        sorted_p = p_values[sorted_idx]
        
        critical_values = alpha * (np.arange(n) + 1) / n
        
        significant = sorted_p <= critical_values
        if np.any(significant):
            max_significant_idx = np.max(np.where(significant))
            threshold = sorted_p[max_significant_idx]
        else:
            threshold = -1
        
        return p_values <= threshold

    def bonferroni_correction(p_values, alpha=0.05):
        n = len(p_values)
        return np.array(p_values) * n <= alpha
    
    for age in ages:
        lower1, upper1 = age - window_size, age
        lower2, upper2 = age, age + window_size
        
        start1 = np.searchsorted(stage_array, lower1, side='left')
        end1 = np.searchsorted(stage_array, upper1, side='right')
        start2 = np.searchsorted(stage_array, lower2, side='left')
        end2 = np.searchsorted(stage_array, upper2, side='right')
        
        p_values = []
        for biom in biom_arrays:
            data1 = biom_arrays[biom][start1:end1]
            data1 = data1[~np.isnan(data1)]
            data2 = biom_arrays[biom][start2:end2]
            data2 = data2[~np.isnan(data2)]
            
            if len(data1) > 1 and len(data2) > 1:
                _, p = stats.ttest_ind(data1, data2, equal_var=False)
                p_values.append(p)
            else:
                p_values.append(1.0)
        
        significant = fdr_correction(p_values, alpha)
        counts.append(np.sum(significant))
    
    return counts, ages

# Data preparation and preprocessing
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

subtype_dfs = {}
for i in range(1, nsubtype+1):
    df_sub = df[df['subtype'] == i].sort_values(by='stage')
    subtype_dfs[i] = {
        'stage': df_sub['stage'].values,
        'biomarkers': {biom: df_sub[biom].values for biom in df.columns[7:-1]}
    }

age_min, age_max = int(df['stage'].min()), int(df['stage'].max())

def compute_counts(args):
    i, window = args
    counts, ages = count_significant_biom_fast(subtype_dfs[i], age_min, age_max, window_size=window)
    return (i, window, counts, ages)

windows = [2, 3, 4, 5, 10]  # Multiple window sizes
counts_results = {w: {} for w in windows}
with ProcessPoolExecutor() as executor:
    args_list = [(i, w) for i in range(1, nsubtype+1) for w in windows]
    for result in executor.map(compute_counts, args_list):
        i, window, counts, ages = result
        counts_results[window][i] = (counts, ages)

# Visualization: Subplots with 2 columns
n_windows = len(windows)
n_cols = 2
n_rows = (n_windows + 1) // 2

fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4*n_rows), dpi=300, facecolor='white')
axes = axes.flatten()

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
    if idx == 0:
        ax.legend(fontsize=8)

for idx in range(n_windows, len(axes)):
    fig.delaxes(axes[idx])

plt.tight_layout()
plt.savefig(OUTPUT_DIR + '/significant_protein_peak_ttest_subplots.png', 
            dpi=300, bbox_inches='tight')
plt.close()