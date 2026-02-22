import pandas as pd
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib.lines import Line2D
from concurrent.futures import ProcessPoolExecutor
import tqdm

def perform_regression_batch(biom_matrix, age_binary):
    """
    批量执行线性回归，返回 age_binary 的系数、p 值和 t 值
    biom_matrix: shape = (n_samples, n_biomarkers)
    age_binary: shape = (n_samples,)
    """
    n_samples, n_biom = biom_matrix.shape
    
    # 构建设计矩阵 X = [1, age_binary]
    X = np.column_stack([
        np.ones(n_samples),
        age_binary,
    ])
    
    XtX_inv = np.linalg.pinv(X.T @ X)
    betas = XtX_inv @ X.T @ biom_matrix  # shape = (3, n_biomarkers)
    
    # 残差
    y_hat = X @ betas
    residuals = biom_matrix - y_hat
    dof = n_samples - X.shape[1]
    mse = np.sum(residuals**2, axis=0) / dof
    
    # 标准误
    se_betas = np.sqrt(np.diag(XtX_inv)[:, None] * mse)  # shape = (3, n_biomarkers)
    
    # t统计量 & p值（只针对 age_binary，即 betas[1]）
    t_stats = betas[1] / se_betas[1]
    p_values = 2 * stats.t.sf(np.abs(t_stats), dof)
    
    return betas[1], p_values, t_stats

def get_window_indices(stage_array, lk, window_size):
    """根据DEswan方法获取窗口 [lk - x/2, lk) 和 (lk, lk + x/2] 的索引"""
    half_window = window_size / 2
    lower1, upper1 = lk - half_window, lk
    lower2, upper2 = lk, lk + half_window
    
    # 获取窗口内的数据索引（半开半闭区间）
    start1 = np.searchsorted(stage_array, lower1, side='left')
    end1 = np.searchsorted(stage_array, upper1, side='left')  # [lk - x/2, lk)
    start2 = np.searchsorted(stage_array, lower2, side='right')  # (lk, lk + x/2]
    end2 = np.searchsorted(stage_array, upper2, side='right')
    
    return start1, end1, start2, end2

def fdr_correction(p_values, alpha=0.05):
    """Benjamini-Hochberg FDR校正，返回显著性标志和q值"""
    p_values = np.array(p_values)
    n = len(p_values)
    sorted_idx = np.argsort(p_values)
    sorted_p = p_values[sorted_idx]
    
    # 计算FDR临界值
    critical_values = alpha * (np.arange(n) + 1) / n
    significant = sorted_p <= critical_values
    
    # 计算q值
    q_values = sorted_p * n / (np.arange(n) + 1)
    q_values = np.minimum.accumulate(q_values[::-1])[::-1]  # 确保单调性
    
    # 恢复原始顺序
    q_values_full = np.zeros_like(p_values)
    q_values_full[sorted_idx] = q_values
    significant_full = np.zeros_like(p_values, dtype=bool)
    significant_full[sorted_idx] = significant
    
    return significant_full, q_values_full

def bonferroni_correction(p_values, alpha=0.05):
    """Bonferroni校正"""
    n = len(p_values)
    return np.array(p_values) * n <= alpha, np.array(p_values) * n

def count_significant_biom_fast(subtype_data, subtype_id, age_min, age_max, window_size=10, alpha=0.05):
    """返回 DataFrame，包含 Biomarker, Subtype, Age, Coefficient, p, t"""
    stage_array = subtype_data['stage']
    biom_arrays = subtype_data['biomarkers']
    biom_names = list(biom_arrays.keys())
    biom_matrix_full = np.column_stack([biom_arrays[biom] for biom in biom_names])
    
    ages = np.arange(age_min + window_size/2, age_max - window_size/2 + 1, 1)
    
    records = []
    
    for lk in ages:
        start1, end1, start2, end2 = get_window_indices(stage_array, lk, window_size)
        indices = np.concatenate([np.arange(start1, end1), np.arange(start2, end2)])
        if len(indices) <= 2:
            continue
        
        age_binary = np.concatenate([
            np.zeros(end1 - start1),
            np.ones(end2 - start2)
        ])
        biom_matrix = biom_matrix_full[indices, :]
        
        # 回归
        coeffs, p_values, t_stats = perform_regression_batch(biom_matrix, age_binary)
        
        significant, q_values = bonferroni_correction(p_values, alpha)
        
        # 记录显著的 biomarker
        for biom, coef, p, t, sig in zip(biom_names, coeffs, p_values, t_stats, significant):
            if True:
            # if sig:  # 只保存显著的
                records.append({
                    "Biomarker": biom,
                    "Subtype": subtype_id,
                    "Age": lk,
                    "Coefficient": coef,
                    "p": p,
                    "t": t
                })
    
    return pd.DataFrame(records)

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
    }

age_min, age_max = int(df['stage'].min()), int(df['stage'].max())

results_all = []

for i in range(1, nsubtype+1):
    df_res = count_significant_biom_fast(
        subtype_dfs[i], i, age_min, age_max,
        window_size=10
    )
    df_res = df_res.sort_values(by="t", ascending=False).reset_index(drop=True)
    results_all.append(df_res)

final_df = pd.concat(results_all, ignore_index=True)

# 只保留 age=49 和 64
final_df = final_df[final_df["Age"].isin([49, 64])]

# 保存 CSV
output_csv = f"{OUTPUT_DIR}/t_stats_protein_49_64.csv"
final_df.to_csv(output_csv, index=False)

print(f"已保存结果到 {output_csv}")


# # 使用多进程并行计算
# def compute_counts(args):
#     i, window, correction_method = args
#     counts, ages, _ = count_significant_biom_fast(subtype_dfs[i], age_min, age_max, window_size=window, correction_method=correction_method)
#     return (i, window, counts, ages)

# # windows = [2, 3, 4, 5, 10]
# windows = [10]
# counts_results = {w: {} for w in windows}


# correction_method = 'bonf'
# with ProcessPoolExecutor() as executor:
#     args_list = [(i, w, correction_method) for i in range(1, nsubtype+1) for w in windows]
#     for result in executor.map(compute_counts, args_list):
#         i, window, counts, ages = result
#         counts_results[window][i] = (counts, ages)

# # 使用ggplot风格绘制单一图表
# fig, ax = plt.subplots(figsize=(8, 6), dpi=300, facecolor='white')

# # 绘制每个subtype和window size
# for i in range(1, nsubtype + 1):
#     color = matlab_colors[i - 1]
#     for window in windows:
#         counts, ages = counts_results[window][i]
#         ax.plot(ages, np.array(counts) + 1,
#                 color=color,
#                 linewidth=1.5,
#                 label=f'Subtype {i}')

# ax.set_xlabel('Age')
# ax.set_ylabel('# Significant Biomarkers')
# ax.set_title('Significant Biomarkers Comparison (nsubtype=5)')
# ax.grid(True, linestyle='--', alpha=0.7)
# ax.legend(fontsize=8)

# plt.tight_layout()
# plt.savefig(f'{OUTPUT_DIR}/significant_protein_peak_linear_regression_{correction_method}_ggplot.png', dpi=300, bbox_inches='tight')
# plt.close()