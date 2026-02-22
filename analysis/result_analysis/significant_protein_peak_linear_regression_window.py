import pandas as pd
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib.lines import Line2D
from concurrent.futures import ProcessPoolExecutor
from statsmodels.stats.multitest import multipletests
import tqdm
import utils.utils as utils

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.labelsize'] = 11  # Axis labels: 12pt
plt.rcParams['axes.titlesize'] = 11  # Axis titles: 12pt
plt.rcParams['xtick.labelsize'] = 11 # X-axis tick labels: 12pt
plt.rcParams['ytick.labelsize'] = 11  # Y-axis tick labels: 12pt
plt.rcParams['axes.labelcolor'] = 'black'
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['axes.linewidth'] = 1.0

def perform_regression_batch(biom_matrix, age_binary):
    """
    批量执行线性回归，返回所有 biomarker 对应的 age_binary 的 p 值
    biom_matrix: shape = (n_samples, n_biomarkers)
    age_binary: shape = (n_samples,)
    """
    n_samples, n_biom = biom_matrix.shape
    
    X = np.column_stack([
        np.ones(n_samples),
        age_binary,
    ])
    
    # (X^T X)^(-1) X^T
    XtX_inv = np.linalg.pinv(X.T @ X)
    betas = XtX_inv @ X.T @ biom_matrix  # shape = (3, n_biomarkers)
    
    # 残差
    y_hat = X @ betas
    residuals = biom_matrix - y_hat
    dof = n_samples - X.shape[1]
    mse = np.sum(residuals**2, axis=0) / dof  # 每个 biomarker 的均方误差
    
    # beta 的标准误差
    se_betas = np.sqrt(np.diag(XtX_inv)[:, None] * mse)  # shape = (3, n_biomarkers)
    
    # t统计量：age_binary 对应的是 betas[1]
    t_stats = betas[1] / se_betas[1]
    
    # 双边 p 值
    p_values = 2 * stats.t.sf(np.abs(t_stats), dof)
    
    return p_values

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

def count_significant_biom_fast(subtype_data, age_min, age_max, window_size=10, alpha=0.05, correction_method='fdr'):
    """使用批量矩阵回归进行显著性检验，符合DEswan方法"""
    stage_array = subtype_data['stage']
    biom_arrays = subtype_data['biomarkers']
    biom_names = list(biom_arrays.keys())
    biom_matrix_full = np.column_stack([biom_arrays[biom] for biom in biom_names])

    ages = np.arange(age_min + window_size/2, age_max - window_size/2 + 1, 1)
    counts = []
    all_q_values = []
    
    for lk in tqdm.tqdm(ages):
        start1, end1, start2, end2 = get_window_indices(stage_array, lk, window_size)
        indices = np.concatenate([np.arange(start1, end1), np.arange(start2, end2)])
        if len(indices) <= 2:
            counts.append(0)
            all_q_values.append(np.ones(len(biom_names)))
            continue
        
        age_binary = np.concatenate([
            np.zeros(end1 - start1),
            np.ones(end2 - start2)
        ])
        biom_matrix = biom_matrix_full[indices, :]
        
        # 批量回归
        p_values = perform_regression_batch(biom_matrix, age_binary)
        print(f"# of tests: {len(p_values)}")

        # 多重检验校正
        
        if correction_method == 'fdr':
            # significant, q_values = fdr_correction(p_values, alpha)
            significant, q_values = multipletests(p_values, alpha=alpha, method='fdr_bh')[:2]
        elif correction_method == 'bonf':
            # significant, q_values = bonferroni_correction(p_values, alpha)
            significant, q_values = multipletests(p_values, alpha=alpha, method='bonferroni')[:2]
        else:
            raise ValueError("Unsupported correction method. Use 'fdr' or 'bonf'.")
        
        counts.append(np.sum(significant))
        all_q_values.append(q_values)
    
    return counts, ages, all_q_values

# 数据准备和预处理
nsubtype = 5
# matlab_colors = ['#0072BD', '#D95319', '#EDB120', '#7E2F8E', '#77AC30']
matlab_colors = utils.subtype_colors + ['#000000']

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

df_sorted = df.copy().sort_values(by='stage')
subtype_all_df = {
    'stage': df_sorted['stage'].values,
    'biomarkers': {biom: df_sorted[biom].values for biom in df.columns[7:-1]},
}

subtype_dfs[6] = subtype_all_df 

# 使用多进程并行计算
def compute_counts(args):
    i, window, correction_method = args
    counts, ages, _ = count_significant_biom_fast(subtype_dfs[i], age_min, age_max, window_size=window, correction_method=correction_method)
    return (i, window, counts, ages)

windows = [4, 6, 8, 10, 12]
# windows = [10]
counts_results = {w: {} for w in windows}


correction_method = 'bonf'
with ProcessPoolExecutor() as executor:
    args_list = [(i, w, correction_method) for i in [1,2,3,4,5,6] for w in windows]
    for result in executor.map(compute_counts, args_list):
        i, window, counts, ages = result
        counts_results[window][i] = (counts, ages)

for window in windows:
    # 使用ggplot风格绘制单一图表
    fig, ax = plt.subplots(figsize=(11/2.54, 8/2.54), dpi=300, facecolor='white')

    # 绘制每个subtype和window size
    for i in range(1, nsubtype + 2):
        color = matlab_colors[i - 1]
        counts, ages = counts_results[window][i]
        ax.plot(ages, np.array(counts) + 1,
                color=color,
                linewidth=1.5,
                label=f'Subtype {i}' if i < 6 else 'All subtypes')

    ax.set_xlabel('Age')
    ax.set_ylabel('# Significant Biomarkers')
    ax.set_xlim([40,70])
    # ax.axvline(49, color='black', linewidth=1.0, linestyle='--')
    # ax.axvline(64, color='black', linewidth=1.0, linestyle='--')
    # ax.set_title('Significant Biomarkers Comparison (nsubtype=5)')
    # ax.grid(True, linestyle='--', alpha=0.7)
    # ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/significant_protein_peak_linear_regression_{correction_method}_{window}.png', dpi=300, bbox_inches='tight')
    plt.close()

# print(f'T-statistic plots saved to "{OUTPUT_DIR}/significant_protein_peak_linear_regression_{correction_method}.png"')