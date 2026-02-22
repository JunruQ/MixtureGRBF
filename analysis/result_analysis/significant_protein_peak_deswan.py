import pandas as pd
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib.lines import Line2D
from concurrent.futures import ProcessPoolExecutor
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
import tqdm

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

def perform_regression(biom_data, age_binary, sex_data):
    """执行线性回归并返回Type II ANOVA的p值，使用公式接口"""
    # 创建DataFrame以兼容statsmodels.formula.api
    data = pd.DataFrame({
        'biomarker': biom_data,
        'age_binary': age_binary,
        'sex': sex_data
    })
    
    # 去除NaN值
    data = data.dropna()
    if len(data) <= 2:
        return 1.0  # 数据不足时返回非显著p值
    
    # 使用公式接口构建模型
    model = smf.ols('biomarker ~ age_binary + sex', data=data)
    results = model.fit()
    
    # 使用Type II sum of squares进行ANOVA分析
    anova_results = anova_lm(results, typ=2)
    p_value = anova_results['PR(>F)'].loc['age_binary']  # 获取age_binary的p值
    
    return p_value


def perform_regression_batch(biom_matrix, age_binary, sex_data):
    """
    批量执行线性回归，返回所有 biomarker 对应的 age_binary 的 p 值
    biom_matrix: shape = (n_samples, n_biomarkers)
    age_binary: shape = (n_samples,)
    sex_data: shape = (n_samples,)
    """
    n_samples, n_biom = biom_matrix.shape
    
    # 构建设计矩阵 X = [1, age_binary, sex]
    X = np.column_stack([
        np.ones(n_samples),
        age_binary,
        sex_data
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


def precompute_window_indices(stage_array, ages, window_size):
    """Precompute window indices for all ages."""
    half_window = window_size / 2
    indices = []
    for lk in ages:
        start1, end1, start2, end2 = get_window_indices(stage_array, lk, window_size)
        indices.append((start1, end1, start2, end2))
    return indices

def count_significant_biom_fast(subtype_data, age_min, age_max, window_size=10, alpha=0.05, correction_method='fdr'):
    """使用批量矩阵回归进行显著性检验，符合DEswan方法"""
    stage_array = subtype_data['stage']
    biom_arrays = subtype_data['biomarkers']
    biom_names = list(biom_arrays.keys())
    biom_matrix_full = np.column_stack([biom_arrays[biom] for biom in biom_names])
    sex_array = subtype_data['sex']
    
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
        sex_data = sex_array[indices]
        biom_matrix = biom_matrix_full[indices, :]
        
        # 批量回归
        p_values = perform_regression_batch(biom_matrix, age_binary, sex_data)
        
        # 多重检验校正
        if correction_method == 'fdr':
            significant, q_values = fdr_correction(p_values, alpha)
        elif correction_method == 'bonf':
            significant, q_values = bonferroni_correction(p_values, alpha)
        else:
            raise ValueError("Unsupported correction method. Use 'fdr' or 'bonf'.")
        
        counts.append(np.sum(significant))
        all_q_values.append(q_values)
    
    return counts, ages, all_q_values

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
    counts, ages, _ = count_significant_biom_fast(subtype_dfs[i], age_min, age_max, window_size=window, correction_method=correction_method)
    return (i, window, counts, ages)

windows = [2, 3, 4, 5, 10]
counts_results = {w: {} for w in windows}


correction_method = 'fdr'
with ProcessPoolExecutor() as executor:
    args_list = [(i, w, correction_method) for i in range(1, nsubtype+1) for w in windows]
    for result in executor.map(compute_counts, args_list):
        i, window, counts, ages = result
        counts_results[window][i] = (counts, ages)

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
plt.savefig(OUTPUT_DIR + f'/significant_protein_peak_deswan_{correction_method}.png', 
            dpi=300, bbox_inches='tight')
plt.close()