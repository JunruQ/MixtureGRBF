import pandas as pd
from lifelines import KaplanMeierFitter
import numpy as np
import matplotlib.pyplot as plt
import utils.utils as utils

nsubtype = 5
exp_name = 'ukb_MixtureGRBF_cv_nsubtype_biom17'

# 读取数据
subtype_stage = utils.get_subtype_stage(exp_name, nsubtype).rename(columns={'PTID': 'eid'})

prot_var_path = f'output/result_analysis/{exp_name}/{nsubtype}_subtypes/protein_intra_variance.csv'
prot_var_table = pd.read_csv(prot_var_path)

prot_age_gap_path = f'output/result_analysis/{exp_name}/{nsubtype}_subtypes/all_protein_stage_pred.csv'
prot_age_gap_table = pd.read_csv(prot_age_gap_path)
prot_age_gap_table = prot_age_gap_table.rename(columns={'RID': 'eid'})

# 合并
merged = pd.merge(prot_var_table, prot_age_gap_table, on='eid', how='inner')

# 添加 subtype_stage 列
merged = pd.merge(merged, subtype_stage[['eid', 'subtype']], on='eid', how='left')

# 按照 std_dev 四分位分组
quantile_number = 5
merged['subtype_sigma'] = pd.qcut(merged['std_dev'], quantile_number, labels=[i for i in range(1, quantile_number + 1)])

# 按照 stage_pred 四分位分组
merged['subtype_delta'] = pd.qcut(merged['age_gap'], quantile_number, labels=[i for i in range(1, quantile_number + 1)])


# 转换为 int 类型（可选）
merged['subtype_sigma'] = merged['subtype_sigma'].astype(int)
merged['subtype_delta'] = merged['subtype_delta'].astype(int)

# 生存分析

def parse_target_field(df: pd.DataFrame) -> pd.DataFrame:
    df_result = pd.DataFrame()
    df_result['eid'] = df['eid']
    df_result['bl2t'] = df['BL2Target_yrs']
    df_result['event'] = df['target_y']
    return df_result

# def calculate_life_expectancy_from_kmf(kmf_fit, max_time=None):
#     """
#     从Kaplan-Meier拟合结果计算生存期望（曲线下面积）
    
#     Parameters:
#     kmf_fit: KaplanMeierFitter对象（已拟合）
#     max_time: 最大时间点，如果为None则使用观察到的最大时间
    
#     Returns:
#     life_expectancy: 生存期望
#     """
#     if max_time is None:
#         max_time = kmf_fit.timeline.max()
#         # print(max_time)
    
#     # 获取生存函数在指定时间点的值
#     survival_function = kmf_fit.survival_function_
#     timeline = survival_function.index
    
#     # 确保时间点在合理范围内
#     timeline = timeline[timeline <= max_time]
#     survival_probs = survival_function.loc[timeline].iloc[:, 0].values
    
#     # 使用梯形法则计算曲线下面积（生存期望）
#     life_expectancy = np.trapz(survival_probs, timeline)
    
#     return life_expectancy

# def calculate_life_expectancy_with_ci(survival_time, event, confidence_level=0.95):
#     """
#     计算生存期望及其置信区间
    
#     Parameters:
#     survival_time: 生存时间
#     event: 事件发生指示符
#     confidence_level: 置信水平
    
#     Returns:
#     dict包含生存期望和置信区间
#     """
#     kmf = KaplanMeierFitter()
#     kmf.fit(survival_time, event, alpha=1-confidence_level)
    
#     # 计算点估计的生存期望
#     life_expectancy = calculate_life_expectancy_from_kmf(kmf)
    
#     # 获取置信区间
#     ci_lower = kmf.confidence_interval_.iloc[:, 0]  # 下界
#     ci_upper = kmf.confidence_interval_.iloc[:, 1]  # 上界
    
#     # 计算置信区间对应的生存期望
#     timeline = ci_lower.index
#     le_lower = np.trapz(ci_lower.values, timeline)
#     le_upper = np.trapz(ci_upper.values, timeline)
    
#     return {
#         'life_expectancy': life_expectancy,
#         'ci_lower': le_lower,
#         'ci_upper': le_upper,
#         'confidence_level': confidence_level
#     }

# def kmf_subtype_with_life_expectancy(survival_time, event, subtype, confidence_level=0.95):
#     """
#     按亚型进行Kaplan-Meier生存分析并计算每组的生存期望
    
#     Parameters:
#     survival_time: 生存时间
#     event: 事件发生指示符
#     subtype: 亚型分组
#     confidence_level: 置信水平
#     plot: 是否绘图
    
#     Returns:
#     results_df: 包含每个亚型生存期望的DataFrame
#     """
#     results = []
    
#     unique_subtypes = sorted(set(subtype.dropna()))
    
#     for idx, sub in enumerate(unique_subtypes):
#         mask = (subtype == sub) & ~pd.isna(survival_time) & ~pd.isna(event)
        
#         if mask.sum() == 0:  # 跳过空组
#             continue
            
#         sub_survival_time = survival_time[mask]
#         sub_event = event[mask]
        
#         # 计算生存期望和置信区间
#         le_results = calculate_life_expectancy_with_ci(
#             sub_survival_time, sub_event, confidence_level
#         )
        
#         # 存储结果
#         results.append({
#             'subtype': sub,
#             'n_samples': mask.sum(),
#             'n_events': sub_event.sum(),
#             'life_expectancy': le_results['life_expectancy'],
#             'le_ci_lower': le_results['ci_lower'],
#             'le_ci_upper': le_results['ci_upper'],
#             'confidence_level': confidence_level
#         })
    
#     # 转换为DataFrame
#     results_df = pd.DataFrame(results)
    
#     return results_df

def calculate_life_expectancy_from_kmf(kmf):
    """
    从KM拟合结果计算生存期望
    """
    timeline = kmf.timeline
    survival_function = kmf.survival_function_.iloc[:, 0]
    return np.trapz(survival_function.values, timeline)

def bootstrap_life_expectancy_ci(survival_time, event, confidence_level=0.95, n_bootstrap=1000, random_state=42):
    """
    使用bootstrap方法计算生存期望的置信区间
    
    Parameters:
    survival_time: 生存时间
    event: 事件发生指示符
    confidence_level: 置信水平
    n_bootstrap: bootstrap重采样次数
    random_state: 随机种子
    
    Returns:
    dict包含生存期望和bootstrap置信区间
    """
    np.random.seed(random_state)
    
    # 原始数据的生存期望
    kmf_original = KaplanMeierFitter()
    kmf_original.fit(survival_time, event)
    original_le = calculate_life_expectancy_from_kmf(kmf_original)
    
    # Bootstrap重采样
    bootstrap_les = []
    n_samples = len(survival_time)
    
    import tqdm
    for _ in tqdm.tqdm(range(n_bootstrap)):
        # 有放回抽样
        bootstrap_indices = np.random.choice(n_samples, size=n_samples, replace=True)
        bootstrap_survival = survival_time.iloc[bootstrap_indices] if hasattr(survival_time, 'iloc') else survival_time[bootstrap_indices]
        bootstrap_event = event.iloc[bootstrap_indices] if hasattr(event, 'iloc') else event[bootstrap_indices]
        
        try:
            # 拟合KM曲线
            kmf_bootstrap = KaplanMeierFitter()
            kmf_bootstrap.fit(bootstrap_survival, bootstrap_event)
            
            # 计算生存期望
            bootstrap_le = calculate_life_expectancy_from_kmf(kmf_bootstrap)
            bootstrap_les.append(bootstrap_le)
            
        except Exception as e:
            # 如果某次bootstrap失败，跳过
            continue
    
    bootstrap_les = np.array(bootstrap_les)
    
    # 计算置信区间
    alpha = 1 - confidence_level
    ci_lower = np.percentile(bootstrap_les, (alpha/2) * 100)
    ci_upper = np.percentile(bootstrap_les, (1 - alpha/2) * 100)
    
    return {
        'life_expectancy': original_le,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'confidence_level': confidence_level,
        'bootstrap_samples': bootstrap_les,
        'n_bootstrap_success': len(bootstrap_les)
    }

def calculate_life_expectancy_with_ci(survival_time, event, confidence_level=0.95, 
                                    method='bootstrap', n_bootstrap=1000, random_state=42):
    """
    计算生存期望及其置信区间
    
    Parameters:
    survival_time: 生存时间
    event: 事件发生指示符
    confidence_level: 置信水平
    method: 'bootstrap' 或 'km_ci'（原方法）
    n_bootstrap: bootstrap重采样次数
    random_state: 随机种子
    
    Returns:
    dict包含生存期望和置信区间
    """
    if method == 'bootstrap':
        return bootstrap_life_expectancy_ci(
            survival_time, event, confidence_level, n_bootstrap, random_state
        )
    else:
        # 保留原来的方法作为备选
        kmf = KaplanMeierFitter()
        kmf.fit(survival_time, event, alpha=1-confidence_level)
        
        life_expectancy = calculate_life_expectancy_from_kmf(kmf)
        
        ci_lower = kmf.confidence_interval_.iloc[:, 0]
        ci_upper = kmf.confidence_interval_.iloc[:, 1]
        
        timeline = ci_lower.index
        le_lower = np.trapz(ci_lower.values, timeline)
        le_upper = np.trapz(ci_upper.values, timeline)
        
        return {
            'life_expectancy': life_expectancy,
            'ci_lower': le_lower,
            'ci_upper': le_upper,
            'confidence_level': confidence_level
        }

def kmf_subtype_with_life_expectancy(survival_time, event, subtype, confidence_level=0.95, 
                                   method='bootstrap', n_bootstrap=1000, random_state=42):
    """
    按亚型进行Kaplan-Meier生存分析并计算每组的生存期望
    
    Parameters:
    survival_time: 生存时间
    event: 事件发生指示符
    subtype: 亚型分组
    confidence_level: 置信水平
    method: 'bootstrap' 或 'km_ci'
    n_bootstrap: bootstrap重采样次数
    random_state: 随机种子
    
    Returns:
    results_df: 包含每个亚型生存期望的DataFrame
    """
    results = []
    
    unique_subtypes = sorted(set(subtype.dropna()))
    
    for idx, sub in enumerate(unique_subtypes):
        mask = (subtype == sub) & ~pd.isna(survival_time) & ~pd.isna(event)
        
        if mask.sum() == 0:  # 跳过空组
            continue
            
        sub_survival_time = survival_time[mask]
        sub_event = event[mask]
        
        # 计算生存期望和置信区间
        le_results = calculate_life_expectancy_with_ci(
            sub_survival_time, sub_event, confidence_level, 
            method=method, n_bootstrap=n_bootstrap, 
            random_state=random_state + idx  # 为每个亚型使用不同的随机种子
        )
        
        # 存储结果
        result_dict = {
            'subtype': sub,
            'n_samples': mask.sum(),
            'n_events': sub_event.sum(),
            'life_expectancy': le_results['life_expectancy'],
            'le_ci_lower': le_results['ci_lower'],
            'le_ci_upper': le_results['ci_upper'],
            'confidence_level': confidence_level,
            'method': method
        }
        
        # 如果使用bootstrap方法，添加额外信息
        if method == 'bootstrap':
            result_dict['n_bootstrap_success'] = le_results.get('n_bootstrap_success', 0)
        
        results.append(result_dict)
    
    # 转换为DataFrame
    results_df = pd.DataFrame(results)
    
    return results_df

disease_info = pd.read_csv(f'./input/disease_info/X0.csv')

df_s = pd.merge(merged, parse_target_field(disease_info), on='eid', how='left')

event = df_s['event']
survival_time = df_s['bl2t'] + df_s['stage']

# 假设你在循环里把每个 results_df 存储起来
all_results = []

for subtype_name in ['subtype', 'subtype_sigma','subtype_delta']:
    subtype = df_s[subtype_name]

    results_df = kmf_subtype_with_life_expectancy(
        survival_time, event, subtype, confidence_level=0.95
    )
    results_df["method"] = subtype_name  # 记录分类方式
    all_results.append(results_df)


# 合并
plot_df = pd.concat(all_results, ignore_index=True)

# 绘图
fig, ax = plt.subplots(figsize=(6,4))

method_names = ['MixtureGRBF Subtype', 'Protein Variablity', 'Protein Age Gap']
# 用误差棒图：x=method+subtype, y=life_expectancy, 误差=95CI
for idx, method in enumerate(plot_df['method'].unique()):
    subset = plot_df[plot_df['method'] == method]
    ax.errorbar(
        subset['subtype'] + (-0.1 if idx == 0 else 0 if idx == 1 else 0.1),  # x轴位置错开
        subset['life_expectancy'],
        yerr=[subset['life_expectancy'] - subset['le_ci_lower'], 
              subset['le_ci_upper'] - subset['life_expectancy']],
        fmt='o', capsize=4, label=method_names[idx]
    )

ax.set_xticks(range(1, len(plot_df['subtype'].unique()) + 1))
ax.set_xlabel("Subtype")
ax.set_ylabel("Life expectancy")
ax.set_title("Comparison of life expectancy by subtype")
ax.legend(title="Method")
plt.savefig('tmp.png', dpi=300)