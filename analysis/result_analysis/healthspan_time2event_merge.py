import pandas as pd
import utils.utils as utils
import os
import json
import numpy as np
from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt

def parse_target_field(df: pd.DataFrame) -> pd.DataFrame:
    df_result = pd.DataFrame()
    df_result['eid'] = df['eid']
    if 'Field' in df.columns:
        df_result['field'] = df['Field'].apply(lambda x: x.split(' ')[1] if not pd.isna(x) else x)
    elif 'target_cancer' in df.columns:
        df_result['field'] = df['target_cancer']
    elif 'target_death' in df.columns:
        df_result['field'] = df['target_death']
    else:
        raise ValueError('Field not found in dataframe')
    df_result['bl2t'] = df['BL2Target_yrs']
    df_result['event'] = df['target_y']
    return df_result

# def calculate_life_expectancy_from_kmf(kmf):
#     """
#     从KM拟合结果计算生存期望
#     """
#     timeline = kmf.timeline
#     survival_function = kmf.survival_function_.iloc[:, 0]
#     return np.trapz(survival_function.values, timeline)

def calculate_life_expectancy_from_kmf(kmf, tau):
    timeline = kmf.timeline
    survival_function = kmf.survival_function_.iloc[:, 0]

    mask = timeline <= tau
    timeline = timeline[mask]
    survival_function = survival_function[mask]

    # 保证最后一个点刚好是 tau
    if timeline[-1] < tau:
        timeline = np.append(timeline, tau)
        survival_function = np.append(
            survival_function.values,
            survival_function.values[-1]
        )
    else:
        timeline = timeline
        survival_function = survival_function.values

    return np.trapz(survival_function, timeline)


from joblib import Parallel, delayed

def bootstrap_life_expectancy_ci(
    survival_time, event,
    confidence_level=0.95,
    n_bootstrap=1000,
    random_state=42,
    n_jobs=-1
):
    """
    使用并行 bootstrap 计算生存期望置信区间
    """
    np.random.seed(random_state)
    
    # 原始数据的生存期望
    kmf_original = KaplanMeierFitter()
    kmf_original.fit(survival_time, event)
    original_le = calculate_life_expectancy_from_kmf(kmf_original, common_tau)

    n_samples = len(survival_time)

    def single_bootstrap(seed):
        """单次bootstrap"""
        rng = np.random.RandomState(seed)
        try:
            indices = rng.choice(n_samples, size=n_samples, replace=True)
            surv = survival_time.iloc[indices] if hasattr(survival_time, "iloc") else survival_time[indices]
            evt = event.iloc[indices] if hasattr(event, "iloc") else event[indices]

            kmf = KaplanMeierFitter()
            kmf.fit(surv, evt)
            return calculate_life_expectancy_from_kmf(kmf, common_tau)
        except Exception:
            return None

    # 并行运行
    results = Parallel(n_jobs=n_jobs, verbose=1)(
        delayed(single_bootstrap)(random_state + i) for i in range(n_bootstrap)
    )

    # 去掉失败的
    bootstrap_les = np.array([r for r in results if r is not None])

    # CI
    alpha = 1 - confidence_level
    ci_lower = np.percentile(bootstrap_les, (alpha/2) * 100)
    ci_upper = np.percentile(bootstrap_les, (1 - alpha/2) * 100)

    return {
        "life_expectancy": original_le,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "confidence_level": confidence_level,
        "bootstrap_samples": bootstrap_les,
        "n_bootstrap_success": len(bootstrap_les)
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
        
        life_expectancy = calculate_life_expectancy_from_kmf(kmf, common_tau)
        
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

def compute_common_tau(survival_time, event, subtype):
    taus = []
    for sub in sorted(set(subtype.dropna())):
        mask = (subtype == sub) & (event == 1)
        if mask.sum() > 0:
            taus.append(survival_time[mask].max())
    return min(taus)


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


with open('preprocess/data/important_disease_healthspan.json', 'r') as f:
    important_disease = json.load(f)

nsubtype = 5
exp_name = 'ukb_MixtureGRBF_cv_nsubtype_biom17'
output_dir = f'./output/result_analysis/{exp_name}/{nsubtype}_subtypes'
subtype_stage = utils.get_subtype_stage(exp_name, nsubtype)
subtype_stage = subtype_stage.rename(columns={'PTID': 'eid'})

subtype_stage['event'] = 0
subtype_stage['bl2t'] = np.inf
subtype_stage.set_index('eid', inplace=True)

os.makedirs(output_dir, exist_ok=True)

result_records = []

# diseases
for i, (disease_name, disease_code) in enumerate(important_disease.items()):
    disease_upper_level_code = disease_code[0][0]
    
    disease_info = pd.read_csv(f'./input/disease_info/{disease_upper_level_code}0.csv')
    disease_info = parse_target_field(disease_info)
    disease_info = disease_info.loc[disease_info['field'].isin(disease_code)]
    merged_df = subtype_stage.merge(
        disease_info[['eid', 'event', 'bl2t']],
        on='eid',
        how='inner', # 只处理在两个DataFrame中都存在的eid
        suffixes=('_current', '_new')
    )
    merged_df.set_index('eid', inplace=True)
    mask = (merged_df['event_new'] == 1) & (merged_df['bl2t_new'] < merged_df['bl2t_current'])

    rows_to_update = merged_df[mask]
    
    if not rows_to_update.empty:
        update_data = rows_to_update[['event_new', 'bl2t_new']]
        update_data.rename(columns={'event_new': 'event', 'bl2t_new': 'bl2t'}, inplace=True)
        subtype_stage.update(update_data)
    
    print(f"处理完疾病 '{disease_name}'，更新了 {len(rows_to_update)} 条记录。")

death_info = pd.read_csv(f'./input/disease_info/X0.csv')
death_info = parse_target_field(death_info)
merged_df = subtype_stage.merge(
        death_info[['eid', 'event', 'bl2t']],
        on='eid',
        how='inner', # 只处理在两个DataFrame中都存在的eid
        suffixes=('_current', '_new')
    )
merged_df.set_index('eid', inplace=True)

mask = ((merged_df['event_new'] == 1) & (merged_df['bl2t_new'] < merged_df['bl2t_current']) | (merged_df['event_new'] == 0))

rows_to_update = merged_df[mask]
    
if not rows_to_update.empty:
    update_data = rows_to_update[['event_new', 'bl2t_new']]
    update_data.rename(columns={'event_new': 'event', 'bl2t_new': 'bl2t'}, inplace=True)
    subtype_stage.update(update_data)

print(f"处理完疾病 'Death'，更新了 {len(rows_to_update)} 条记录。")

subtype_stage.reset_index(inplace=True)

event = subtype_stage['event']
survival_time = subtype_stage['bl2t'] + subtype_stage['stage']


# 读取数据
prot_var_path = f'output/result_analysis/{exp_name}/{nsubtype}_subtypes/protein_intra_variance.csv'
prot_var_table = pd.read_csv(prot_var_path)

prot_age_gap_path = f'output/result_analysis/{exp_name}/{nsubtype}_subtypes/all_protein_stage_pred.csv'
prot_age_gap_table = pd.read_csv(prot_age_gap_path)
prot_age_gap_table = prot_age_gap_table.rename(columns={'RID': 'eid'})

# 合并
merged = pd.merge(prot_var_table, prot_age_gap_table, on='eid', how='inner')

# 按照 std_dev 四分位分组
quantile_number = 5
merged['subtype_sigma'] = pd.qcut(merged['std_dev'], quantile_number, labels=[i for i in range(1, quantile_number + 1)])

# 按照 stage_pred 四分位分组
merged['subtype_delta'] = pd.qcut(merged['age_gap'], quantile_number, labels=[i for i in range(1, quantile_number + 1)])

# 转换为 int 类型（可选）
merged['subtype_sigma'] = merged['subtype_sigma'].astype(int)
merged['subtype_delta'] = merged['subtype_delta'].astype(int)

df_s = pd.merge(subtype_stage, merged[['eid', 'subtype_sigma', 'subtype_delta']], on='eid', how='left')

df_s.to_csv('tmp.csv')

common_tau = compute_common_tau(survival_time, event, df_s['subtype'])

# 假设你在循环里把每个 results_df 存储起来
all_results = []

for subtype_name in ['subtype','subtype_sigma','subtype_delta']:
    subtype = df_s[subtype_name]

    results_df = kmf_subtype_with_life_expectancy(
        survival_time, event, subtype, confidence_level=0.95
    )
    results_df["method"] = subtype_name  # 记录分类方式
    all_results.append(results_df)

# 合并
plot_df = pd.concat(all_results, ignore_index=True)


# 绘图

plt.rcParams.update({
    'font.family': 'Arial',
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'legend.fontsize': 8
})

fig, ax = plt.subplots(figsize=(7/2.54, 6/2.54))

method_names = ['MGRBF Subtype', 'ProtVar percentile', 'ProtAgeGap percentile']
# 用误差棒图：x=method+subtype, y=life_expectancy, 误差=95CI
for idx, method in enumerate(plot_df['method'].unique()):
    subset = plot_df[plot_df['method'] == method]
    ax.errorbar(
        subset['subtype'] + (-0.2 if idx == 0 else 0 if idx == 1 else 0.2),  # x轴位置错开
        subset['life_expectancy'],
        yerr=[subset['life_expectancy'] - subset['le_ci_lower'], 
              subset['le_ci_upper'] - subset['life_expectancy']],
        fmt='o', capsize=0, markersize=3, label=method_names[idx]
    )

ax.set_xticks(range(1, len(plot_df['subtype'].unique()) + 1))
ax.set_xlabel("Subtype")
ax.set_ylabel("Healthspan")
ax.set_title("Comparison of healthspan by subtype")
ax.legend(frameon=False)
plt.savefig('output/result_analysis/ukb_MixtureGRBF_cv_nsubtype_biom17/5_subtypes/healthspan_comparison_errorplot.png', dpi=300)
