import utils.utils as utils
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
import numpy as np
import json

sites = ['Northern England', 'Southern England', 'Midlands', 'Scotland', 'Wales']

models = []

exp_name = 'ukb_MixtureGRBF_cv_nsubtype_biom17'
nsubtype = 5
subtype_stage = utils.get_subtype_stage_with_cov(exp_name, nsubtype).rename(columns={'PTID': 'eid'})

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

'''
1
'''

prot_var_path = f'output/result_analysis/{exp_name}/{nsubtype}_subtypes/protein_intra_variance.csv'
prot_var_table = pd.read_csv(prot_var_path)

prot_age_gap_path = f'output/result_analysis/{exp_name}/{nsubtype}_subtypes/all_protein_stage_pred.csv'
prot_age_gap_table = pd.read_csv(prot_age_gap_path)
prot_age_gap_table = prot_age_gap_table.rename(columns={'RID': 'eid'})

# 合并
merged = pd.merge(prot_var_table, prot_age_gap_table, on='eid', how='inner')

# 添加 subtype_stage 列
subtype_stage = pd.merge(merged, subtype_stage[['eid', 'subtype', 'centre', 'sex']], on='eid', how='left')

with open('preprocess/data/important_disease_healthspan.json', 'r') as f:
    important_disease = json.load(f)

subtype_stage['event'] = 0
subtype_stage['bl2t'] = np.inf
subtype_stage.set_index('eid', inplace=True)

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

# print(f"处理完疾病 'Death'，更新了 {len(rows_to_update)} 条记录。")

subtype_stage.reset_index(inplace=True)

df = subtype_stage



'''
2
'''
site_map = pd.read_csv('input/ukb/ukb_site_map.csv', header=None, index_col=1)

'''
3
'''
import matplotlib.pyplot as plt
import seaborn as sns
n_bootstrap = 1000  # bootstrap次数

cov_combination = [
    ['stage', 'sex'], ['std_dev'], ['age_gap'],
    ['stage', 'sex', 'std_dev'], ['stage', 'sex', 'age_gap'],
    ['std_dev', 'age_gap'], ['stage', 'sex', 'std_dev', 'age_gap']
]

cov_combination_names = [
    'Age + Sex', 'ProtVar', 'ProtAgeGap',
    'Age + Sex + ProtVar', 'Age + Sex + ProtAgeGap',
    'ProtVar + ProtAgeGap',
    'Age + Sex + ProtVar + ProtAgeGap'
]

results = []

# 按 centre 统计人数
centre_counts = (
    df['centre']
    .value_counts()
    .sort_index()
    .reset_index()
)
centre_counts.columns = ['Centre ID', 'Count']

# 先把 site_map 展平成 centre -> region 的映射
centre_region = site_map.reset_index()
centre_region.columns = ['Region', 'CentreID']

# 合并人数信息
df_summary = centre_counts.merge(
    centre_region,
    left_on='中心ID (Centre ID)',
    right_on='CentreID',
    how='left'
)

df_summary = df_summary[['Region', 'Centre ID', 'Count']]


# 3. 打印输出
print("--- UKB 生存分析站点数据统计 ---")
print(df_summary.to_string(index=False))

# 4. 可选：保存为 CSV
df_summary.to_csv('output/site_population_summary.csv', index=False, encoding='utf-8-sig')

from joblib import Parallel, delayed

def bootstrap_single(cph, test_data, covariates, random_seed):
    """单次bootstrap的函数"""
    np.random.seed(random_seed)
    sample = test_data.sample(frac=1.0, replace=True).reset_index()
    risk_scores = cph.predict_partial_hazard(sample[covariates])
    c_index = concordance_index(
        sample["bl2t"], -risk_scores, sample["event"]
    )
    return c_index

# joblib版本的并行bootstrap
for idx, covariates in enumerate(cov_combination):
    print(f"--- 正在计算 {cov_combination_names[idx]} 的 Concordance Index ---")
    cov_label = cov_combination_names[idx]
    
    for site in sites:
        train_idx = df['centre'].isin(site_map.loc[site, 0].tolist())
        print(f"{site}: {len(df[train_idx])}")
        test_idx = ~train_idx

        # 训练 Cox 模型
        cph = CoxPHFitter()
        cph.fit(
            df.loc[train_idx, ["bl2t", "event"] + covariates],
            duration_col="bl2t", event_col="event"
        )

        test_data = df.loc[test_idx, ["bl2t", "event"] + covariates]

        # 生成随机种子
        random_seeds = np.random.randint(0, 2**32 - 1, n_bootstrap)
        
        # 使用joblib并行执行
        c_indices = Parallel(n_jobs=-1, verbose=1)(
            delayed(bootstrap_single)(cph, test_data, covariates, seed)
            for seed in random_seeds
        )
        # import tqdm
        # c_indices = [bootstrap_single(cph, test_data, covariates, seed) for seed in tqdm.tqdm(random_seeds)]

        # 收集结果
        for c_index in c_indices:
            results.append({
                "site": site,
                "covariates": cov_label,
                "c_index": c_index
            })

# 转为 DataFrame
df_results = pd.DataFrame(results)

for cov in cov_combination_names:
    vals = df_results.loc[(df_results['covariates'] == cov),'c_index'].values
    print(f"{cov}: {np.mean(vals):.3f}")

site_names = ['Northern\nEngland', 'Southern\nEngland', 'Midlands', 'Scotland', 'Wales']
covs = cov_combination_names

# 每个 site 上 box 的宽度
width = 0.1  
site_positions = np.arange(1, len(sites) + 1)

plt.rcParams.update({
    'font.family': 'Arial',  # 更清晰的字体
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'legend.fontsize': 8
})

fig, ax = plt.subplots(figsize=(18/2.54, 5/2.54))

colors = utils.categorical_palette # 颜色循环

for j, cov in enumerate(covs):
    # 平移后的位置
    positions = site_positions + (j - len(covs)/2) * width

    heights = []
    errors = []
    for site in sites:
        vals = df_results.loc[
            (df_results['site'] == site) & (df_results['covariates'] == cov),
            'c_index'
        ].values

        heights.append(np.mean(vals))
        # 使用标准差作为误差棒，可以改为置信区间
        errors.append(np.std(vals))  

    ax.bar(
        positions, heights,
        width=width*0.9,
        color=colors[j % len(colors)],
        alpha=0.9,
        label=cov,
        yerr=errors,  # 添加误差棒
        capsize=0     # 误差棒横线长度
    )

# 设置坐标轴
ax.set_xticks(site_positions)
ax.set_xticklabels(site_names)
ax.set_ylim(0.5, 0.8)
ax.set_ylabel("C-index")

# 图例
# ax.legend(
#     title="Covariates",
#     bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.
# )

plt.savefig("output/result_analysis/ukb_MixtureGRBF_cv_nsubtype_biom17/5_subtypes/hold_out_c-index.png", dpi=300, bbox_inches="tight")
plt.show()
