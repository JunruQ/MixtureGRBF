import pandas as pd
import numpy as np
from scipy import stats
import json

def bonferroni_correction(p_values):
    n = len(p_values)
    corrected_p_values = np.array(p_values) * n
    return corrected_p_values

nsubtype = 5
exp_name = 'ukb_MixtureGRBF_cv_nsubtype_biom17'
INPUT_TABLE_PATH = 'input/ukb/ukb_covreg1_trans1_nanf1_biom0.csv'
SUBTYPE_STAGE_PATH = f'output/{exp_name}/{nsubtype}_subtypes/subtype_stage.csv'
OUTPUT_DIR = f'output/result_analysis/{exp_name}/{nsubtype}_subtypes'
SUBTYPE_ORDER_PATH = f'output/result_analysis/{exp_name}/{nsubtype}_subtypes/all_cause_mortality_order.csv'
CLASSIFICATION_PATH = 'analysis/result_analysis/protein_classification_form.csv'

df = pd.read_csv(INPUT_TABLE_PATH)
df.rename(columns={'RID':'PTID'}, inplace=True)
subtype_stage = pd.read_csv(SUBTYPE_STAGE_PATH)
classification = pd.read_csv(CLASSIFICATION_PATH)
try:
    subtype_order = pd.read_csv(SUBTYPE_ORDER_PATH, header=None)
except:
    subtype_order = pd.DataFrame(np.array(list(range(1, nsubtype+1))))
subtype_order_map = {int(subtype_order.iloc[i]): i+1 for i in range(nsubtype)}
subtype_stage['subtype'] = subtype_stage['subtype'].map(subtype_order_map)

# 合并数据
df = pd.merge(df, subtype_stage, how='inner', on=['PTID', 'stage'])
biomarker_names = df.iloc[:, 7:].columns.tolist()

alpha = 0.05
window_size = 5  # 设置窗口大小

# 找到最小和最大stage
min_stage = int(df['stage'].min())
max_stage = int(df['stage'].max())

# 计算stage窗口
stage_windows = [(start, min(start + window_size - 1, max_stage)) 
                for start in range(min_stage, max_stage + 1 - window_size + 1)]

# 存储所有结果的字典
stage_window_results = {}

# 以subtype作为外层循环
for i in range(2, nsubtype + 1):
    subtype_results = {}
    
    # 遍历每个stage window
    for start, end in stage_windows:
        window_df = df[(df['stage'] >= start) & (df['stage'] <= end)]
        
        case_group = window_df['subtype'] == i
        control_group = window_df['subtype'] == 1
        
        # 跳过样本数不足的情况
        if len(window_df[case_group]) < 2 or len(window_df[control_group]) < 2:
            continue
            
        ts = []
        ps = []
        
        for biom in biomarker_names:
            case = window_df.loc[case_group, biom]
            control = window_df.loc[control_group, biom]
            t_value, p_value = stats.ttest_ind(case, control, equal_var=False)
            ts.append(t_value)
            ps.append(p_value)
        
        corrected_ps = bonferroni_correction(ps)
        is_significant = (corrected_ps <= alpha).tolist()
        
        # 只存储显著的蛋白
        significant_proteins = [biom for j, biom in enumerate(biomarker_names) if is_significant[j]]
        if significant_proteins:
            subtype_results[start] = {
                'significant_proteins': significant_proteins,
                'sig_count': len(significant_proteins)
            }
    
    if subtype_results:
        stage_window_results[i] = subtype_results

# 将stage window结果保存为JSON
with open(f'{OUTPUT_DIR}/stage_window_significant_proteins.json', 'w') as f:
    json.dump(stage_window_results, f, indent=4)

# 打印结果
print(f"Min stage: {min_stage}, Max stage: {max_stage}")
print(f"Stage windows: {stage_windows}")
for subtype_key, subtype_data in stage_window_results.items():
    print(f"\n{subtype_key}:")
    for window_key, window_data in subtype_data.items():
        print(f"  {window_key}: {window_data['sig_count']} significant proteins")