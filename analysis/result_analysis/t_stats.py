# import pandas as pd
# import numpy as np
# from scipy import stats
# import os

# def bonferroni_correction(p_values):
#     n = len(p_values)
#     corrected_p_values = np.array(p_values) * n
#     return np.minimum(corrected_p_values, 1.0)  # Cap at 1.0 as p-values can't exceed 1

# nsubtype = 5
# exp_name = 'ukb_MixtureGRBF_cv_nsubtype_biom17'
# INPUT_TABLE_PATH = 'input/ukb/ukb_covreg1_trans1_nanf1_biom0.csv'
# SUBTYPE_STAGE_PATH = f'output/{exp_name}/{nsubtype}_subtypes/subtype_stage.csv'
# OUTPUT_DIR = f'output/result_analysis/{exp_name}/{nsubtype}_subtypes'
# SUBTYPE_ORDER_PATH = f'output/result_analysis/{exp_name}/{nsubtype}_subtypes/all_cause_mortality_order.csv'

# df = pd.read_csv(INPUT_TABLE_PATH)
# df.rename(columns={'RID':'PTID'}, inplace=True)
# subtype_stage = pd.read_csv(SUBTYPE_STAGE_PATH)
# try:
#     subtype_order = pd.read_csv(SUBTYPE_ORDER_PATH, header=None)
# except:
#     subtype_order = pd.DataFrame(np.array(list(range(1, nsubtype+1))))
# biomarker_names = df.iloc[:, 7:].columns.tolist()
# df = pd.merge(df, subtype_stage, how='inner', on=['PTID', 'stage'])

# alpha = 0.05

# # 创建输出目录（如果不存在）
# os.makedirs(OUTPUT_DIR, exist_ok=True)

# # 创建列表来存储所有数据
# results = []

# for i in range(1, nsubtype + 1):
#     k = int(subtype_order.iloc[i-1, 0])
#     case_group = df['subtype'] == k
#     control_group = df['subtype'] != k
    
#     # 计算 t 统计量和 p 值
#     for biom in biomarker_names:
#         case = df.loc[case_group, biom]
#         control = df.loc[control_group, biom]
#         t_value, p_value = stats.ttest_ind(case, control, equal_var=False, nan_policy='omit')
#         # 添加到结果列表
#         results.append({
#             'biom': biom,
#             'subtype': i,
#             't_statistic': t_value,
#             'p_value': p_value
#         })

# # 创建 DataFrame
# t_stats_df = pd.DataFrame(results)

# # 应用 Bonferroni 校正
# t_stats_df['corrected_p_value'] = bonferroni_correction(t_stats_df['p_value'])
# t_stats_df['significant'] = t_stats_df['corrected_p_value'] < alpha

# # 按 biom 和 subtype 排序（可选）
# t_stats_df = t_stats_df.sort_values(['biom', 'subtype'])

# # 保存到 CSV 文件
# output_csv_path = f'{OUTPUT_DIR}/t_stats_by_subtype.csv'
# t_stats_df.to_csv(output_csv_path, index=False)

# # 打印确认信息
# print(f"t-statistics saved to: {output_csv_path}")
# print(f"Total rows: {len(t_stats_df)}")
# print("First few rows of the data:")
# print(t_stats_df.head())

import pandas as pd
import numpy as np
from scipy import stats
import os

def bonferroni_correction(p_values):
    n = len(p_values)
    corrected_p_values = np.array(p_values) * n
    return np.minimum(corrected_p_values, 1.0)  # Cap at 1.0 as p-values can't exceed 1

nsubtype = 5
exp_name = 'ukb_MixtureGRBF_cv_nsubtype_biom17'
INPUT_TABLE_PATH = 'input/ukb/ukb_covreg1_trans1_nanf1_biom0.csv'
SUBTYPE_STAGE_PATH = f'output/{exp_name}/{nsubtype}_subtypes/subtype_stage.csv'
OUTPUT_DIR = f'output/result_analysis/{exp_name}/{nsubtype}_subtypes'
SUBTYPE_ORDER_PATH = f'output/result_analysis/{exp_name}/{nsubtype}_subtypes/all_cause_mortality_order.csv'

df = pd.read_csv(INPUT_TABLE_PATH)
df.rename(columns={'RID':'PTID'}, inplace=True)
subtype_stage = pd.read_csv(SUBTYPE_STAGE_PATH)
try:
    subtype_order = pd.read_csv(SUBTYPE_ORDER_PATH, header=None)
except:
    subtype_order = pd.DataFrame(np.array(list(range(1, nsubtype+1))))
subtype_order_map = {int(subtype_order.iloc[i]): i+1 for i in range(nsubtype)}
subtype_stage['subtype'] = subtype_stage['subtype'].map(subtype_order_map)
biomarker_names = df.iloc[:, 7:].columns.tolist()
df = pd.merge(df, subtype_stage, how='inner', on=['PTID', 'stage'])

alpha = 0.05

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Create list to store all data
results = []

for i in range(1, nsubtype + 1):
    case_group = df['subtype'] == i
    control_group = df['subtype'] != i
    
    # Calculate t statistics and p-values
    for biom in biomarker_names:
        case = df.loc[case_group, biom]
        control = df.loc[control_group, biom]
        t_value, p_value = stats.ttest_ind(case, control, equal_var=False, nan_policy='omit')
        # Add to results list
        results.append({
            'biom': biom,
            'subtype': i,
            't_statistic': t_value,
            'p_value': p_value
        })

# Create DataFrame
t_stats_df = pd.DataFrame(results)

# # Keep only top 100 biomarkers by absolute t-statistic for each subtype
# t_stats_df['abs_t_statistic'] = t_stats_df['t_statistic'].abs()
# # Group by subtype and get top 100 based on absolute t-statistic
# t_stats_df = (t_stats_df.groupby('subtype')
#              .apply(lambda x: x.nlargest(100, 'abs_t_statistic'))
#              .reset_index(drop=True))

# Apply Bonferroni correction
t_stats_df['corrected_p_value'] = bonferroni_correction(t_stats_df['p_value'])
t_stats_df['significant'] = t_stats_df['corrected_p_value'] < alpha

# Sort by biom and subtype (optional)
t_stats_df = t_stats_df.sort_values(['biom', 'subtype'])

# # Drop the temporary abs_t_statistic column
# t_stats_df = t_stats_df.drop('abs_t_statistic', axis=1)

# Save to CSV file
output_csv_path = f'{OUTPUT_DIR}/t_stats_by_subtype.csv'
t_stats_df.to_csv(output_csv_path, index=False)

# Print confirmation
print(f"t-statistics (top 100 per subtype) saved to: {output_csv_path}")
print(f"Total rows: {len(t_stats_df)}")
print("First few rows of the data:")
print(t_stats_df.head())