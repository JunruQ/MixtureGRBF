# analysis/result_analysis/demographic_form.py
import pandas as pd
import utils.utils as utils
import os
nsubtype = 5
result_folder = 'ukb_MixtureGRBF_cv_nsubtype_biom17'
output_dir = f'./output/result_analysis/{result_folder}/{nsubtype}_subtypes'

subtype_stage = utils.get_subtype_stage_with_cov(result_folder, nsubtype=nsubtype, subtype_order=True)
subtype_stage = subtype_stage.rename(columns={'PTID': 'eid'})

cov1_path = 'data/ukb_cov_info.csv'
cov1_columns = ['eid', 'Townsend', 'Smoking', 'Alcohol', 'BMI']
cov1_df = pd.read_csv(cov1_path, usecols=cov1_columns)

cov2_path = 'data/prot_Modifiable_bl_data.csv'
cov2_columns = ['eid', 'Average total household income before tax']
cov2_df = pd.read_csv(cov2_path, usecols=cov2_columns)

subtype_stage = subtype_stage.merge(cov1_df, on='eid', how='left')
subtype_stage = subtype_stage.merge(cov2_df, on='eid', how='left')


# --- 统计分析代码 ---

# 定义变量列名
continuous_vars = {
    'stage': 'Age', 
    'education': 'Education', 
    'Townsend': 'TDI', 
    'BMI': 'BMI'
}
sex_var = 'sex'
categorical_vars = {
    'Ethnic': 'Ethnicity', 
    'centre': 'Site', 
    'Smoking': 'Smoking Status', 
    'Alcohol': 'Alcohol Intake',
    'Average total household income before tax': 'Household Income'
}

ethnic_mapping = {
    '1': 'White',
    '3': 'Asian',
    '4': 'Black',
    '6': 'Mixed',
}

location_region_map = {
    # England - North
    '11008': 'North England',       # Bury
    '11024': 'North England',       # Cheadle (revisit)
    '11010': 'North England',       # Leeds
    '11016': 'North England',       # Liverpool
    '11001': 'North England',       # Manchester
    '11017': 'North England',       # Middlesborough
    '11009': 'North England',       # Newcastle
    '11014': 'North England',       # Sheffield
    '10003': 'North England',       # Stockport (pilot)

    # England - Midlands
    '11021': 'Midlands',            # Birmingham
    '11013': 'Midlands',            # Nottingham
    '11006': 'Midlands',            # Stoke

    # England - South
    '11012': 'South England',       # Barts
    '11011': 'South England',       # Bristol
    '11020': 'South England',       # Croydon
    '11018': 'South England',       # Hounslow
    '11002': 'South England',       # Oxford
    '11007': 'South England',       # Reading

    # Scotland
    '11005': 'Scotland',              # Edinburgh
    '11004': 'Scotland',              # Glasgow

    # Wales
    '11003': 'Wales',                 # Cardiff
    '11022': 'Wales',                 # Swansea
    '11023': 'Wales',                 # Wrexham
}

sml_alc_mapping = {
    '0': 'Never',
    '1': 'Previous',
    '2': 'Current',
    '-3': 'N/A',    
}

household_income_mapping = {
    '1': 'Less than £18,000',
    '2': '£18,000 to £31,000',
    '3': '£31,000 to £52,000',
    '4': '£52,000 to £100,000',
    '5': 'Greater than £100,000',
    '-3': 'N/A',
}

subtype_stage['Ethnic'] = subtype_stage['Ethnic'].astype(str).map(ethnic_mapping)
subtype_stage['centre'] = subtype_stage['centre'].astype(str).map(location_region_map)
subtype_stage['Smoking'] = subtype_stage['Smoking'].fillna(-3).astype(int).astype(str).map(sml_alc_mapping)
subtype_stage['Alcohol'] = subtype_stage['Alcohol'].fillna(-3).astype(int).astype(str).map(sml_alc_mapping)
subtype_stage['Average total household income before tax'] = subtype_stage['Average total household income before tax'].fillna(-3).astype(int).astype(str).map(household_income_mapping)

# 获取所有亚型的有序列表
subtypes = sorted(subtype_stage['subtype'].unique())
# 创建一个字典来存储最终的统计数据
stats_dict = {}

# 按亚型进行分组
grouped = subtype_stage.groupby('subtype')

# 1. 统计总人数 (N)
n_counts = grouped.size().to_dict()
stats_dict[('Subjects', '')] = {f'Subtype {s}': f"{n_counts[s]} ({n_counts[s] / sum(n_counts.values()) * 100:.1f})" for s in subtypes}

# 2. 统计连续变量 (mean ± std)
for col, name in continuous_vars.items():
    means = grouped[col].mean()
    stds = grouped[col].std()
    stats_dict[(name, 'mean ± std')] = {
        f'Subtype {s}': f"{means.get(s, 0):.1f} ± {stds.get(s, 0):.1f}" for s in subtypes
    }

# 3. 统计性别 (n (%))
if sex_var in subtype_stage.columns:
    # 统计值为1的数量
    sex_counts = grouped[sex_var].apply(lambda x: (x == 1).sum())
    # 统计值为1的比例
    sex_ratios = grouped[sex_var].apply(lambda x: (x == 1).mean())
    stats_dict[('Males', 'n (%)')] = {
        f'Subtype {s}': f"{sex_counts.get(s, 0)} ({sex_ratios.get(s, 0)*100:.1f})" for s in subtypes
    }

# 4. 统计分类变量 (n (%))
for col, name in categorical_vars.items():
    # 使用crosstab计算每个类别的数量和比例
    counts = pd.crosstab(subtype_stage['subtype'], subtype_stage[col])
    proportions = pd.crosstab(subtype_stage['subtype'], subtype_stage[col], normalize='index') * 100
    
    # 获取此变量的所有可能类别
    categories = subtype_stage[col].dropna().unique()
    
    # 为分类变量添加一个主标题行
    stats_dict[(name, '')] = {f'Subtype {s}': '' for s in subtypes}
    
    for category in sorted(categories):
        stats_dict[(name, category)] = {
            f'Subtype {s}': f"{counts.loc[s, category] if category in counts.columns else 0} ({proportions.loc[s, category] if category in proportions.columns else 0:.1f})"
            for s in subtypes
        }


# --- 格式化和输出 ---

# 将字典转换为DataFrame
stats_df = pd.DataFrame.from_dict(stats_dict, orient='index')

# 为索引设置名称
stats_df.index = pd.MultiIndex.from_tuples(stats_df.index, names=['Variable', 'Category'])

# 5. 输出到CSV文件
output_csv_path = os.path.join(output_dir, 'descriptive_stats.csv')
stats_df.to_csv(output_csv_path)
print(f"\n✅ 描述性统计结果已成功保存到: {output_csv_path}")


# 6. 在终端中打印美观的表格
print("\n--- 描述性统计结果 ---")
try:
    from tabulate import tabulate
    # 为了更好的显示效果，将多重索引扁平化
    printable_df = stats_df.reset_index()
    print(tabulate(printable_df, headers='keys', tablefmt='grid', showindex=False))
except ImportError:
    print("建议安装 'tabulate' 库 (pip install tabulate) 以获得更好的终端输出效果。")
    print(stats_df.to_string())
