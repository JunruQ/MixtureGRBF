import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import os
from matplotlib.colors import LinearSegmentedColormap
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

# File paths and parameters
nsubtype = 5
N_TOP_FACTORS = 50
BLOOD_CHEM_PATH = 'data/ClinicalLabData.csv'
OTHER_IND_PATH = 'data/prot_Modifiable_bl_data.csv'
INPUT_TABLE_PATH = 'input/ukb/ukb_covreg1_trans1_nanf1_biom17.csv'
exp_name = 'ukb_MixtureGRBF_cv_nsubtype_biom17'
SUBTYPE_STAGE_PATH = f'output/{exp_name}/{nsubtype}_subtypes/subtype_stage.csv'
SUBTYPE_ORDER_PATH = f'output/result_analysis/{exp_name}/{nsubtype}_subtypes/all_cause_mortality_order.csv'
CLASSIFICATION_PATH = 'analysis/result_analysis/clinical_lab_data_dict.csv'
OUTPUT_DIR = f'output/result_analysis/{exp_name}/{nsubtype}_subtypes'

palette = ['#0072BD', '#D95319', '#EDB120', '#7E2F8E', '#77AC30']
gray = '#777777'

# Load and merge data
df = pd.read_csv(INPUT_TABLE_PATH)
import utils.utils as utils
subtype_stage = utils.get_subtype_stage(exp_name, nsubtype)

df = pd.merge(df, subtype_stage[['PTID', 'subtype']], left_on='RID', right_on='PTID', how='left')
other_ind = pd.read_csv(OTHER_IND_PATH)
df = pd.merge(df, other_ind, how='left', left_on='RID', right_on='eid')

# Define continuous indicators
env_lifestyle_factors = [
    'Duration of walks', 'Usual walking pace', 'IPAQ activity group', 'Summed minutes activity',
    'MET minutes per week for walking', 'MET minutes per week for moderate activity',
    'MET minutes per week for vigorous activity', 'Summed MET minutes per week for all activity',
    'Sleep duration', 'Getting up in morning', 'Chronotype', 'Nap during day', 'Sleeplessness',
    'Snoring', 'Daytime dozing', 'Cooked vegetable intake', 'Raw vegetable intake',
    'Fresh fruit intake', 'Dried fruit intake', 'Oily fish intake', 'Non-oily fish intake',
    'Processed meat intake', 'Poultry intake', 'Beef intake', 'Lamb intake', 'Pork intake',
    'Cheese intake', 'Milk type used', 'Spread type', 'Bread intake', 'Cereal intake',
    'Salt added to food', 'Tea intake', 'Coffee intake', 'Water intake', 'Variation in diet',
    'Alcohol intake frequency', 'Smoking status', 'Alcohol drinker status',
    'Nitrogen dioxide air pollution; 2010', 'Nitrogen oxides air pollution; 2010',
    'Particulate matter air pollution (pm10); 2010', 'Particulate matter air pollution (pm2.5); 2010',
    'Particulate matter air pollution (pm2.5) absorbance; 2010', 'Particulate matter air pollution 2.5-10um; 2010',
    'Traffic intensity on the nearest major road', 'Inverse distance to the nearest major road',
    'Total traffic load on major roads', 'Sum of road length of major roads within 100m',
    'Nitrogen dioxide air pollution; 2005', 'Nitrogen dioxide air pollution; 2006',
    'Nitrogen dioxide air pollution; 2007', 'Particulate matter air pollution (pm10); 2007',
    'Average daytime sound level of noise pollution', 'Average evening sound level of noise pollution',
    'Average night-time sound level of noise pollution', 'Average 16-hour sound level of noise pollution',
    'Average 24-hour sound level of noise pollution', 'Greenspace percentage, buffer 1000m',
    'Domestic garden percentage, buffer 1000m', 'Water percentage, buffer 1000m',
    'Greenspace percentage, buffer 300m', 'Domestic garden percentage, buffer 300m',
    'Water percentage, buffer 300m', 'Natural environment percentage, buffer 1000m',
    'Natural environment percentage, buffer 300m', 'Townsend deprivation index',
    'Average total household income before tax', 'In paid employment or self-employed', 'Retired',
    'Looking after home and/or family', 'Unable to work because of sickness or disability',
    'Unemployed', 'Doing unpaid or voluntary work', 'Full or part-time student',
    'Vitamin A', 'Vitamin B', 'Vitamin C', 'Vitamin D_ukb_Modifiable_categorical_bl_data',
    'Vitamin E', 'Folic acid', 'Multivitamins', 'Fish oil', 'Glucosamine',
    'Calcium_ukb_Modifiable_categorical_bl_data', 'Zinc', 'Iron', 'Selenium'
]

life_events = [
    'Comparative body size at age 10', 'Comparative height size at age 10', 'Maternal smoking around birth',
    'Age first had sexual intercourse', 'Ever had same-sex intercourse', 'College or University degree',
    'A levels/AS levels or equivalent', 'O levels/GCSEs or equivalent', 'CSEs or equivalent',
    'NVQ or HND or HNC or equivalent', 'Other professional qualifications', 'Husband, wife or partner',
    'Son and/or daughter', 'Brother and/or sister', 'Mother and/or father', 'Grandparent', 'Grandchild',
    'Other related'
]

all_indicators = env_lifestyle_factors + life_events

categorical_non_ordinal = [
    'Milk type used', 'Spread type'
]
continuous_indicators = [ind for ind in all_indicators if ind not in categorical_non_ordinal and ind in df.columns]

# Define subcategories
subcategories = [
    'Physical activity', 'Sleep', 'Diet', 'Substance use', 'Supplement use',
    'Air pollution', 'Traffic exposure', 'noise pollution', 'Natural environment',
    'Wealth', 'Employment status', 'Early life', 'Sexual history',
    'Education', 'Family relationships'
]

# Map indicators to their subcategories
factor_subcategory_map = {
    'Duration of walks': 'Physical activity',
    'Usual walking pace': 'Physical activity',
    'IPAQ activity group': 'Physical activity',
    'Summed minutes activity': 'Physical activity',
    'MET minutes per week for walking': 'Physical activity',
    'MET minutes per week for moderate activity': 'Physical activity',
    'MET minutes per week for vigorous activity': 'Physical activity',
    'Summed MET minutes per week for all activity': 'Physical activity',
    'Sleep duration': 'Sleep',
    'Getting up in morning': 'Sleep',
    'Chronotype': 'Sleep',
    'Nap during day': 'Sleep',
    'Sleeplessness': 'Sleep',
    'Snoring': 'Sleep',
    'Daytime dozing': 'Sleep',
    'Cooked vegetable intake': 'Diet',
    'Raw vegetable intake': 'Diet',
    'Fresh fruit intake': 'Diet',
    'Dried fruit intake': 'Diet',
    'Oily fish intake': 'Diet',
    'Non-oily fish intake': 'Diet',
    'Processed meat intake': 'Diet',
    'Poultry intake': 'Diet',
    'Beef intake': 'Diet',
    'Lamb intake': 'Diet',
    'Pork intake': 'Diet',
    'Cheese intake': 'Diet',
    'Milk type used': 'Diet',
    'Spread type': 'Diet',
    'Bread intake': 'Diet',
    'Cereal intake': 'Diet',
    'Salt added to food': 'Diet',
    'Tea intake': 'Diet',
    'Coffee intake': 'Diet',
    'Water intake': 'Diet',
    'Variation in diet': 'Diet',
    'Alcohol intake frequency': 'Substance use',
    'Alcohol drinker status': 'Substance use',
    'Smoking status': 'Substance use',
    'Vitamin A': 'Supplement use',
    'Vitamin B': 'Supplement use',
    'Vitamin C': 'Supplement use',
    'Vitamin D_ukb_Modifiable_categorical_bl_data': 'Supplement use',
    'Vitamin E': 'Supplement use',
    'Folic acid': 'Supplement use',
    'Multivitamins': 'Supplement use',
    'Fish oil': 'Supplement use',
    'Glucosamine': 'Supplement use',
    'Calcium_ukb_Modifiable_categorical_bl_data': 'Supplement use',
    'Zinc': 'Supplement use',
    'Iron': 'Supplement use',
    'Selenium': 'Supplement use',
    'Nitrogen dioxide air pollution; 2010': 'Air pollution',
    'Nitrogen oxides air pollution; 2010': 'Air pollution',
    'Particulate matter air pollution (pm10); 2010': 'Air pollution',
    'Particulate matter air pollution (pm2.5); 2010': 'Air pollution',
    'Particulate matter air pollution (pm2.5) absorbance; 2010': 'Air pollution',
    'Particulate matter air pollution 2.5-10um; 2010': 'Air pollution',
    'Nitrogen dioxide air pollution; 2005': 'Air pollution',
    'Nitrogen dioxide air pollution; 2006': 'Air pollution',
    'Nitrogen dioxide air pollution; 2007': 'Air pollution',
    'Particulate matter air pollution (pm10); 2007': 'Air pollution',
    'Traffic intensity on the nearest major road': 'Traffic exposure',
    'Inverse distance to the nearest major road': 'Traffic exposure',
    'Total traffic load on major roads': 'Traffic exposure',
    'Sum of road length of major roads within 100m': 'Traffic exposure',
    'Average daytime sound level of noise pollution': 'noise pollution',
    'Average evening sound level of noise pollution': 'noise pollution',
    'Average night-time sound level of noise pollution': 'noise pollution',
    'Average 16-hour sound level of noise pollution': 'noise pollution',
    'Average 24-hour sound level of noise pollution': 'noise pollution',
    'Greenspace percentage, buffer 1000m': 'Natural environment',
    'Domestic garden percentage, buffer 1000m': 'Natural environment',
    'Water percentage, buffer 1000m': 'Natural environment',
    'Greenspace percentage, buffer 300m': 'Natural environment',
    'Domestic garden percentage, buffer 300m': 'Natural environment',
    'Water percentage, buffer 300m': 'Natural environment',
    'Natural environment percentage, buffer 1000m': 'Natural environment',
    'Natural environment percentage, buffer 300m': 'Natural environment',
    'Townsend deprivation index': 'Wealth',
    'Average total household income before tax': 'Wealth',
    'In paid employment or self-employed': 'Employment status',
    'Retired': 'Employment status',
    'Looking after home and/or family': 'Employment status',
    'Unable to work because of sickness or disability': 'Employment status',
    'Unemployed': 'Employment status',
    'Doing unpaid or voluntary work': 'Employment status',
    'Full or part-time student': 'Employment status',
    'Comparative body size at age 10': 'Early life',
    'Comparative height size at age 10': 'Early life',
    'Maternal smoking around birth': 'Early life',
    'Age first had sexual intercourse': 'Sexual history',
    'Ever had same-sex intercourse': 'Sexual history',
    'College or University degree': 'Education',
    'A levels/AS levels or equivalent': 'Education',
    'O levels/GCSEs or equivalent': 'Education',
    'CSEs or equivalent': 'Education',
    'NVQ or HND or HNC or equivalent': 'Education',
    'Other professional qualifications': 'Education',
    'Husband, wife or partner': 'Family relationships',
    'Son and/or daughter': 'Family relationships',
    'Brother and/or sister': 'Family relationships',
    'Mother and/or father': 'Family relationships',
    'Grandparent': 'Family relationships',
    'Grandchild': 'Family relationships',
    'Other related': 'Family relationships'
}

# Define aliases for long indicator names
indicator_aliases = {
    'MET minutes per week for walking': 'Walk MET min/week',
    'MET minutes per week for moderate activity': 'Moderate MET min/week',
    'MET minutes per week for vigorous activity': 'Vigorous MET min/week',
    'Summed MET minutes per week for all activity': 'Total MET min/week',
    'Nitrogen dioxide air pollution; 2010': 'NO2 pollution 2010',
    'Nitrogen oxides air pollution; 2010': 'NOx pollution 2010',
    'Particulate matter air pollution (pm10); 2010': 'PM10 pollution 2010',
    'Particulate matter air pollution (pm2.5); 2010': 'PM2.5 pollution 2010',
    'Particulate matter air pollution (pm2.5) absorbance; 2010': 'PM2.5 absorbance 2010',
    'Particulate matter air pollution 2.5-10um; 2010': 'PM2.5-10um pollution 2010',
    'Nitrogen dioxide air pollution; 2005': 'NO2 pollution 2005',
    'Nitrogen dioxide air pollution; 2006': 'NO2 pollution 2006',
    'Nitrogen dioxide air pollution; 2007': 'NO2 pollution 2007',
    'Particulate matter air pollution (pm10); 2007': 'PM10 pollution 2007',
    'Traffic intensity on the nearest major road': 'Traffic intensity',
    'Inverse distance to the nearest major road': 'Road proximity',
    'Total traffic load on major roads': 'Traffic load',
    'Sum of road length of major roads within 100m': 'Road length 100m',
    'Average daytime sound level of noise pollution': 'Day noise level',
    'Average evening sound level of noise pollution': 'Evening noise level',
    'Average night-time sound level of noise pollution': 'Night noise level',
    'Average 16-hour sound level of noise pollution': '16hr noise level',
    'Average 24-hour sound level of noise pollution': '24hr noise level',
    'Greenspace percentage, buffer 1000m': 'Greenspace 1000m',
    'Domestic garden percentage, buffer 1000m': 'Garden 1000m',
    'Water percentage, buffer 1000m': 'Water 1000m',
    'Greenspace percentage, buffer 300m': 'Greenspace 300m',
    'Domestic garden percentage, buffer 300m': 'Garden 300m',
    'Water percentage, buffer 300m': 'Water 300m',
    'Natural environment percentage, buffer 1000m': 'Natural env 1000m',
    'Natural environment percentage, buffer 300m': 'Natural env 300m',
    'Average total household income before tax': 'Household income',
    'In paid employment or self-employed': 'Employed',
    'Unable to work because of sickness or disability': 'Unable to work',
    'Doing unpaid or voluntary work': 'Voluntary work',
    'Comparative body size at age 10': 'Body size age 10',
    'Comparative height size at age 10': 'Height age 10',
    'Maternal smoking around birth': 'Maternal smoking',
    'Age first had sexual intercourse': 'Age first sex',
    'Ever had same-sex intercourse': 'Same-sex intercourse',
    'College or University degree': 'University degree',
    'A levels/AS levels or equivalent': 'A levels',
    'O levels/GCSEs or equivalent': 'GCSEs',
    'NVQ or HND or HNC or equivalent': 'NVQ/HND/HNC',
    'Other professional qualifications': 'Other qualifications',
    'Calcium_ukb_Modifiable_categorical_bl_data': 'Calcium',
    'Vitamin D_ukb_Modifiable_categorical_bl_data': 'Vitamin D'
}

df_raw = df[continuous_indicators + ['subtype', 'RID']].copy()

# Standardize continuous variables
scaler = StandardScaler()
for col in continuous_indicators:
    df[col] = scaler.fit_transform(df[[col]].fillna(df[col].mean()))

# Create dummy variables for subtypes
subtype_cols = {f'subtype_{i}': (df['subtype'] == i).astype(int) for i in range(1, nsubtype + 1)}
subtype_df = pd.DataFrame(subtype_cols, index=df.index)
df = pd.concat([df, subtype_df], axis=1)

# Regression analysis
results = []
for ind in continuous_indicators:
    for subtype in range(1, nsubtype + 1):
        X = df[['sex', 'stage', f'subtype_{subtype}']].dropna()
        y = df[ind].reindex(X.index)
        X = sm.add_constant(X)
        model = sm.OLS(y, X).fit()
        var = f'subtype_{subtype}'
        coef = model.params[var]
        pval = model.pvalues[var]
        results.append({
            'Indicator': ind,
            'Subtype': subtype,
            'Coefficient': coef,
            'P-value': pval
        })

# Convert results to DataFrame
results_df = pd.DataFrame(results)
print(len(results_df['P-value']))
# FDR correction 
from statsmodels.stats.multitest import multipletests
results_df['FDR'] = multipletests(results_df['P-value'], method='fdr_bh')[1]
results_df['-log10_p'] = -np.log10(results_df['FDR'].replace(0, np.finfo(float).eps))
results_df['Sign'] = np.where(results_df['Coefficient'] > 0, 1, -1)
results_df['Signed_-log10_p'] = results_df['-log10_p'] * results_df['Sign']

from collections import defaultdict

# 2. 提取子类的出现顺序（保持顺序且去重）
subcategories_order = []
for sub in factor_subcategory_map.values():
    if sub not in subcategories_order:
        subcategories_order.append(sub)

# 3. 将项按子类归类
grouped_data = defaultdict(list)
for factor, sub in factor_subcategory_map.items():
    grouped_data[sub].append(factor)

# 4. 按照子类顺序，对每类中的项进行字母排序，并合并到结果列表
sorted_factor_list = []
for sub in subcategories_order:
    # 对当前子类下的小项进行排序
    current_group_sorted = sorted(grouped_data[sub], key=lambda x: indicator_aliases.get(x, x))
    # 添加到最终列表
    sorted_factor_list.extend(current_group_sorted)

top_indicators = results_df.groupby('Indicator').max()['-log10_p'].nlargest(N_TOP_FACTORS).index.tolist()

top_indicators = [i for i in sorted_factor_list if i in top_indicators]

# Filter continuous_indicators to top N
continuous_indicators = [ind for ind in continuous_indicators if ind in top_indicators]

# Replace indicator names with aliases in results_df
results_df['Indicator'] = results_df['Indicator'].map(lambda x: indicator_aliases.get(x, x))

# Create custom colormap
colors = ['#2980b9', '#ffffff', '#c0392b']
n_bins = 256
cmap = LinearSegmentedColormap.from_list('custom_diverging', colors, N=n_bins)

# Estimate figure height based on number of indicators per subcategory
indicators_per_subcategory = {
    subcategory: [ind for ind, subcat in factor_subcategory_map.items()
                  if subcat == subcategory and ind in continuous_indicators]
    for subcategory in subcategories
}

# Filter out subcategories with no valid indicators
indicators_per_subcategory = {k: v for k, v in indicators_per_subcategory.items() if v}

# Calculate the number of indicators for each subcategory
num_indicators_per_subcategory = [len(inds) for inds in indicators_per_subcategory.values()]

# Calculate the total number of indicators
total_indicators = sum(num_indicators_per_subcategory)

# Determine the height ratio for each subplot
height_ratios = [num_inds if num_inds > 0 else 0.1 for num_inds in num_indicators_per_subcategory]

# Estimate figure height: 0.4 cm per indicator, converted to inches (1 cm = 0.393701 inches)
fig_height = total_indicators * 0.75 * 0.393701
fig_width = 11 / 2.54  # Width in inches

import math
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import re

# =========================
# 0. 路径与辅助函数准备
# =========================

def read_csv_smart(path):
    encodings = ["utf-8-sig", "utf-8", "gbk", "gb18030"]
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"无法读取文件: {path}")

ylab_df = read_csv_smart('output/result_analysis/ukb_MixtureGRBF_cv_nsubtype_biom17/5_subtypes/environment_factor_label.csv')

# --- 解析 Ylabel.csv ---
cat_pattern = re.compile(r"^\s*([+-]?\d+)\s*[:：]\s*(.+?)\s*$")

ylab_df["phenotype"] = ylab_df["phenotype"].astype(str).str.strip()
ylab_df["type"] = ylab_df["type"].astype(str).str.strip().str.lower()

mapping_cols = []
for c in ylab_df.columns:
    if re.fullmatch(r"[A-Z]", str(c).strip()):
        mapping_cols.append(c)

ylab_info = {}
for _, row in ylab_df.iterrows():
    p = str(row["phenotype"]).strip()
    t = str(row["type"]).strip().lower()

    unit = ""
    if "A" in ylab_df.columns and pd.notna(row.get("A", np.nan)) and t == "numerical":
        unit = str(row.get("A", "")).strip()

    cat_map = {}
    if t == "category":
        for c in mapping_cols:
            v = row.get(c, np.nan)
            if pd.isna(v):
                continue
            s = str(v).strip()
            if not s:
                continue
            m = cat_pattern.match(s)
            if not m:
                continue
            code = int(m.group(1))
            label = m.group(2).strip()
            cat_map[code] = label

    ylab_info[p] = {"type": t, "unit": unit, "cat_map": cat_map}


# --- 文本换行辅助函数 ---
def wrap_label(label: str, threshold: int = 8) -> str:
    """如果标签超过阈值，尝试在中间空格处折行"""
    if not label or len(str(label)) <= threshold:
        return str(label)
    s = str(label)
    # 找最靠近中间的空格
    middle = len(s) // 2
    best_space = -1
    min_dist = len(s)
    
    for i, char in enumerate(s):
        if char == ' ':
            dist = abs(i - middle)
            if dist < min_dist:
                min_dist = dist
                best_space = i
                
    if best_space != -1:
        return s[:best_space] + '\n' + s[best_space+1:]
    else:
        # 如果没有空格，强制在中间折断
        return s[:middle] + '\n' + s[middle:]

# =========================
# 1. 绘图配置参数
# =========================
N_COLS = 5             # 每行显示6个小图
FIG_WIDTH = 20/2.54         # 画布总宽

# 散点设置
POINTS_SIZE = 2.
POINTS_ALPHA = 0.2
MAX_POINTS_DISPLAY = 300  # 关键：为了避免点太多导致重叠成一团黑或渲染太慢，每个subtype随机抽样显示的点的最大数量
Y_JITTER_AMOUNT = 0.5

# 简约 Boxplot 样式
BOX_PROPS = {'facecolor': 'none', 'edgecolor': '#333333', 'linewidth': 1.2}
WHISKER_PROPS = {'color': '#333333', 'linewidth': 1.2}
CAP_PROPS = {'color': '#333333', 'linewidth': 1.2}
MEDIAN_PROPS = {'color': 'red', 'linewidth': 1.5} # 中位数线设为红色突出显示，或者改为 '#333333'

# 字体
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = False
rng = np.random.default_rng(42) # 固定随机种子

# =========================
# 2. 数据重排与布局计算 (修改部分)
# =========================

# 2.1 定义希望排在第一列的指标 (优先级列表)
target_first_col = [
    'Chronotype',
    'Daytime dozing',
    'Beef intake', 
    'Cheese intake',
    'Lamb intake',
    'Oily fish intake', 
    'Pork intake', 
    'Poultry intake', 
    'Processed meat intake',
    'Average total household income before tax'
]

# 2.2 将原始指标分为两组
# 确保 target_first_col 中的指标确实存在于 top_indicators 中
priority_inds = [ind for ind in target_first_col if ind in top_indicators]
# 剩下的指标
other_inds = [ind for ind in top_indicators if ind not in priority_inds]

# 2.3 计算行数
# 行数由两个因素决定：总指标数不够放，或者第一列的指标太多放不下
# 必须取两者的最大值，确保第一列能放得下 priority_inds
n_total = len(top_indicators)
n_rows_by_count = math.ceil(n_total / N_COLS)
n_rows_by_priority = len(priority_inds)
n_rows = max(n_rows_by_count, n_rows_by_priority)

# 2.4 构建符合网格布局的绘图顺序列表
# 创建一个长度为 grid 大小的空列表
ordered_indicators = [None] * (n_rows * N_COLS)

p_idx = 0 # 优先组指针
o_idx = 0 # 普通组指针

for r in range(n_rows):
    for c in range(N_COLS):
        flat_idx = r * N_COLS + c
        
        if c == 0:
            # --- 第一列 ---
            if p_idx < len(priority_inds):
                ordered_indicators[flat_idx] = priority_inds[p_idx]
                p_idx += 1
            elif o_idx < len(other_inds):
                # 如果优先指标放完了，第一列剩下的位置放普通指标
                ordered_indicators[flat_idx] = other_inds[o_idx]
                o_idx += 1
        else:
            # --- 其他列 ---
            if o_idx < len(other_inds):
                ordered_indicators[flat_idx] = other_inds[o_idx]
                o_idx += 1

# 去除列表末尾多余的 None (如果网格比指标多)
# 注意：中间的 None (如果有空缺) 应该保留以免索引错位，但上面的逻辑通常是紧凑填充的
# 这里我们只把列表截断到实际需要的长度，或者保留 None 在循环里跳过
final_plot_list = ordered_indicators

# 重新计算画布高度
fig_height = 44 / 2.54 # 或者根据 n_rows 动态计算: n_rows * ROW_HEIGHT
# 如果行数变多了，建议动态调整高度：
# fig_height = n_rows * 3.5  (假设每行高 3.5 inch)

fig, axes = plt.subplots(n_rows, N_COLS, figsize=(FIG_WIDTH, fig_height), dpi=150)
axes = axes.flatten()

import matplotlib.colors as mcolors

# 1. 预先计算全局 count 的最大值和最小值
all_counts = []
for ind in final_plot_list:
    temp_data = df_raw[['subtype', ind]].dropna()
    if not temp_data.empty:
        counts = temp_data.groupby(['subtype', ind]).size()
        all_counts.extend(counts.values)

global_min = 0
global_max = max(all_counts) if all_counts else 1

print(f"Global Min: {global_min}, Global Max: {global_max}")

print(np.log1p(global_max), np.log1p(global_min))

# norm = mcolors.Normalize(vmin=np.log1p(global_min), vmax=np.log1p(global_max))
norm = mcolors.Normalize(vmin=global_min, vmax=global_max)

global_max_proportion = 0

for i, ind in enumerate(final_plot_list):
    ax = axes[i]
    
    info = ylab_info.get(ind, {'type': 'numerical', 'unit': '', 'cat_map': {}})
    ptype = info['type']
    unit = info['unit']
    cat_map = info['cat_map']
    # print("Processing:", ind, ptype, unit, cat_map)

    # 原始数据（用于 boxplot 和 确定刻度）
    plot_data = df_raw[['subtype', ind]].dropna()
    
    if plot_data.empty:
        ax.text(0.5, 0.5, 'No Data', ha='center', va='center')
        continue

    # --- 3.1 离散变量聚合绘图 (针对 category 变量) ---
    if ptype == 'category' and cat_map:
        # 1. 统计频数和比例
        # counts: 每种 subtype 下，每个 ind 类别的数量
        stats = plot_data.groupby(['subtype', ind]).size().reset_index(name='count')
        
        # 计算比例：该类别在当前 subtype 中的占比
        total_per_subtype = plot_data.groupby('subtype').size()
        stats['proportion'] = stats.apply(lambda row: row['count'] / total_per_subtype[row['subtype']], axis=1)

        # 2. 绘制聚合散点图
        # s: 点的大小，由比例决定 (乘以一个缩放因子如 500)
        # c: 点的颜色深浅，由 count 决定
        global_max_proportion = max([global_max_proportion, max(stats['proportion'])])

        
        scatter = ax.scatter(
            x=stats['subtype'], 
            y=stats[ind], 
            s=stats['proportion'] * 100,
            # c=np.log1p(stats['count']), 
            c=stats['count'], 
            norm=norm,
            cmap='Blues',   
            edgecolors='#0072BD', 
            # edgecolors='black',
            alpha=0.8,
            zorder=5
        )

        ax.set_xticks([1,2,3,4,5])
        ax.set_xlim(0, 6)
        
        # 如果需要显示颜色条（可选，但子图多时建议不显示或统一显示）
        # plt.colorbar(scatter, ax=ax)

    else:
        # --- 原有的数值型变量逻辑 (Stripplot + Boxplot) ---
        sampled_dfs = []
        for st in sorted(plot_data['subtype'].unique()):
            sub_df = plot_data[plot_data['subtype'] == st]
            if len(sub_df) > MAX_POINTS_DISPLAY:
                sub_df = sub_df.sample(n=MAX_POINTS_DISPLAY, random_state=42)
            sampled_dfs.append(sub_df)
        
        if sampled_dfs:
            scatter_data = pd.concat(sampled_dfs).copy()
            y_noise = rng.uniform(-Y_JITTER_AMOUNT, Y_JITTER_AMOUNT, size=len(scatter_data))
            scatter_data[ind] = scatter_data[ind] + y_noise
            # 数值型变量保留微量抖动
            sns.stripplot(
                data=scatter_data, x='subtype', y=ind, color='#0072BD',
                size=POINTS_SIZE, alpha=POINTS_ALPHA,
                jitter=0.25, zorder=0, ax=ax, legend=False
            )
        
        ax.set_xlim(-1, 5)

    if ptype == 'numerical':
        sns.boxplot(
            data=plot_data, x='subtype', y=ind,
            showfliers=False, width=0.5,
            boxprops=BOX_PROPS, whiskerprops=WHISKER_PROPS,
            capprops=CAP_PROPS, medianprops=MEDIAN_PROPS,
            zorder=10, ax=ax
        )
    

    # --- 3.3 坐标轴与标签美化 ---
    title_text = indicator_aliases.get(ind, ind)
    if len(title_text) > 30: title_text = title_text[:28] + '...' 
    ax.set_title(title_text, fontsize=9, pad=-5)

    n_count = len(plot_data)

    ax.set_title(f"{title_text}\n(n={n_count:,})", fontsize=9, pad=10)
    
    ax.set_xlabel('')
    ax.set_ylabel('') # 默认清空
    
    sns.despine(ax=ax, top=True, right=True)
    # ax.set_xticklabels([f'S{int(x)}' for x in sorted(plot_data['subtype'].unique())], fontsize=9)

    # --- Y轴处理逻辑 ---
    if ptype == 'category' and cat_map:
        # 1. 锁定 Y 轴范围，防止散点抖动导致范围扩大太多，稍微留点余地即可
        codes_present = sorted(plot_data[ind].unique())
        if codes_present:
            # 设置刻度为精确的整数
            ax.set_yticks(codes_present)
            # 生成标签
            if 'intake' in title_text or 'income' in title_text:
                labels = [wrap_label(cat_map.get(c, str(c)), threshold=18) for c in codes_present]
            else:
                labels = [wrap_label(cat_map.get(c, str(c)), threshold=12) for c in codes_present]
            ax.set_yticklabels(labels, fontsize=8)
            # 限制视图范围，确保不会因为 jitter 显得太空或被切掉
            # 范围设为: min-0.5 到 max+0.5，这样整数刻度正好在中间
            ax.set_ylim(min(codes_present) - 0.6, max(codes_present) + 0.6)
            
        ax.grid(axis='y', linestyle=':', alpha=0.2)

    else:
        # --- Numerical 变量的异常值处理与坐标轴设置 ---
        ax.tick_params(axis='y', labelsize=8)
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        
        # # 计算 IQR 以移除极端离群值的显示
        # q1 = plot_data[ind].quantile(0.25)
        # q3 = plot_data[ind].quantile(0.75)
        # iqr = q3 - q1
        
        # # 定义显示范围：通常为 [Q1 - 1.5*IQR, Q3 + 1.5*IQR]
        # # 增加 10% 的 margin (0.1) 让箱线图的须线不会贴在坐标轴边缘
        # lower_bound = q1 - 1.5 * iqr
        # upper_bound = q3 + 1.5 * iqr
        
        # # 计算实际数据的最小值和最大值（在 non-outlier 范围内）
        # # 这样可以防止 ylim 设置得比实际数据范围还大
        # actual_min = plot_data[ind][plot_data[ind] >= lower_bound].min()
        # actual_max = plot_data[ind][plot_data[ind] <= upper_bound].max()
        
        # # 容错处理：如果计算失败（数据太集中），则使用原始范围
        # if pd.notna(actual_min) and pd.notna(actual_max) and actual_min != actual_max:
        #     margin = (actual_max - actual_min) * 0.1
        #     ax.set_ylim(actual_min - margin, actual_max + margin)
        
        ylabel_text = unit if unit else 'Value'
        ax.set_ylabel(ylabel_text, fontsize=9, color='black')

print(f"Global Max Proportion: {global_max_proportion}")
# =========================
# 4. 清理多余子图并保存
# =========================
# 如果 grid 数量多于指标数量，隐藏剩下的空图
for j in range(i + 1, len(axes)):
    axes[j].axis('off')

plt.tight_layout()
# 增加一点顶部边距给总标题
# plt.subplots_adjust(top=0.97, hspace=0.4, wspace=0.25)
plt.subplots_adjust(
    top=0.96, 
    bottom=0.05,
    hspace=0.8,
    wspace=1
)

out_file = os.path.join(OUTPUT_DIR, 'environment_factors_boxplot.png')
plt.savefig(out_file, bbox_inches='tight', dpi=500)

print(f"Fig saved: {out_file}")


