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

continuous_indicators = [
    ind for ind in continuous_indicators
    if ind in df.columns
    and ind in factor_subcategory_map
]

# Standardize continuous variables
scaler = StandardScaler()
for col in continuous_indicators:
    df[col] = scaler.fit_transform(df[[col]].fillna(df[col].mean()))

# Create dummy variables for subtypes
subtype_cols = {f'subtype_{i}': (df['subtype'] == i).astype(int) for i in range(1, nsubtype + 1)}
subtype_df = pd.DataFrame(subtype_cols, index=df.index)
df = pd.concat([df, subtype_df], axis=1)

# 2. 准备协变量 (Covariates)
# 仿照示例：区分连续型和需要 One-hot 的分类型协变量
continuous_covs = ['sex', 'stage']
categorical_covs = ['centre']  # 如果你有中心效应的话

# 处理分类协变量的 Dummy Encoding
covariates = df[continuous_covs + categorical_covs].copy()
if len(categorical_covs) > 0:
    categorical_encoded = pd.get_dummies(covariates[categorical_covs], drop_first=True, dtype=int)
    covariates_final = pd.concat([covariates[continuous_covs], categorical_encoded], axis=1)
else:
    covariates_final = covariates[continuous_covs]

# 3. 循环回归分析
final_results = []
subtypes = range(1, nsubtype + 1)

for ind in continuous_indicators:
    y = df[ind].astype(float)
    
    for sub in subtypes:
        # 生成当前亚型的指示变量 (1 vs Others)
        subtype_indicator = (df['subtype'] == sub).astype(int).rename(f'subtype_{sub}')
        
        # 构建设计矩阵 X: 亚型指示变量 + 协变量
        X = pd.concat([subtype_indicator, covariates_final], axis=1)
        X = sm.add_constant(X)
        
        # 严格对齐 y 和 X，并剔除缺失值
        mask = y.notna() & X.notna().all(axis=1)
        y_clean = y[mask]
        X_clean = X[mask]
        
        # 鲁棒性检查：确保样本量大于特征数
        if len(y_clean) < X_clean.shape[1] + 2:
            final_results.append({
                'Indicator': ind,
                'Subtype': sub,
                'P-value': 1.0,
                'Coefficient': 0.0,
                'Sign': 0.0
            })
            continue
            
        # 拟合 OLS 模型
        try:
            model = sm.OLS(y_clean, X_clean).fit()
            
            # 提取当前亚型列的统计量
            target_col = f'subtype_{sub}'
            coef = model.params[target_col]
            pval = model.pvalues[target_col]
            sign = np.sign(coef)
            
            final_results.append({
                'Indicator': ind,
                'Subtype': sub,
                'Coefficient': coef,
                'P-value': pval,
                'Sign': sign
            })
        except Exception as e:
            print(f"Error fitting {ind} for subtype {sub}: {e}")

results_df = pd.DataFrame(final_results)
print(len(results_df['P-value']))
# FDR correction 
from statsmodels.stats.multitest import multipletests
results_df['FDR'] = multipletests(results_df['P-value'], method='fdr_bh')[1]
results_df['-log10_p'] = -np.log10(results_df['FDR'].replace(0, np.finfo(float).eps))
results_df['Sign'] = np.where(results_df['Coefficient'] > 0, 1, -1)
results_df['Signed_-log10_p'] = results_df['-log10_p'] * results_df['Sign']


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

# Set matplotlib parameters
import matplotlib as mpl
mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['axes.labelsize'] = 9
mpl.rcParams['xtick.labelsize'] = 9
mpl.rcParams['ytick.labelsize'] = 9
mpl.rcParams['xtick.color'] = 'black'
mpl.rcParams['ytick.color'] = 'black'
# 刻度线长度
mpl.rcParams['xtick.major.size'] = 0
mpl.rcParams['axes.titlesize'] = 11
mpl.rcParams['axes.labelcolor'] = 'black'
mpl.rcParams['axes.edgecolor'] = 'black'
mpl.rcParams['axes.linewidth'] = 0.5


# Maximum absolute signed -log10(p-value) for scaling
max_abs_value = results_df[results_df['Indicator'].isin([indicator_aliases.get(ind, ind) for ind in continuous_indicators])]['Signed_-log10_p'].abs().max()
vmin, vmax = -max_abs_value, max_abs_value

ordered_indicators = []
for subcat in subcategories:
    inds = [
        ind for ind, sc in factor_subcategory_map.items()
        if sc == subcat and ind in continuous_indicators
    ]
    ordered_indicators.extend(inds)


# === Replace indicator names with aliases for ordering ===
ordered_indicators_alias = [
    indicator_aliases.get(ind, ind) for ind in ordered_indicators
]

# === Prepare data for single long heatmap ===
plot_df = results_df[
    results_df['Indicator'].isin(ordered_indicators_alias)
].copy()

# 保证 Indicator 是有序分类变量（非常关键）
plot_df['Indicator'] = pd.Categorical(
    plot_df['Indicator'],
    categories=ordered_indicators_alias,
    ordered=True
)

# Pivot table
pivot_signed = plot_df.pivot_table(
    values='Signed_-log10_p',
    index='Indicator',
    columns='Subtype',
    fill_value=0
)

# === Figure size (long figure) ===
n_indicators = len(ordered_indicators_alias)
fig_height = n_indicators * 0.42 * 0.393701
fig_width = 11 / 2.54

fig, ax = plt.subplots(figsize=(fig_width, fig_height))

# === Heatmap ===
sns.heatmap(
    pivot_signed,
    cmap=cmap,
    center=0,
    vmin=vmin,
    vmax=vmax,
    annot=True,
    annot_kws={"size": 9},
    fmt=".1f",
    cbar=False,
    ax=ax
)

ax.set_xlabel('')
ax.set_ylabel('')
ax.tick_params(axis='y', rotation=0)

plt.tight_layout()


output_path = f'{OUTPUT_DIR}/environment_factor_signed_logp_heatmaps.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()

print(f'Heatmaps saved to "{output_path}"')

import matplotlib.cm as cm
# Create a figure and axis for the colorbar
fig, ax = plt.subplots(figsize=(3/2.54, 3/2.54))  # Reduced width for shorter colorbar

# Create a ScalarMappable with the colormap and normalization
norm = plt.Normalize(vmin=vmin, vmax=vmax)
scalar_mappable = cm.ScalarMappable(norm=norm, cmap=cmap)

# Add the colorbar to the figure (horizontal)
cbar = plt.colorbar(scalar_mappable, ax=ax)

# Set the ticks on the colorbar
cbar.set_ticks(np.linspace(vmin, vmax, 3))  # Set to 3 ticks

# Remove the colorbar outline
cbar.outline.set_visible(False)

# Turn off the axis to show only the colorbar
plt.axis('off')

# Adjust layout to fit the colorbar tightly
plt.tight_layout()

# Save the colorbar to a file
colorbar_path = 'colorbar.png'
plt.savefig(colorbar_path, dpi=300, bbox_inches='tight')

# Optionally, show the plot
# plt.show()

print(f'Colorbar saved to "{colorbar_path}"')