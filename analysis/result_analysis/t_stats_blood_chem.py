import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import textwrap

def bonferroni_correction(p_values):
    n = len(p_values)
    corrected_p_values = np.minimum(np.array(p_values) * n, 1.0)
    return corrected_p_values

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.titlesize'] = 10
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 7  # 减小y轴标签字体以适应更多指标
plt.rcParams['axes.labelcolor'] = 'black'
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['axes.linewidth'] = 1.0

# 文件路径和参数保持不变
nsubtype = 5
INPUT_TABLE_PATH = 'data/ClinicalLabData.csv'
exp_name = 'ukb_MixtureGRBF_cv_nsubtype_biom17'
SUBTYPE_STAGE_PATH = f'output/{exp_name}/{nsubtype}_subtypes/subtype_stage.csv'
SUBTYPE_ORDER_PATH = f'output/result_analysis/{exp_name}/{nsubtype}_subtypes/all_cause_mortality_order.csv'
CLASSIFICATION_PATH = 'analysis/result_analysis/clinical_lab_data_dict.csv'
OUTPUT_DIR = f'output/result_analysis/{exp_name}/{nsubtype}_subtypes'

palette = ['#206491', '#038db2', '#f9637c', '#fe7966', '#fbb45c', '#ffcb5d', '#81d0bb', '#45aab4', '#8470FF']
gray = '#777777'

df = pd.read_csv(INPUT_TABLE_PATH)
df.columns = df.columns.map(lambda x: x[:5] if x != 'eid' else x)
subtype_stage = pd.read_csv(SUBTYPE_STAGE_PATH)
classification = pd.read_csv(CLASSIFICATION_PATH)
classification = classification[['Field ID', 'Category', 'Abbreviation']]
classification['Field ID'] = classification['Field ID'].astype(str)
df = pd.merge(df, subtype_stage, how='left', left_on='eid', right_on='PTID')
df = df.dropna(subset=['subtype'])
df['subtype'] = df['subtype'].astype(int)

try:
    subtype_order = pd.read_csv(SUBTYPE_ORDER_PATH, header=None)
except:
    subtype_order = pd.DataFrame([list(range(1, nsubtype+1))])

covariate_names = [col for col in df.columns if col not in ['eid', 'PTID', 'stage', 'subtype']]

category_map = dict(zip(classification['Field ID'], classification['Category']))
abbr_map = dict(zip(classification['Field ID'], classification['Abbreviation']))
unique_categories = classification['Category'].unique()
category_colors = {cat: palette[i % len(palette)] for i, cat in enumerate(unique_categories)}
cov_color_map = {}
for cov in covariate_names:
    category = category_map.get(cov, 'Unknown')
    cov_color_map[cov] = category_colors.get(category, gray)

alpha = 0.05

# 修改图形尺寸：增加高度以适应50+指标
fig = plt.figure(figsize=(15, 7), dpi=300, facecolor='white')  # 宽度20，高度12
fig.subplots_adjust(hspace=0.3, wspace=0.3, left=0.05, bottom=0.1, right=0.95, top=0.9)

for i in range(1, nsubtype + 1):
    k = int(subtype_order.iloc[i-1, 0])
    ax = plt.subplot(1, nsubtype, i)
    
    group = df['subtype'] == k
    ts = []
    ps = []
    for cov in covariate_names:
        case = df.loc[group, cov].dropna()
        control = df.loc[~group, cov].dropna()
        t_value, p_value = stats.ttest_ind(case, control, equal_var=False, nan_policy='omit')
        ts.append(t_value)
        ps.append(p_value)

    ps = bonferroni_correction(ps)
    is_significant = (ps <= alpha).tolist()
    info_item = {cov: [ts[idx], is_significant[idx]] for idx, cov in enumerate(covariate_names)}

    x = []
    y = []
    count = 0
    abbreviations = []
    colors = []
    t_values_dict = {}  # 存储t值用于标注

    for j in range(len(unique_categories)):
        c = classification[classification['Category'] == unique_categories[j]]['Field ID'].tolist()
        c = sorted(list(set(c).intersection(covariate_names)))
        for biom in c:
            count += 1
            t_val = info_item[biom][0]
            x.append(t_val)
            y.append(count - 1)
            abbreviations.append(abbr_map.get(biom, biom))
            t_values_dict[count - 1] = (t_val, abbr_map.get(biom, biom))  # 存储t值和简称
            if info_item[biom][1]:
                colors.append(palette[j % len(palette)])
            else:
                colors.append(gray)
    
    sig_count = sum(is_significant)

    for xi, yi, color in zip(x, y, colors):
        ax.plot(xi, yi, marker='o', markersize=3, color=color, linestyle='none')

    # 标注最显著的指标（选择绝对值最大的前5个）
    top_n = 5
    significant_items = sorted(t_values_dict.items(), key=lambda x: abs(x[1][0]), reverse=True)[:top_n]
    for yi, (t_val, abbr) in significant_items:
        if abs(t_val) > 0:  # 确保t值不为0
            ax.annotate(abbr, (t_val, yi), xytext=(-10, 5) if t_val > 0 else (10, 5), textcoords='offset points', 
                       fontsize=6, color='black', ha='right' if t_val > 0 else 'left')

    ax.set_xlabel('t-statistic', fontsize=9, labelpad=5, color='black')
    ax.set_title(f'Subtype {i}\n{sig_count} significant', fontsize=9, pad=10, color='black')
    ax.set_yticks(y)
    ax.set_yticklabels(abbreviations, fontsize=6)  # 减小字体适应更多标签
    
    x_abs_max = np.max(np.abs([xi for xi in x if np.isfinite(xi)])) * 1.1
    ax.set_xlim(-x_abs_max, x_abs_max)
    ax.set_ylim(-1, count)

    ax.grid(False)
    ax.tick_params(direction='in')
    ax.axvline(0, color='black', linewidth=1.0)

    if i == 1:
        handles = [plt.Line2D([0], [0], marker='o', color=category_colors[cat], 
                             markersize=6, linestyle='', label=textwrap.fill(cat, 20)) 
                  for cat in unique_categories]
        ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(-0.9, 1.1),
                  frameon=True, framealpha=1, edgecolor='black',
                  title='Categories', title_fontsize=9, fontsize=8)

fig.text(0.5, 0.02, 'Colored by Category, Bonferroni corrected p<0.05 for significance',
         ha='center', fontsize=9, color='black')

plt.savefig(OUTPUT_DIR+'/t_stats_blood_chem.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()