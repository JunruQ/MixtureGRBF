import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import utils.utils as utils
from statsmodels.api import OLS
import statsmodels.api as sm

# ==========================================
# 1. Configuration & Styles
# ==========================================
nsubtype = 5
exp_name = 'ukb_MixtureGRBF_cv_nsubtype_biom17'
OUTPUT_DIR = f'output/result_analysis/{exp_name}/{nsubtype}_subtypes'

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# --- 视觉风格常量 (与 PRS/Cognitive 一致) ---
MAIN_COLOR = '#0072BD' # 散点颜色
BOX_PROPS = {'facecolor': 'none', 'edgecolor': '#333333', 'linewidth': 1.2}
WHISKER_PROPS = {'color': '#333333', 'linewidth': 1.2}
CAP_PROPS = {'color': '#333333', 'linewidth': 1.2}
MEDIAN_PROPS = {'color': 'red', 'linewidth': 1.5}
POINTS_SIZE = 2
POINTS_ALPHA = 0.3
MAX_POINTS_DISPLAY = 500 

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 2. Helper Functions
# ==========================================
def merge_info(df, info_path, subset_field):
    field = pd.read_csv(info_path)
    fields = field['Value'].tolist()
    field_name_map = dict(zip(field['Value'], field['Name']))
    subset = subset_field[subset_field['Field_ID'].isin(fields)]
    
    for subset_idx in subset['Subset_ID'].unique():
        subset_fields = subset[subset['Subset_ID'] == subset_idx]['Field_ID'].tolist()
        data_path = f'data/brain_mri_PRS_2025_9_29/UKB_Tabular_merged_10/UKB_subset_{subset_idx}.csv'
        if os.path.exists(data_path):
            data = pd.read_csv(data_path, usecols=['eid'] + subset_fields).rename(columns=field_name_map)
            df = pd.merge(df, data, on='eid', how='left')
    return df

def wrap_label(label: str, threshold: int = 20) -> str:
    if not label or len(str(label)) <= threshold: return str(label)
    s = str(label); middle = len(s) // 2
    spaces = [i for i, c in enumerate(s) if c == ' ']
    if not spaces: return s
    best_space = min(spaces, key=lambda x: abs(x - middle))
    return s[:best_space] + '\n' + s[best_space+1:]

# ==========================================
# 3. Data Loading & Mental Health Aggregation
# ==========================================
print("Loading Mental Health data...")
subset_field = pd.read_csv('data/brain_mri_PRS_2025_9_29/UKB_Tabular_merged_10/UKB_FieldID_Subset.csv')
subtype_stage = utils.get_subtype_stage_with_cov(exp_name, nsubtype).rename(columns={'PTID': 'eid'})

# 合并时间字段并更新 stage
subtype_stage = merge_info(subtype_stage, 'data/brain_mri_PRS_2025_9_29/time_field.csv', subset_field)
subtype_stage = merge_info(subtype_stage, 'data/brain_mri_PRS_2025_9_29/mh_field.csv', subset_field)
subtype_stage['time_delta'] = (pd.to_datetime(subtype_stage['Date']) - pd.to_datetime(subtype_stage['Instance 0'])).dt.days / 365.25
subtype_stage['stage'] = subtype_stage['time_delta'] + subtype_stage['stage']

mh_df = pd.read_csv('data/brain_mri_PRS_2025_9_29/mh_field.csv')
mental_health_categories = ['Anxiety', 'Depression', 'Happiness', 'Mania', 'Mental distress', 'Psychotic experience', 'Self-harm', 'Trauma']
categorical_not_in_use = ['Weight change during worst episode of depression','Mental health problems ever diagnosed by a professional','Manifestations of mania or irritability','Methods of self-harm used']

# 计算各分类的均值得分
for mental_health in mental_health_categories:
    mh_names = mh_df[mh_df['MentalCategory'] == mental_health]['Name'].tolist()
    mh_names = [name for name in mh_names if name in subtype_stage.columns and name not in categorical_not_in_use]
    
    mh_data = subtype_stage[mh_names].copy()
    mh_data = mh_data.mask(mh_data < 0) # 过滤非法值
    
    # Min-Max 缩放各细分项并计算行均值
    mh_scaled = (mh_data - mh_data.min()) / (mh_data.max() - mh_data.min())
    subtype_stage[mental_health] = mh_scaled.mean(axis=1, skipna=True)

# ==========================================
# 4. Residualization & Plotting Data Prep
# ==========================================
final_plot_df = pd.DataFrame()

for col in mental_health_categories:
    # 提取子集并清洗
    sub_df = subtype_stage[['eid', 'subtype', 'stage', 'sex', 'centre', col]].copy().dropna()
    
    # 协变量残差化 (OLS)
    y = sub_df[col]
    covariates = pd.get_dummies(sub_df[['stage', 'sex', 'centre']], drop_first=True)
    X = sm.add_constant(covariates)
    res = OLS(y, X).fit().resid
    
    # Min-Max 归一化残差
    norm_res = (res - res.min()) / (res.max() - res.min())
    
    temp_df = pd.DataFrame({
        'subtype': sub_df['subtype'],
        'score': norm_res,
        'feature': col
    })
    final_plot_df = pd.concat([final_plot_df, temp_df], axis=0)

# ==========================================
# 5. Plotting (5-Column Grid Boxplot)
# ==========================================
N_COLS = 5
n_rows = int(np.ceil(len(mental_health_categories) / N_COLS))
fig, axes = plt.subplots(n_rows, N_COLS, figsize=(33/2.54, 15/2.54), dpi=150)
axes = axes.flatten()

for i, mh_cat in enumerate(mental_health_categories):
    ax = axes[i]
    plot_subset = final_plot_df[final_plot_df['feature'] == mh_cat]
    
    if plot_subset.empty:
        ax.axis('off')
        continue

    # --- 1. Stripplot (Background) ---
    sampled_list = []
    for st in range(1, nsubtype + 1):
        st_data = plot_subset[plot_subset['subtype'] == st]
        if len(st_data) > MAX_POINTS_DISPLAY:
            st_data = st_data.sample(n=MAX_POINTS_DISPLAY, random_state=42)
        sampled_list.append(st_data)
    
    if mh_cat in ['Psychotic experience']:
        for st_data in sampled_list:
            if mh_cat == 'Psychotic experience':
                st_data.loc[st_data['score'] >= 0.6, 'score'] = 0.6
    
    sns.stripplot(
        data=pd.concat(sampled_list), x='subtype', y='score',
        color=MAIN_COLOR, size=POINTS_SIZE, alpha=POINTS_ALPHA,
        jitter=0.25, zorder=0, ax=ax
    )
    
    # --- 2. Boxplot (Foreground) ---
    sns.boxplot(
        data=plot_subset, x='subtype', y='score',
        showfliers=False, width=0.5,
        boxprops=BOX_PROPS, whiskerprops=WHISKER_PROPS,
        capprops=CAP_PROPS, medianprops=MEDIAN_PROPS,
        zorder=10, ax=ax
    )

    # --- 3. Aesthetics ---
    # 计算该指标的总样本量
    n_obs = len(plot_subset)

    # 在标题中加入 (N=...)，\n 表示换行，让标题更整齐
    ax.set_title(f"{wrap_label(mh_cat, 30)}\n(n={n_obs:,})", fontsize=9, pad=10)
    ax.set_xlabel('Subtype', fontsize=9)
    ax.set_ylabel('Normalized score', fontsize=9)
    
    sns.despine(ax=ax)
    ax.tick_params(labelsize=9)
    ax.grid(axis='y', linestyle='--', alpha=0.3)

# 隐藏多余子图
for j in range(len(mental_health_categories), len(axes)):
    axes[j].axis('off')

plt.tight_layout()
plt.subplots_adjust(hspace=0.5, wspace=0.3)

# 保存
out_path = os.path.join(OUTPUT_DIR, 'mh_difference_boxplot.png')
plt.savefig(out_path, bbox_inches='tight', dpi=500)
print(f"Mental Health boxplot saved to: {out_path}")
plt.show()