import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import utils.utils as utils
from statsmodels.api import OLS
import statsmodels.api as sm

# ==========================================
# 1. Configuration & Styles (Matched to your logic)
# ==========================================
nsubtype = 5
exp_name = 'ukb_MixtureGRBF_cv_nsubtype_biom17'
OUTPUT_DIR = f'output/result_analysis/{exp_name}/{nsubtype}_subtypes'

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# --- Visual Style Constants ---
MAIN_COLOR = '#0072BD' # 蓝色散点
BOX_PROPS = {'facecolor': 'none', 'edgecolor': '#333333', 'linewidth': 1.2}
WHISKER_PROPS = {'color': '#333333', 'linewidth': 1.2}
CAP_PROPS = {'color': '#333333', 'linewidth': 1.2}
MEDIAN_PROPS = {'color': 'red', 'linewidth': 1.5} # 红色中位数线
POINTS_SIZE = 2
POINTS_ALPHA = 0.3
MAX_POINTS_DISPLAY = 500 

# 所有绘图字体改为Arial
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 2. Data Loading & Preprocessing
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

# --- Step A: Load Data ---
subset_field = pd.read_csv('data/brain_mri_PRS_2025_9_29/UKB_Tabular_merged_10/UKB_FieldID_Subset.csv')
subtype_stage = utils.get_subtype_stage_with_cov(exp_name, nsubtype).rename(columns={'PTID': 'eid'})

# 合并时间字段计算 Instance 2 的年龄
subtype_stage = merge_info(subtype_stage, 'data/brain_mri_PRS_2025_9_29/time_field.csv', subset_field)
subtype_stage['age_instance_2'] = subtype_stage['stage'] + (pd.to_datetime(subtype_stage['Instance 2']) - pd.to_datetime(subtype_stage['Instance 0'])).dt.days / 365.25

# 合并认知功能字段
subtype_stage = merge_info(subtype_stage, 'data/brain_mri_PRS_2025_9_29/cf_field.csv', subset_field)
cf_df = pd.read_csv('data/brain_mri_PRS_2025_9_29/cf_field.csv')

# --- Step B: Residualization & Normalization ---
# 用于最终绘图的 DataFrame
final_plot_df = pd.DataFrame()

for _, row in cf_df.iterrows():
    cf_name, cf_instance, cf_inversed = row['Name'], row['Instance'], row['IsInversed']
    age_col = 'stage' if cf_instance == 0 else 'age_instance_2'
    
    # 提取子集并清洗
    sub_df = subtype_stage[['eid', 'subtype', age_col, 'sex', 'centre', cf_name]].copy().dropna()
    
    if cf_name == 'Prospective memory':
        sub_df[cf_name] = sub_df[cf_name].map({0:0, 2:1, 1:2})
    if cf_name == 'Numeric memory':
        sub_df[cf_name] = np.where(sub_df[cf_name] < 0, np.nan, sub_df[cf_name])
    sub_df.dropna(inplace=True)

    # 协变量残差化
    y = sub_df[cf_name]
    covariates = pd.get_dummies(sub_df[[age_col, 'sex', 'centre']], drop_first=True)
    X = sm.add_constant(covariates)
    res = OLS(y, X).fit().resid
    
    # MinMax 归一化 (包含反转逻辑)
    if cf_inversed:
        norm_res = 1 - (res - res.min()) / (res.max() - res.min())
    else:
        norm_res = (res - res.min()) / (res.max() - res.min())
    
    # 收集处理后的数据
    temp_df = pd.DataFrame({
        'subtype': sub_df['subtype'],
        'score': norm_res,
        'feature': cf_name
    })
    final_plot_df = pd.concat([final_plot_df, temp_df], axis=0)

# ==========================================
# 3. Plotting Logic (Grid Wrap)
# ==========================================
cf_names = cf_df['Name'].tolist()
N_COLS = 5
n_rows = int(np.ceil(len(cf_names) / N_COLS))
fig, axes = plt.subplots(n_rows, N_COLS, figsize=(33/2.54, 15/2.54), dpi=150)
axes = axes.flatten()

for i, cf_name in enumerate(cf_names):
    ax = axes[i]
    plot_subset = final_plot_df[final_plot_df['feature'] == cf_name]
    
    if plot_subset.empty:
        ax.axis('off')
        continue

    # --- 1. Stripplot ---
    sampled_list = []
    for st in range(1, nsubtype + 1):
        st_data = plot_subset[plot_subset['subtype'] == st]
        if len(st_data) > MAX_POINTS_DISPLAY:
            st_data = st_data.sample(n=MAX_POINTS_DISPLAY, random_state=42)
        sampled_list.append(st_data)

    if cf_name in ['Pairs matching', 'Reaction time', 'Trail making']:
        for st_data in sampled_list:
            if cf_name == 'Pairs matching':
                st_data.loc[st_data['score'] <= 0.4, 'score'] = 0.4
            if cf_name == 'Reaction time':
                st_data.loc[st_data['score'] <= 0.4, 'score'] = 0.4
            if cf_name == 'Trail making':
                st_data.loc[st_data['score'] <= 0.5, 'score'] = 0.5
    
    sns.stripplot(
        data=pd.concat(sampled_list), x='subtype', y='score',
        color=MAIN_COLOR, size=POINTS_SIZE, alpha=POINTS_ALPHA,
        jitter=0.25, zorder=0, ax=ax
    )
    
    # --- 2. Boxplot ---
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
    ax.set_title(f"{wrap_label(cf_name, 30)}\n(n={n_obs:,})", fontsize=9, pad=10)
    ax.set_xlabel('Subtype', fontsize=9)
    ax.set_ylabel('Normalized score', fontsize=9)
    
    sns.despine(ax=ax)
    ax.tick_params(labelsize=8)
    ax.grid(axis='y', linestyle='--', alpha=0.3)

# 隐藏多余的子图
for j in range(len(cf_names), len(axes)):
    axes[j].axis('off')

plt.tight_layout()
plt.subplots_adjust(hspace=0.5, wspace=0.3)

# 保存文件
out_path = os.path.join(OUTPUT_DIR, 'cf_difference_boxplot.png')
plt.savefig(out_path, bbox_inches='tight', dpi=500)
print(f"Cognitive function plot saved to: {out_path}")
plt.show()