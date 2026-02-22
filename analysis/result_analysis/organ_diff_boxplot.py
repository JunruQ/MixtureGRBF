import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
import math
import matplotlib.colors as mcolors
from sklearn.preprocessing import StandardScaler
import utils.utils as utils

# ==========================================
# 0. 辅助函数 (Helper Functions)
# ==========================================
def read_csv_smart(path):
    """尝试多种编码读取 CSV"""
    encodings = ["utf-8-sig", "utf-8", "gbk", "gb18030"]
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"无法读取文件: {path}")

def wrap_label(label: str, threshold: int = 15) -> str:
    """长文本换行"""
    if not label or len(str(label)) <= threshold:
        return str(label)
    s = str(label)
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
        return s[:middle] + '\n' + s[middle:]

# ==========================================
# 1. 配置与样式 (Configuration)
# ==========================================
nsubtype = 5
BLOOD_CHEM_PATH = 'data/ClinicalLabData.csv'
OTHER_IND_PATH = 'data/prot_Modifiable_bl_data.csv'
exp_name = 'ukb_MixtureGRBF_cv_nsubtype_biom17'
CLASSIFICATION_PATH = 'analysis/result_analysis/clinical_lab_data_dict.csv'
OUTPUT_DIR = f'output/result_analysis/{exp_name}/{nsubtype}_subtypes'

LABEL_FILE_PATH = f'output/result_analysis/{exp_name}/{nsubtype}_subtypes/environment_factor_label.csv'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# --- 核心样式定义 ---
MAIN_COLOR = '#0072BD'
BOX_PROPS = {'facecolor': 'none', 'edgecolor': '#333333', 'linewidth': 1.2}
WHISKER_PROPS = {'color': '#333333', 'linewidth': 1.2}
CAP_PROPS = {'color': '#333333', 'linewidth': 1.2}
MEDIAN_PROPS = {'color': 'red', 'linewidth': 1.5}
POINTS_SIZE = 2
POINTS_ALPHA = 0.4
MAX_POINTS_DISPLAY = 500

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 2. 数据加载 (Data Loading)
# ==========================================
print("Loading clinical and phenotype data...")
df = utils.get_subtype_stage_with_cov(exp_name, nsubtype)

# 读取分类字典
classification = pd.read_csv(CLASSIFICATION_PATH)
classification = classification[['Field ID', 'Category', 'Abbreviation']]
classification['Field ID'] = classification['Field ID'].astype(str)
abbr_map = dict(zip(classification['Field ID'], classification['Abbreviation']))

# 读取并合并 Blood Chem
blood_chem = pd.read_csv(BLOOD_CHEM_PATH)
blood_chem_cols = ['eid'] + [abbr_map.get(col.replace('-0.0', ''), col) for col in blood_chem.columns if col != 'eid']
blood_chem.columns = blood_chem_cols
common_cols = df.columns.intersection(blood_chem.columns).difference(['PTID', 'eid'])
blood_chem_for_merge = blood_chem.drop(columns=common_cols)
df = pd.merge(df, blood_chem_for_merge, how='left', left_on='PTID', right_on='eid')

# 读取并合并 Other Indicators
other_ind = pd.read_csv(OTHER_IND_PATH)
common_cols = df.columns.intersection(other_ind.columns).difference(['PTID', 'eid'])
other_ind_for_merge = other_ind.drop(columns=common_cols)
df = pd.merge(df, other_ind_for_merge, how='left', left_on='PTID', right_on='eid')

# 定义需要绘图的指标列表
indicators = ['Pulse rate', 'Systolic blood pressure', 'Diastolic blood pressure',
              'ALP', 'ALT', 'AST', 'GGT', 'Total bilirubin', 'Albumin',
              'WBC', 'RBC', 'MPV', 'CRP',
              'Glucose', 'Cholesterol', 'HDL-C', 'LDL-C', 'Triglycerides',
              'Forced vital capacity', 'Forced expiratory volume in 1-second', 'Peak expiratory flow',
              'Creatinine', 'Cystatin C', 'Sodium in urine']

# 过滤出存在于 DataFrame 中的指标
final_plot_list = [ind for ind in indicators if ind in df.columns]

# ==========================================
# 3. [核心整合] 加载与解析 Ylabel 信息
# ==========================================
print(f"Loading label metadata from {LABEL_FILE_PATH}...")
try:
    ylab_df = read_csv_smart(LABEL_FILE_PATH)
    cat_pattern = re.compile(r"^\s*([+-]?\d+)\s*[:：]\s*(.+?)\s*$")
    ylab_df["phenotype"] = ylab_df["phenotype"].astype(str).str.strip()
    ylab_df["type"] = ylab_df["type"].astype(str).str.strip().str.lower()
    mapping_cols = [c for c in ylab_df.columns if re.fullmatch(r"[A-Z]", str(c).strip())]
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
                if pd.isna(v): continue
                s = str(v).strip()
                match = cat_pattern.match(s)
                if match:
                    cat_map[int(match.group(1))] = match.group(2).strip()
        ylab_info[p] = {"type": t, "unit": unit, "cat_map": cat_map, "full_name": p}
    print("Ylabel data loaded successfully.")
except Exception as e:
    print(f"Warning: Could not load label file ({e}). Using default formatting.")
    ylab_info = {}

# 备用字典
fallback_names = {
    'WBC': 'White blood cell count', 'RBC': 'Red blood cell count', 'MPV': 'Mean platelet volume',
    'CRP': 'C-reactive protein', 'ALP': 'Alkaline phosphatase', 'ALT': 'Alanine aminotransferase',
    'AST': 'Aspartate aminotransferase', 'GGT': 'Gamma-glutamyl transferase',
    'HDL-C': 'HDL cholesterol', 'LDL-C': 'LDL cholesterol'
}

# Category mapping（用于分组与顺序）
category_mapping = {
    'Pulse rate': 'Cardiovascular',
    'Systolic blood pressure': 'Cardiovascular',
    'Diastolic blood pressure': 'Cardiovascular',
    'Alkaline phosphatase': 'Hepatic',
    'Alanine aminotransferase': 'Hepatic',
    'Aspartate aminotransferase': 'Hepatic',
    'Gamma-glutamyl transferase': 'Hepatic',
    'Total bilirubin': 'Hepatic',
    'Albumin': 'Hepatic',
    'White blood cell count': 'Immune',
    'Red blood cell count': 'Immune',
    'Mean platelet volume': 'Immune',
    'C-reactive protein': 'Immune',
    'Glucose': 'Metabolic',
    'Cholesterol': 'Metabolic',
    'HDL cholesterol': 'Metabolic',
    'LDL cholesterol': 'Metabolic',
    'Triglycerides': 'Metabolic',
    'Forced vital capacity': 'Pulmonary',
    'Forced expiratory volume in 1-second': 'Pulmonary',
    'Peak expiratory flow': 'Pulmonary',
    'Creatinine': 'Renal',
    'Cystatin C': 'Renal',
    'Sodium in urine': 'Renal'
}

# ==========================================
# 4. 按 category_mapping 排序 final_plot_list（保证同类别连续）
# ==========================================
short_to_full = {ind: fallback_names.get(ind, ind) for ind in final_plot_list}
desired_full_order = list(category_mapping.keys())  # 保持定义顺序
sorted_plot_list = []
for full_name in desired_full_order:
    for short, full in short_to_full.items():
        if full == full_name:
            sorted_plot_list.append(short)
            break
final_plot_list = sorted_plot_list

# ==========================================
# 5. 绘图逻辑 (Plotting)
# ==========================================
N_COLS = 6
n_rows = 6
FIG_WIDTH = 33/2.54
fig_height = 23/2.54
fig, axes = plt.subplots(n_rows, N_COLS, figsize=(FIG_WIDTH, fig_height), dpi=150)
axes = axes.flatten()

df_raw = df.copy()

# 用于分组换行与标题添加的变量
current_cat = None
ax_idx = 0  # 当前使用的子图索引

for ind in final_plot_list:
    full_name = fallback_names.get(ind, ind)
    cat = category_mapping.get(full_name, 'Other')
    info = ylab_info.get(full_name, {})
    unit = info.get('unit', '')

    # 检测是否为新类别
    is_new_group = (cat != current_cat)

    # 如果是新类别且不在行首 → 填充空子图，强制换行
    if is_new_group:
        pos_in_row = ax_idx % N_COLS
        if pos_in_row != 0:
            pad_num = N_COLS - pos_in_row
            for _ in range(pad_num):
                if ax_idx < len(axes):
                    axes[ax_idx].axis('off')
                ax_idx += 1
        current_cat = cat

    # 当前子图
    if ax_idx >= len(axes):
        break
    ax = axes[ax_idx]

    # 提取数据
    plot_data = df_raw[['subtype', ind]].dropna()
    if plot_data.empty:
        ax.text(0.5, 0.5, 'No Data', ha='center', va='center')
        ax_idx += 1
        continue

    unique_subtypes = sorted(plot_data['subtype'].unique())

    # --- Stripplot ---
    sampled_dfs = []
    for st in unique_subtypes:
        sub_df = plot_data[plot_data['subtype'] == st]
        if len(sub_df) > MAX_POINTS_DISPLAY:
            sub_df = sub_df.sample(n=MAX_POINTS_DISPLAY, random_state=42)
        sampled_dfs.append(sub_df)

    if full_name in ['Forced vital capacity', 'Forced expiratory volume in 1-second']:
        for st_data in sampled_dfs:
            if full_name == 'Forced vital capacity':
                st_data.loc[st_data[full_name] >= 10, full_name] = 10
            if full_name == 'Forced expiratory volume in 1-second':
                st_data.loc[st_data[full_name] >= 10, full_name] = 10

    if sampled_dfs:
        scatter_data = pd.concat(sampled_dfs)
        sns.stripplot(
            data=scatter_data, x='subtype', y=ind,
            color=MAIN_COLOR, size=POINTS_SIZE, alpha=POINTS_ALPHA,
            jitter=0.25, zorder=0, ax=ax, legend=False
        )

    # --- Boxplot ---
    sns.boxplot(
        data=plot_data, x='subtype', y=ind,
        showfliers=False, width=0.5,
        boxprops=BOX_PROPS, whiskerprops=WHISKER_PROPS,
        capprops=CAP_PROPS, medianprops=MEDIAN_PROPS,
        zorder=10, ax=ax
    )

    # --- 标题（新类别时添加类别名） ---
    title_display = wrap_label(full_name, threshold=45)
    n_count = len(plot_data)
    ax.set_title(f"{title_display}\n(n={n_count:,})", fontsize=9, pad=10)

    # --- 轴标签美化 ---
    ax.set_xlabel('')
    if cat in ['Hepatic', 'Immune', 'Metabolic'] or full_name in ["Creatinine", "Cystatin C"]:
        ax.set_ylabel("z-score", fontsize=9)
    elif unit:
        ax.set_ylabel(unit, fontsize=9)
    else:
        ax.set_ylabel('', fontsize=9)
    
    if cat == 'Renal':
        ax.set_xlabel('Subtype', fontsize=9)
    else:
        ax.set_xlabel('')

    sns.despine(ax=ax, top=True, right=True)
    ax.tick_params(axis='y', labelsize=9)
    ax.grid(axis='y', linestyle='--', alpha=0.3)

    ax_idx += 1

# 隐藏剩余空子图
for j in range(ax_idx, len(axes)):
    axes[j].axis('off')

plt.tight_layout()
plt.subplots_adjust(hspace=1, wspace=0.5)

out_file = os.path.join(OUTPUT_DIR, 'organ_indicators_boxplot.png')
plt.savefig(out_file, bbox_inches='tight', dpi=500)
print(f"Figure saved successfully to: {out_file}")