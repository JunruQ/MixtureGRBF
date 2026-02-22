import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import textwrap
import textalloc

def bonferroni_correction(p_values):
    n = len(p_values)
    corrected_p_values = np.minimum(np.array(p_values) * n, 1.0)  # 限制最大值为1
    return corrected_p_values

# MATLAB 风格设置
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.titlesize'] = 10
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['axes.labelcolor'] = 'black'
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['axes.linewidth'] = 1.0

# 文件路径和参数
nsubtype = 5
INPUT_TABLE_PATH = 'data/ClinicalLabData.csv'
exp_name = 'ukb_MixtureGRBF_cv_nsubtype_biom17'
SUBTYPE_STAGE_PATH = f'output/{exp_name}/{nsubtype}_subtypes/subtype_stage.csv'
OUTPUT_DIR = f'output/result_analysis/{exp_name}/{nsubtype}_subtypes'
SUBTYPE_ORDER_PATH = f'output/result_analysis/{exp_name}/{nsubtype}_subtypes/all_cause_mortality_order.csv'
CLASSIFICATION_PATH = 'analysis/result_analysis/clinical_lab_data_dict.csv'

# 定义颜色（为每个分类分配颜色）
palette = ['#206491', '#038db2', '#f9637c', '#fe7966', '#fbb45c', '#ffcb5d', '#81d0bb', '#45aab4', '#8470FF']
gray = '#777777'  # 灰色用于未分类或默认

# 参数
top_n = 5  # 选择前 5 个显著协变量
alpha = 0.05

# 读取数据
df = pd.read_csv(INPUT_TABLE_PATH)
df.columns = df.columns.map(lambda x: x[:5] if x != 'eid' else x)
subtype_stage = pd.read_csv(SUBTYPE_STAGE_PATH)
classification = pd.read_csv(CLASSIFICATION_PATH)
classification = classification[['Field ID', 'Category', 'Abbreviation']]
classification['Field ID'] = classification['Field ID'].astype(str)
df = pd.merge(df, subtype_stage, how='left', left_on='eid', right_on='PTID')
df = df.dropna(subset=['subtype'])  # 移除 subtype 为空的行
df['subtype'] = df['subtype'].astype(int)

try:
    subtype_order = pd.read_csv(SUBTYPE_ORDER_PATH, header=None)
except:
    subtype_order = pd.DataFrame([list(range(1, nsubtype+1))])

# 获取协变量列
covariate_names = [col for col in df.columns if col not in ['eid', 'PTID', 'stage', 'subtype']]

# 创建分类到颜色的映射
category_map = dict(zip(classification['Field ID'], classification['Category']))
abbr_map = dict(zip(classification['Field ID'], classification['Abbreviation']))
unique_categories = classification['Category'].unique()
category_colors = {cat: palette[i % len(palette)] for i, cat in enumerate(unique_categories)}
cov_color_map = {}
for cov in covariate_names:
    category = category_map.get(cov, 'Unknown')
    cov_color_map[cov] = category_colors.get(category, gray)

# 创建图形
if nsubtype == 5:
    fig = plt.figure(figsize=(9.5, 9), dpi=300, facecolor='white')
elif nsubtype == 4:
    fig = plt.figure(figsize=(9.5, 7), dpi=300, facecolor='white')
else:
    fig = plt.figure(figsize=(9.5, 7), dpi=300, facecolor='white')

fig.subplots_adjust(hspace=0.3, wspace=0.2, left=0.05, bottom=0.1, right=0.95, top=0.9)

for i in range(1, nsubtype + 1):
    k = int(subtype_order.iloc[i-1, 0])
    ax = plt.subplot((nsubtype + 1) // 2, 2, i)
    
    # 对每个协变量进行 t-test
    group = df['subtype'] == k
    ts = []
    ps = []
    for cov in covariate_names:
        case = df.loc[group, cov].dropna()
        control = df.loc[~group, cov].dropna()
        t_value, p_value = stats.ttest_ind(case, control, equal_var=False, nan_policy='omit')
        ts.append(t_value)
        ps.append(p_value)

    # Bonferroni 校正
    ps = bonferroni_correction(ps)
    is_significant = (ps <= alpha).tolist()
    info_item = {cov: [ts[idx], ps[idx], is_significant[idx]] for idx, cov in enumerate(covariate_names)}

    # 选出前 top_n 个显著协变量
    sig_results = [(cov, abs(info_item[cov][0]), info_item[cov][1]) 
                   for cov in covariate_names if info_item[cov][2]]
    sig_results.sort(key=lambda x: x[1], reverse=True)  # 按 t 值绝对值排序
    top_significant = [cov for cov, _, _ in sig_results[:top_n]]

    # 准备绘图数据
    x = []
    y = []
    count = 0
    bioms_new = []
    colors = []
    abbreviations = []
    t_values_dict = {}

    for j in range(len(unique_categories)):
        c = classification[classification['Category'] == unique_categories[j]]['Field ID'].tolist()
        c = sorted(list(set(c).intersection(covariate_names)))
        for biom in c:
            count += 1
            t_val = info_item[biom][0]
            x.append(count)
            bioms_new.append(biom)
            y.append(info_item[biom][0])
            t_values_dict[count - 1] = (t_val, abbr_map.get(biom, biom))  # 存储t值和简称
            if info_item[biom][2]:
                colors.append(palette[j % len(palette)])
            else:
                colors.append(gray)
            abbreviations.append(abbr_map.get(biom, biom))

    sig_count = sum(is_significant)

    # 绘制散点图
    texts = []
    for xi, yi, color, biom in zip(x, y, colors, bioms_new):
        ax.plot(xi, yi, marker='o', markersize=3, color=color, linestyle='none')


    top_n = 10
    significant_indices = np.argsort(np.abs(y))[::-1][:top_n]
    significant_indices = significant_indices[::-1]
    text_x = [x[i] for i in significant_indices]
    text_y = [y[i] for i in significant_indices]
    text_abbr = [abbreviations[i] for i in significant_indices]

    text_x = text_x[::-1]
    text_y = text_y[::-1]
    text_abbr = text_abbr[::-1]
    
    
    # top_n = 5
    # significant_items = sorted(t_values_dict.items(), key=lambda x: abs(x[1][0]), reverse=True)[:top_n]
    # for xi, (t_val, abbr) in significant_items:
    #     if abs(t_val) > 0:
    #         # 在数据点右侧创建初始标注（稍后自动调整）
    #         text = ax.annotate(abbr, (xi+1, t_val), xytext=(0, 10) if t_val > 0 else (0, -10), textcoords='offset points', 
    #                    fontsize=6, color='black', ha='left', va='top' if t_val > 0 else 'bottom')
    #         texts.append(text)
    # top_n = 5
    # significant_items = sorted(t_values_dict.items(), key=lambda x: abs(x[1][0]), reverse=True)[:top_n]
    # for xi, (t_val, abbr) in significant_items:
    #     if abs(t_val) > 0:
    #         # 在数据点右侧创建初始标注
    #         text = ax.annotate(abbr, 
    #                         (xi, t_val),  # 数据点位置
    #                         xytext=(5, 5 if t_val > 0 else -5),  # 初始偏移量稍微调整
    #                         textcoords='offset points', 
    #                         fontsize=6, 
    #                         color='black', 
    #                         ha='left', 
    #                         va='center'  # 垂直对齐改为 center 以便更灵活调整
    #                         # arrowprops=dict(arrowstyle='-', color='black', linewidth=0.5))  # 添加箭头
    #         )
    #         texts.append(text)

    # 使用 adjust_text 自动调整文本位置
    # adjust_text(texts, 
    #             expand_points=(1.2, 1.2),  # 点之间的扩展因子
    #             expand_text=(1.2, 1.2),    # 文本之间的扩展因子
    #             force_points=(0.2, 0.5),   # 推动点的力度
    #             force_text=(0.2, 0.5),     # 推动文本的力度
    #             arrowprops=dict(arrowstyle='-', color='black', linewidth=0.5),  # 调整箭头样式
    #             avoid_self=True,           # 避免文本与自身重叠
    #             autoalign='xy')            # 根据 x 和 y 方向自动对齐

    # 设置图形属性
    ax.set_ylabel('t-statistic', fontsize=9, labelpad=5, color='black')
    ax.set_title(f'Subtype {i} vs. rest\n{sig_count} significant', 
                 fontsize=9, pad=10, color='black')
    ax.set_xlim(0, len(covariate_names) + 1)
    ax.set_ylim(-80,80)

    ax.grid(False)
    ax.tick_params(direction='in')
    ax.axhline(0, color='black', linewidth=1.0)
    ax.set_xticks([])

    # 在第一个 subplot 添加图例
    if i == 1:
        handles = [plt.Line2D([0], [0], marker='o', color=category_colors[cat], 
                             markersize=6, linestyle='', label=textwrap.fill(cat, 20)) 
                  for cat in unique_categories]
        ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(-0.55, 1.1),
                  frameon=True, framealpha=1, edgecolor='black',
                  title='Categories', title_fontsize=9, fontsize=8)
    
    textalloc.allocate(ax,text_x,text_y,
            text_abbr,
            x_scatter=x, y_scatter=y,
            priority_strategy=1,
            textsize=6,
            linecolor='black', avoid_label_lines_overlap=True, avoid_crossing_label_lines=True, draw_all=False)

fig.text(0.5, 0.04, 'Colored by Category, Labeled: Top significant, Bonferroni corrected p<0.05',
         ha='center', fontsize=9, color='black')

plt.savefig(f'{OUTPUT_DIR}/t_stats_blood_chem_horizontal.png', dpi=300, bbox_inches='tight', facecolor='white')

plt.close()
