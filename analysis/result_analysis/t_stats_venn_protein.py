import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from venn import venn  # 需要安装 python-venn 包：pip install venn

# 原有的参数设置保持不变
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['axes.labelcolor'] = 'black'
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['axes.linewidth'] = 1.0

nsubtype = 5
exp_name = 'ukb_MixtureGRBF_cv_nsubtype_biom17'

SUBTYPE_STAGE_PATH = f'output/{exp_name}/{nsubtype}_subtypes/subtype_stage.csv'
OUTPUT_DIR = f'output/result_analysis/{exp_name}/{nsubtype}_subtypes'
T_STAT_PATH = f'output/result_analysis/{exp_name}/{nsubtype}_subtypes/t_stats_by_subtype.csv'

def bonferroni_correction(p_values):
    n = len(p_values)
    corrected_p_values = np.array(p_values) * n
    return corrected_p_values

# 读取数据
import utils.utils as utils
df = utils.get_subtype_stage(exp_name, nsubtype)
t_stat = pd.read_csv(T_STAT_PATH)
biomarker_names = t_stat.loc[t_stat['subtype'] == 1, 'biom'].tolist()

alpha = 0.05

# 计算每个subtype的最显著蛋白
significant_proteins = {}
for i in range(1, nsubtype + 1):
    k = i
    ts = t_stat.loc[t_stat['subtype'] == k, 't_statistic'].tolist()
    case_group = df['subtype'] == k
    control_group = df['subtype'] != k
    
    # Bonferroni校正
    corrected_ps = t_stat.loc[t_stat['subtype'] == k, 'corrected_p_value'].tolist()
    
    # 创建包含biomarker、t值和校正后p值的列表
    biom_stats = [(biom, abs(ts[idx]), corrected_ps[idx]) 
                 for idx, biom in enumerate(biomarker_names)]
    
    # 按t值绝对值排序并取前100个（如果少于100则取全部）
    sorted_bioms = sorted(biom_stats, key=lambda x: x[1], reverse=True)
    top_100 = [x[0] for x in sorted_bioms[:min(100, len(sorted_bioms))]]
    
    # 存储到字典中，使用subtype编号作为键
    significant_proteins[f'Subtype {i}'] = set(top_100)

# 创建维恩图
plt.figure(figsize=(6/2.54, 6/2.54), dpi=300, facecolor='white')

venn(significant_proteins, 
     cmap=utils.subtype_cmap,
     alpha=0.6,         # 透明度
     fontsize=24,        # 标签字体大小
     legend_loc=None,    # 不显示图例
    #  legend_loc="upper right")
)

# 保存图片
plt.savefig(f'{OUTPUT_DIR}/venn_top100_proteins.png', 
            dpi=300, 
            bbox_inches='tight', 
            facecolor='white')
plt.close()

print(f"维恩图已保存至: {OUTPUT_DIR}/venn_top100_proteins.png")