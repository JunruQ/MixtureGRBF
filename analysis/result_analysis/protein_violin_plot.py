import pandas as pd
import numpy as np
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import kruskal
import scikit_posthocs as sp

# ===== Step 1: 加载和预处理数据 (使用您提供的代码) =====

# --- 设置基本变量 ---
nsubtype = 5
result_folder = 'ukb_MixtureGRBF_cv_nsubtype_biom17'
output_dir = f'./output/result_analysis/{result_folder}/{nsubtype}_subtypes'

# --- 加载亚型和分期数据 ---
subtype_stage_path = f'./output/{result_folder}/{nsubtype}_subtypes/subtype_stage.csv'
subtype_stage = pd.read_csv(subtype_stage_path)

# --- 加载并应用亚型排序 ---
SUBTYPE_ORDER_PATH = f'output/result_analysis/{result_folder}/{nsubtype}_subtypes/all_cause_mortality_order.csv'
try:
    subtype_order_df = pd.read_csv(SUBTYPE_ORDER_PATH, header=None)
    subtype_order = subtype_order_df.iloc[:, 0].tolist()
except FileNotFoundError:
    warnings.warn("亚型排序文件未找到，将使用默认顺序。")
    subtype_order = list(range(1, nsubtype + 1))

subtype_mapping = {subtype: i + 1 for i, subtype in enumerate(subtype_order)}
subtype_stage['subtype'] = subtype_stage['subtype'].map(subtype_mapping).fillna(np.nan)
subtype_stage = subtype_stage.dropna(subset=['subtype'])
subtype_stage['subtype'] = subtype_stage['subtype'].astype(int)


# --- 加载蛋白质数据 ---
protein_path = 'input/ukb/ukb_covreg2_trans1_nanf1_biom0.csv'
protein_df = pd.read_csv(protein_path)

# ===== Step 2: 整合数据 =====

# --- 解决 PTID 和 RID 的差异 ---
# 将 subtype_stage 中的 'PTID' 重命名为 'RID' 以便合并
subtype_stage.rename(columns={'PTID': 'eid'}, inplace=True)

# --- 提取所需的蛋白质数据并与亚型数据合并 ---
# 选择目标蛋白质和个体标识符
target_proteins = ['GFAP', 'NEFL']
protein_subset_df = protein_df[['eid'] + target_proteins]

# 基于 'RID' 合并数据
merged_df = pd.merge(subtype_stage, protein_subset_df, on='eid', how='inner')

# --- 将宽格式数据转换为长格式，以便于绘图和统计 ---
plot_df = merged_df.melt(
    id_vars=['eid', 'subtype'],
    value_vars=target_proteins,
    var_name='Protein',
    value_name='Expression'
)

# 确保亚型是有序的类别，以便绘图时按指定顺序排列
ordered_subtypes = sorted(subtype_stage['subtype'].unique())

print("数据准备完成，合并后的数据包含 {} 个样本。".format(merged_df.shape[0]))
print(f"亚型顺序为: {ordered_subtypes}\n")


# ===== Step 3: 执行统计检验 =====

# 对每种蛋白质分别进行检验
for protein in target_proteins:
    print(f"--- 正在对 {protein} 进行统计分析 ---")
    
    # --- Kruskal-Wallis H-检验 ---
    # 准备一个列表，其中每个元素是该亚型下蛋白质表达值的序列
    groups = [merged_df[merged_df['subtype'] == i][protein].dropna() for i in ordered_subtypes]
    
    # 执行检验
    h_stat, p_val = kruskal(*groups)
    
    print(f"Kruskal-Wallis 检验结果:")
    print(f"H-statistic: {h_stat:.4f}")
    print(f"P-value: {p_val:.4e}\n")

    # --- Dunn's 事后检验 (如果Kruskal结果显著) ---
    if p_val < 0.05:
        print(f"Kruskal-Wallis 检验显著，执行 Dunn's 事后检验...")
        
        # 执行Dunn检验，自动进行多重比较校正（默认为Benjamini-Hochberg）
        dunn_results = sp.posthoc_dunn(merged_df, val_col=protein, group_col='subtype', p_adjust='bonferroni')
        
        print("Dunn's 检验 p-值 (Bonferroni 校正后):")
        # 将亚型标签设置为列名和索引名
        dunn_results.columns = [f"Subtype {i}" for i in ordered_subtypes]
        dunn_results.index = [f"Subtype {i}" for i in ordered_subtypes]
        print(dunn_results)
        print("-" * 50 + "\n")
    else:
        print("Kruskal-Wallis 检验不显著，无需进行事后检验。")
        print("-" * 50 + "\n")


# ===== Step 4: 可视化 =====

# 设置绘图风格
sns.set(style="ticks", font_scale=1.2)

# 创建一个 1x2 的子图网格
fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=False)
fig.suptitle('Expression Difference of GFAP & NEFL', fontsize=20, y=1.02)

# 调色板
palette = "viridis"

# --- 绘制 GFAP 的小提琴图 ---
sns.violinplot(ax=axes[0], data=plot_df[plot_df['Protein'] == 'GFAP'],
               x='subtype', y='Expression', order=ordered_subtypes,
               palette=palette, inner='quartile', cut=0)
sns.stripplot(ax=axes[0], data=plot_df[plot_df['Protein'] == 'GFAP'],
              x='subtype', y='Expression', order=ordered_subtypes,
              color='black', size=1.5, alpha=0.3)
axes[0].set_title('GFAP Expression', fontsize=16)
axes[0].set_xlabel('Subtype', fontsize=12)
axes[0].set_ylabel('Protein Expression Level', fontsize=12)

# --- 绘制 NEFL 的小提琴图 ---
sns.violinplot(ax=axes[1], data=plot_df[plot_df['Protein'] == 'NEFL'],
               x='subtype', y='Expression', order=ordered_subtypes,
               palette=palette, inner='quartile', cut=0)
sns.stripplot(ax=axes[1], data=plot_df[plot_df['Protein'] == 'NEFL'],
              x='subtype', y='Expression', order=ordered_subtypes,
              color='black', size=1.5, alpha=0.3)
axes[1].set_title('NEFL Expression', fontsize=16)
axes[1].set_xlabel('Subtype', fontsize=12)
axes[1].set_ylabel('Protein Expression Level', fontsize=12)


# 优化布局并保存图像
plt.tight_layout(rect=[0, 0, 1, 0.96])
save_path = f"{output_dir}/GFAP_NEFL_violin_plots.png"
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"图表已保存至: {save_path}")

plt.show()