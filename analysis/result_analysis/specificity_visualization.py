import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np

# 文件路径
input_file = "output/result_analysis/ukb_MixtureGRBF_cv_nsubtype_biom17/5_subtypes/specificity_t_stats.csv"
specificity_file = "preprocess/hbca/adata_specificity.csv"
var_file = "preprocess/hbca/adata_var.csv"
output_plot = "output/result_analysis/ukb_MixtureGRBF_cv_nsubtype_biom17/5_subtypes/combined_heatmap.png"

# 创建输出目录
os.makedirs(os.path.dirname(output_plot), exist_ok=True)

# 1. 处理第一个热力图数据（t-statistic）
print("加载 specificity_t_stats.csv 数据...")
data = pd.read_csv(input_file)
print("数据维度:", data.shape)

# 检查必要列
required_cols = ["subtype", "cluster_id", "t_statistic"]
if not all(col in data.columns for col in required_cols):
    raise ValueError("未找到必要列：subtype, cluster_id, t_statistic")

# 数据重塑为矩阵（行：subtype，列：cluster_id）
data_matrix = data.pivot(index="subtype", columns="cluster_id", values="t_statistic").fillna(0)
print("t-statistic 数据矩阵维度:", data_matrix.shape)

# 计算颜色范围
max_abs_t = np.abs(data_matrix).max().max()
print("最大绝对 t 统计量:", max_abs_t)

# 2. 处理第二个热力图数据（specificity）
print("加载 adata_specificity.csv 和 adata_var.csv 数据...")
specificity_df = pd.read_csv(specificity_file)
var_df = pd.read_csv(var_file)

# 检查 Gene 列
if "Gene" not in var_df.columns:
    raise ValueError("'Gene' 列未在 adata_var.csv 中找到")

# 创建映射
print("映射 specificity 列名为基因名...")
gene_mapping = dict(zip(var_df["Unnamed: 0"].astype(str), var_df["Gene"]))
specificity_df.columns = [gene_mapping.get(col, col) for col in specificity_df.columns]
print("映射后的前几个列名:", specificity_df.columns[:5].tolist())

# 筛选关注的基因
genes_of_interest = ["INA", "SLC17A6", "SLC17A7", "SLC32A1", "PTPRC", "CLDN5", "ACTA2", 
                    "LUM", "PDGFRA", "SOX10", "PLP1", "AQP4", "FOXJ1", "TTR"]
print("筛选关注的基因:", genes_of_interest)

# 检查缺失基因
missing_genes = [gene for gene in genes_of_interest if gene not in specificity_df.columns]
if missing_genes:
    print("警告: 以下基因未在数据中找到:", missing_genes)

# 假设 specificity_df 的行索引是 cluster_id，转换为矩阵
specificity_df["cluster_id"] = specificity_df.index.astype(str)
specificity_matrix = specificity_df[["cluster_id"] + [i for i in genes_of_interest if i not in missing_genes]].set_index("cluster_id").T


# 计算颜色范围
max_spec = specificity_matrix.max().max()
print("最大 specificity 值:", max_spec)

# 3. 绘制热力图
print("创建热力图...")

# 创建图形
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True, 
                               gridspec_kw={'height_ratios': [len(genes_of_interest) - len(missing_genes), len(data_matrix)]})

# 第二个热力图（specificity，上方）
sns.heatmap(
    specificity_matrix,
    cmap="Blues",  # 单一颜色映射（白到深蓝）
    vmin=0, vmax=max_spec,
    ax=ax1,
    cbar_kws={'label': 'Specificity'},
    xticklabels=10  # 每 10 个 cluster_id 显示一个标签
)
ax1.set_title("Specificity Heatmap (Selected Genes)")
ax1.set_xlabel("")
ax1.set_ylabel("Gene")

# 第一个热力图（t-statistic，下方）
sns.heatmap(
    data_matrix,
    cmap=sns.diverging_palette(220, 20, as_cmap=True),  # 蓝白红，区分正负
    vmin=-max_abs_t, vmax=max_abs_t,
    center=0,
    ax=ax2,
    cbar_kws={'label': 't-statistic'},
    xticklabels=10
)
# ax2.set_title("T-Statistic Heatmap")
ax2.set_xlabel("Cluster ID")
ax2.set_ylabel("Subtype")

# 调整 x 轴标签
for ax in [ax1, ax2]:
    ax.tick_params(axis='x', rotation=45, labelsize=6)
ax1.tick_params(axis='y', labelsize=8)
ax2.tick_params(axis='y', labelsize=10)

# 调整布局
plt.tight_layout()

# 4. 保存图形
print("保存图形到", output_plot)
plt.savefig(output_plot, dpi=300, bbox_inches='tight')
plt.close()

print("可视化完成。")