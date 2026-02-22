import loompy
import pandas as pd
import gseapy as gp
import os

# === Step 1: 从 loom 文件中提取表达量 >10 的通路集合 ===
def load_pathways_from_loom(loom_path, background_protein, threshold=10):
    print("正在加载 loom 文件...")
    with loompy.connect(loom_path, "r") as ds:
        mean_x = ds[:, :].T  # (n_clusters, n_genes)
        gene_names = ds.ra['Gene']  # 假设 row attribute 名叫 'Gene'
    
    df = pd.DataFrame(mean_x, columns=gene_names)
    df = df[background_protein]

    # 行和归一化为10000
    counts = df.sum(axis=1)
    counts[counts == 0.] = 1.0  # 避免除以0
    scaling_factor = 10000. / counts
    df = df.mul(scaling_factor, axis=0)

    pathways = {}
    for idx, row in df.iterrows():
        cluster_name = f"cluster_{idx}"
        expressed_genes = row[row > threshold].index.tolist()
        pathways[cluster_name] = expressed_genes
    return pathways, set(gene_names)  # 同时返回合法基因名集合

# === Step 2: 加载 t_statistic 表，并构造每个 subtype 的基因排名列表 ===
def load_rankings(csv_path, valid_genes=None):
    tstat_df = pd.read_csv(csv_path)
    rankings_by_subtype = {}
    for subtype, group in tstat_df.groupby("subtype"):
        rnk = group.set_index("biom")["t_statistic"]
        if valid_genes is not None:
            rnk = rnk[rnk.index.isin(valid_genes)]  # 保证基因存在于 loom 中
        rnk = rnk.sort_values(ascending=False)
        rankings_by_subtype[subtype] = rnk
    return rankings_by_subtype

# === Step 3: 对每个 subtype 执行 GSEA ===

def run_gsea_for_subtypes(rankings_by_subtype, pathways, outdir_base="output/result_analysis/ukb_MixtureGRBF_cv_nsubtype_biom17/5_subtypes", processes=-1, output_csv="brain_cell_gsea_NES_matrix.csv"):
    os.makedirs(outdir_base, exist_ok=True)

    all_clusters = [f"cluster_{i}" for i in range(len(pathways))]  # 所有 cluster 名称
    nes_df = pd.DataFrame(index=all_clusters)  # 初始化 NES 表格，行是 cluster

    for subtype, rnk in rankings_by_subtype.items():
        print(f"\n🔍 Running GSEA for subtype: {subtype} ({len(rnk)} genes)")
        result = gp.prerank(
            rnk=rnk,
            gene_sets=pathways,
            processes=processes,
            permutation_num=1000,
            outdir=None,
            format=None,
            seed=42,
            min_size=5,
            max_size=5000,
            verbose=True,
            no_plot=True
        )
        res = result.res2d  # DataFrame with NES, FDR, etc.

        # 建立 NES Series，自动对齐 cluster
        nes_series = res["NES"]
        nes_series.index = res["Term"]
        nes_series.name = subtype

        # 合并到总 NES 矩阵中
        nes_df = nes_df.join(nes_series, how="left")

    # 保存最终 NES 表
    nes_df.to_csv(os.path.join(outdir_base, output_csv), index=False)
    print(f"\n✅ NES matrix saved to {os.path.join(outdir_base, output_csv)}")

    return nes_df  # 可选返回 DataFrame 用于进一步分析

# === 主程序入口 ===
if __name__ == "__main__":
    loom_path = "/data/datasets/human_brain_cell_atlas/adult_human_20221007.agg.loom"
    tstat_csv = "output/result_analysis/ukb_MixtureGRBF_cv_nsubtype_biom17/5_subtypes/t_stats_by_subtype.csv"
    background_protein_path = 'preprocess/hbca/adata_var.csv'
    background_protein = pd.read_csv(background_protein_path)['Gene'].values

    # Step 1: 构造通路集
    pathways, valid_genes = load_pathways_from_loom(loom_path, background_protein)

    # Step 2: 构造每个 subtype 的排名列表
    rankings_by_subtype = load_rankings(tstat_csv, valid_genes)

    # Step 3: 执行 GSEA
    run_gsea_for_subtypes(rankings_by_subtype, pathways)
