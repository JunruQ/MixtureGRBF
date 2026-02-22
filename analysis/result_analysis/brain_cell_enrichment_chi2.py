import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import loompy
from statsmodels.stats.multitest import multipletests
import seaborn as sns
import matplotlib.pyplot as plt

def load_high_expr_genes_from_loom(loom_path, background_protein, threshold=10):
    with loompy.connect(loom_path, "r") as ds:
        mean_x = ds[:, :].T  # shape: (n_clusters, n_genes)
        gene_names = ds.ra['Gene']
    
    df = pd.DataFrame(mean_x, columns=gene_names)
    df = df[background_protein]

    # 行和归一化为10000
    counts = df.sum(axis=1)
    counts[counts == 0.] = 1.0  # 避免除以0
    scaling_factor = 10000. / counts
    df = df.mul(scaling_factor, axis=0)

    high_expr_data = []

    for i, row in df.iterrows():
        cluster_name = i
        high_genes = row[row > threshold].index.tolist()
        for gene in high_genes:
            high_expr_data.append({'tissue': cluster_name, 'gene': gene})
    
    return pd.DataFrame(high_expr_data)

background_protein_path = 'preprocess/hbca/adata_var.csv'
background_protein = pd.read_csv(background_protein_path)['Gene'].values

exp_path = 'output/result_analysis/ukb_MixtureGRBF_cv_nsubtype_biom17/5_subtypes/'
t_stats_path = exp_path + 't_stats_by_subtype.csv'
t_stats = pd.read_csv(t_stats_path, index_col=0)
t_stats = t_stats[t_stats.index.isin(background_protein)]

loom_path = '/data/datasets/human_brain_cell_atlas/adult_human_20221007.agg.loom'
cell_expr = load_high_expr_genes_from_loom(loom_path, background_protein, threshold=10)

for top_n in [100,200,500]:
    # top_n = 200
    results = []

    for subtype in t_stats['subtype'].unique():
        t_sub = t_stats[t_stats['subtype'] == subtype]
        top_genes = t_sub['t_statistic'].abs().sort_values(ascending=False).head(top_n).index.tolist()

        for cell in cell_expr['tissue'].unique():
            high_expr_genes = cell_expr[cell_expr['tissue'] == cell]['gene'].unique()

            in_top_and_expr = len(set(top_genes) & set(high_expr_genes))
            in_top_not_expr = len(set(top_genes) - set(high_expr_genes))
            not_in_top_and_expr = len(set(high_expr_genes) - set(top_genes))
            not_in_top_not_expr = len(set(background_protein)) - (in_top_and_expr + in_top_not_expr + not_in_top_and_expr)

            contingency = np.array([
                [in_top_and_expr, in_top_not_expr],
                [not_in_top_and_expr, not_in_top_not_expr]
            ])

            chi2, p, dof, expected = chi2_contingency(contingency)

            results.append({
                'subtype': subtype,
                'tissue': cell,
                'chi2_stat': chi2,
                'p_value': p,
                'overlap': in_top_and_expr
            })

    chi_df = pd.DataFrame(results)
    chi_df['p_adj'] = multipletests(chi_df['p_value'], method='fdr_bh')[1]
    chi_df.to_csv(exp_path + f'brain_cell_chi_square_{top_n}.csv', index=False)

