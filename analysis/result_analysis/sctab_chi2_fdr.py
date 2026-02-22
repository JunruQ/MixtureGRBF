import pandas as pd
import numpy as np
import gseapy as gp
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm
import textalloc
from adjustText import adjust_text
from scipy.stats import chi2_contingency
from statsmodels.stats.multitest import multipletests

# ===== Read and Prepare Data =====

# Load background protein list
background_protein_path = 'preprocess/cellxgene_data/adata_var.csv'
background_protein = pd.read_csv(background_protein_path)['feature_name'].values

print(f'Number of background proteins: {len(background_protein)}')
# Load t-statistics
exp_path = 'output/result_analysis/ukb_MixtureGRBF_cv_nsubtype_biom17/5_subtypes/'
t_stats_path = exp_path + 't_stats_by_subtype.csv'
t_stats = pd.read_csv(t_stats_path, index_col=0)
t_stats = t_stats[t_stats.index.isin(background_protein)] # Filter for background proteins

# Load cell type specific gene sets
cell_expr_path = 'preprocess/cellxgene_data/high_expr_genes.csv'
cell_expr = pd.read_csv(cell_expr_path, index_col=0)
cell_expr = cell_expr.apply(lambda x: x.str.split(';').explode()).reset_index()
cell_expr.columns = ['cell_type', 'gene']

# Create a dictionary of gene sets for GSEA
# Each key is a cell type, and the value is a list of highly expressed genes
gene_sets = cell_expr.groupby('cell_type')['gene'].apply(list).to_dict()

# ===== Perform GSEA Analysis =====

# ===== 参数设置 =====
top_n = 200
results = []

# ===== 卡方富集分析 =====
for subtype in t_stats['subtype'].unique():
    t_sub = t_stats[t_stats['subtype'] == subtype]
    top_genes = t_sub['t_statistic'].abs().sort_values(ascending=False).head(top_n).index.tolist()

    for cell in cell_expr['cell_type'].unique():
        high_expr_genes = cell_expr[cell_expr['cell_type'] == cell]['gene'].unique()

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
            'cell_type': cell,
            'chi2_stat': chi2,
            'p_value': p,
            'overlap': in_top_and_expr
        })

# ===== 整理结果 =====
chi_df = pd.DataFrame(results)
print(f"Number of tests: {len(chi_df['p_value'])}")
chi_df['fdr'] = multipletests(chi_df['p_value'], method='fdr_bh')[1]
chi_df['neg_log10_fdr'] = -np.log10(chi_df['fdr'].astype(float))
chi_df.to_csv(exp_path + 'sctab_chi_square.csv', index=False)


# gsea_results = []

# # Iterate over each subtype to perform GSEA
# for subtype in t_stats['subtype'].unique():
#     # Prepare the ranked list of genes for the current subtype
#     # The list is ranked by the t-statistic
#     rankings = t_stats[t_stats['subtype'] == subtype]['t_statistic'].sort_values(ascending=False)
    
#     # Run GSEA using the prerank function
#     gsea_result = gp.prerank(
#         rnk=rankings,
#         gene_sets=gene_sets,
#         min_size=5, # Minimum size of a gene set to be considered
#         permutation_num=1000, # Number of permutations for significance testing
#         seed=42,
#         verbose=True
#     ).res2d
    
#     # Store the results for the current subtype
#     gsea_result['subtype'] = subtype
#     gsea_results.append(gsea_result)

# # Concatenate all GSEA results into a single DataFrame
# gsea_df = pd.concat(gsea_results)

# # Rename columns for consistency and clarity
# gsea_df = gsea_df.rename(columns={'Term': 'cell_type', 'FDR q-val': 'fdr'})

# # Calculate -log10(FDR) for plotting
# # Clipping the lower bound of FDR to avoid log(0) errors
# gsea_df['neg_log10_fdr'] = -np.log10(gsea_df['fdr'].astype(float).clip(lower=1e-300))

# ===== Visualize GSEA Results =====

# Define the cell type mapping for categorization and coloring
cell_type_mapping = {
    "B cell related cells": [
        "B cell", "naive B cell", "memory B cell", "mature B cell", "plasma cell", "transitional stage B cell",
        "class switched memory B cell", "immature B cell", "IgA plasma cell", "IgG plasma cell", "plasmablast"
    ],
    "T cell related cells": [
        "T cell", "alpha-beta T cell", "memory T cell", "CD4-positive, alpha-beta T cell", "CD8-positive, alpha-beta T cell",
        "naive thymus-derived CD4-positive, alpha-beta T cell", "naive thymus-derived CD8-positive, alpha-beta T cell",
        "CD4-positive, alpha-beta memory T cell", "CD8-positive, alpha-beta memory T cell",
        "effector CD8-positive, alpha-beta T cell", "regulatory T cell", "T follicular helper cell",
        "CD4-positive helper T cell", 'CD4-positive, alpha-beta cytotoxic T cell',"CD8-positive, alpha-beta cytotoxic T cell", "mature alpha-beta T cell",
        "gamma-delta T cell", "mature gamma-delta T cell", "mucosal invariant T cell", "T-helper 22 cell",
        "CD8-alpha-alpha-positive, alpha-beta intraepithelial T cell", "double negative T regulatory cell",
        "central memory CD4-positive, alpha-beta T cell", "central memory CD8-positive, alpha-beta T cell",
        "effector memory CD4-positive, alpha-beta T cell", "effector memory CD8-positive, alpha-beta T cell",
        "effector memory CD8-positive, alpha-beta T cell, terminally differentiated",
        "activated CD4-positive, alpha-beta T cell", "activated CD8-positive, alpha-beta T cell",
        "double-positive, alpha-beta thymocyte", "double negative thymocyte"
    ],
    "NK cell related cells": [
        "natural killer cell", "mature NK T cell", "CD16-positive, CD56-dim natural killer cell, human",
        "CD16-negative, CD56-bright natural killer cell, human",
    ],
    "Other lymphocytes": [
        "lymphocyte","innate lymphoid cell", "immature innate lymphoid cell", "lymphoid lineage restricted progenitor cell"
    ],
    "Mononuclear phagocytes": [
        "monocyte", "classical monocyte", "non-classical monocyte", "intermediate monocyte",
        "CD14-positive, CD16-negative classical monocyte", "CD14-positive, CD16-positive monocyte",
        "CD14-low, CD16-positive monocyte", "CD14-positive monocyte", "macrophage", "alveolar macrophage",
        "elicited macrophage", "lung macrophage", "inflammatory macrophage", "alternatively activated macrophage",
        "central nervous system macrophage"
    ],
    "Dendritic cells": [
        "dendritic cell", "CD1c-positive myeloid dendritic cell", "plasmacytoid dendritic cell",
        "conventional dendritic cell"
    ],
    "Other immune cells": [
        "granulocyte", "neutrophil", "mast cell", "leukocyte"
    ],
    "Hematopoietic cells": [
        "erythrocyte", "platelet", "hematopoietic stem cell"
    ],
    "Respiratory epithelium": [
        "tracheal goblet cell", "club cell", "type I pneumocyte", "type II pneumocyte",
        "ciliated columnar cell of tracheobronchial tree", "respiratory basal cell", "respiratory hillock cell",
        "nasal mucosa goblet cell", "goblet cell"
    ],
    "Renal epithelium": [
        "kidney collecting duct intercalated cell", "kidney collecting duct principal cell",
        "kidney connecting tubule epithelial cell", "epithelial cell of proximal tubule",
        "kidney distal convoluted tubule epithelial cell", "kidney loop of Henle thick ascending limb epithelial cell",
        "kidney loop of Henle thin descending limb epithelial cell", "kidney loop of Henle thin ascending limb epithelial cell"
    ],
    "Digestive epithelium": [
        "enterocyte", "intestine goblet cell", "enteroendocrine cell", "foveolar cell of stomach", "mucous neck cell"
    ],
    "Other epithelium": [
        "keratinocyte", "luminal epithelial cell of mammary gland", "myoepithelial cell of mammary gland", "basal cell"
    ],
    "Endothelial cells": [
        "endothelial cell", "blood vessel endothelial cell", "vein endothelial cell", "endothelial cell of artery",
        "capillary endothelial cell", "endothelial cell of lymphatic vessel", "pulmonary artery endothelial cell"
    ],
    "Neurons": [
        "neuron", "L2/3-6 intratelencephalic projecting glutamatergic cortical neuron",
        "L6b glutamatergic cortical neuron", "near-projecting glutamatergic cortical neuron",
        "corticothalamic-projecting glutamatergic cortical neuron",
        "caudal ganglionic eminence derived GABAergic cortical interneuron", "lamp5 GABAergic cortical interneuron",
        "sst GABAergic cortical interneuron", "pvalb GABAergic cortical interneuron",
        "vip GABAergic cortical interneuron", "sncg GABAergic cortical interneuron",
        "chandelier pvalb GABAergic cortical interneuron", "retinal ganglion cell", "amacrine cell",
        "retina horizontal cell"
    ],
    "Neuroglia": [
        "astrocyte", "astrocyte of the cerebral cortex", "oligodendrocyte", "oligodendrocyte precursor cell",
        "microglial cell", "ependymal cell", "Bergmann glial cell", "Schwann cell"
    ],
    "Retinal cells": [
        "retinal rod cell", "retinal cone cell"
    ],
    "Fibroblasts": [
        "fibroblast of cardiac tissue", "bronchus fibroblast of lung", "alveolar type 1 fibroblast cell",
        "alveolar type 2 fibroblast cell", "kidney interstitial fibroblast"
    ],
    "Pericytes": [
        "pericyte", "lung pericyte", "renal interstitial pericyte"
    ],
    "Smooth muscle cells": [
        "smooth muscle cell", "vascular associated smooth muscle cell", "tracheobronchial smooth muscle cell"
    ],
    "Other specialized cells": [
        "mesothelial cell"
    ]
}

cell_type_order = []
category_colors = {}
color_map = cm.get_cmap('tab20', len(cell_type_mapping))
for i, (category, cell_types) in enumerate(cell_type_mapping.items()):
    cell_type_order.extend(cell_types)
    for cell_type in cell_types:
        category_colors[cell_type] = color_map(i / len(cell_type_mapping))

# Ensure all cell types in the GSEA results are in the mapping
chi_df_filtered = chi_df[chi_df['cell_type'].isin(cell_type_order)]

# --- Plotting ---
subtypes = chi_df_filtered['subtype'].unique()
n_subtypes = len(subtypes)

plt.rcParams.update({
    'font.family': 'sans-serif',  # 指定字体家族为无衬线
    'font.sans-serif': ['Arial'], # 在无衬线字体列表中首选 Arial
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10
})

# Create subplots for each subtype with space for a legend at the top
fig = plt.figure(figsize=(33/2.54, (2 * n_subtypes + 1.5)))
gs = fig.add_gridspec(n_subtypes + 1, 1, height_ratios=[1] + [3] * n_subtypes)
axes = [fig.add_subplot(gs[i + 1, 0]) for i in range(n_subtypes)]

if n_subtypes == 1:
    axes = [axes]

# Generate a plot for each subtype
for i, subtype in enumerate(subtypes):
    ax = axes[i]
    subtype_data = chi_df_filtered[chi_df_filtered['subtype'] == subtype]
    
    # Create the scatter plot
    for cell_type in cell_type_order:
        if cell_type in subtype_data['cell_type'].values:
            data_point = subtype_data[subtype_data['cell_type'] == cell_type]
            ax.scatter(
                cell_type,
                data_point['neg_log10_fdr'].iloc[0],
                color=category_colors[cell_type],
                s=60
            )
            
    # Prepare data for text labels to avoid overlap
    present_cell_types = [ct for ct in cell_type_order if ct in subtype_data['cell_type'].values]
    ct_to_x = {ct: i for i, ct in enumerate(present_cell_types)}
    
    all_x = [ct_to_x[ct] for ct in present_cell_types]
    all_y = [subtype_data[subtype_data['cell_type'] == ct]['neg_log10_fdr'].iloc[0] for ct in present_cell_types]
    
    # Select the top N points to label based on -log10(FDR)
    n_labels = 3
    top_n_data = subtype_data.nlargest(n_labels, 'neg_log10_fdr')
    print(top_n_data)
    label_x = [ct_to_x[ct] for ct in top_n_data['cell_type']]
    print(label_x)
    label_y = top_n_data['neg_log10_fdr'].tolist()
    label_texts = top_n_data['cell_type'].tolist()
    
    # # Use textalloc to intelligently place labels
    # textalloc.allocate(
    #     ax=ax,
    #     x=label_x[::-1],
    #     y=label_y[::-1],
    #     text_list=label_texts[::-1],
    #     x_scatter=all_x,
    #     y_scatter=all_y,
    #     linecolor='black',
    #     xlims=(10, ax.get_xlim()[1])
    # )
    
    # Customize plot aesthetics
    ax.axhline(y=-np.log10(0.05), color='gray', linestyle='--', linewidth=1.5, label="FDR < 0.05")
    ax.text(0.01, 0.95, f'Subtype {subtype}', transform=ax.transAxes, fontsize=14, verticalalignment='top', fontweight='bold')
    ax.set_ylabel('-log10(FDR)')
    ax.set_xticks([])
    ax.set_xticklabels([])
    # ax.grid(True, axis='y', linestyle='--', alpha=0.6)

# Label the x-axis on the last plot
axes[-1].set_xlabel('Cell Type', fontdict={'fontsize': 12})

# Create a legend in the dedicated top axes
legend_ax = fig.add_subplot(gs[0, 0])
legend_ax.axis('off')
handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map(i / len(cell_type_mapping)), markersize=12) 
           for i in range(len(cell_type_mapping))]
legend = legend_ax.legend(handles, cell_type_mapping.keys(), loc='center', ncol=5, 
                          bbox_to_anchor=(0.5, 0.5), frameon=False, fontsize=11)

# Adjust layout and save the figure
plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust rect to make space for the main title if needed
plt.savefig(exp_path + 'cell_type_chi2_fdr_scatter.png', bbox_inches='tight', dpi=500)

