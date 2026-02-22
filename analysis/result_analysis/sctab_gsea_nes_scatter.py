import pandas as pd
import numpy as np
import gseapy as gp
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm
import textalloc
from statsmodels.stats.multitest import multipletests

# ===== Read and Prepare Data (No Changes Here) =====

# Load background protein list
background_protein_path = 'preprocess/cellxgene_data/adata_var.csv'
background_protein = pd.read_csv(background_protein_path)['feature_name'].values

# Load t-statistics
exp_path = 'output/result_analysis/ukb_MixtureGRBF_cv_nsubtype_biom17/5_subtypes/'
t_stats_path = exp_path + 't_stats_by_subtype.csv'
t_stats = pd.read_csv(t_stats_path, index_col=0)
t_stats = t_stats[t_stats.index.isin(background_protein)]

# Load cell type specific gene sets
cell_expr_path = 'preprocess/cellxgene_data/high_expr_genes.csv'
cell_expr = pd.read_csv(cell_expr_path, index_col=0)
cell_expr = cell_expr.apply(lambda x: x.str.split(';').explode()).reset_index()
cell_expr.columns = ['cell_type', 'gene']

gene_sets = cell_expr.groupby('cell_type')['gene'].apply(list).to_dict()

# ===== Perform GSEA Analysis (No Changes Here) =====

gsea_results = []

for subtype in t_stats['subtype'].unique():
    rankings = t_stats[t_stats['subtype'] == subtype]['t_statistic'].sort_values(ascending=False)
    
    gsea_result = gp.prerank(
        rnk=rankings,
        gene_sets=gene_sets,
        min_size=5,
        permutation_num=1000,
        seed=42,
        verbose=True
    ).res2d
    
    gsea_result['subtype'] = subtype
    gsea_results.append(gsea_result)

gsea_df = pd.concat(gsea_results)
gsea_df = gsea_df.rename(columns={'Term': 'cell_type', 'FDR q-val': 'fdr', 'NES': 'NES'})


# ===== Visualize GSEA Results (MODIFIED SECTION) =====

# Define the cell type mapping for categorization and coloring
cell_type_mapping = {
    "B cell related cells": [
        "B cell", "naive B cell", "memory B cell", "plasma cell", "transitional stage B cell",
        "class switched memory B cell", "immature B cell", "IgA plasma cell", "IgG plasma cell", "plasmablast"
    ],
    "T cell related cells": [
        "T cell", "CD4-positive, alpha-beta T cell", "CD8-positive, alpha-beta T cell",
        "naive thymus-derived CD4-positive, alpha-beta T cell", "naive thymus-derived CD8-positive, alpha-beta T cell",
        "CD4-positive, alpha-beta memory T cell", "CD8-positive, alpha-beta memory T cell",
        "effector CD8-positive, alpha-beta T cell", "regulatory T cell", "T follicular helper cell",
        "CD4-positive helper T cell", "CD8-positive, alpha-beta cytotoxic T cell", "mature alpha-beta T cell",
        "gamma-delta T cell", "mature gamma-delta T cell", "mucosal invariant T cell", "T-helper 22 cell",
        "CD8-alpha-alpha-positive, alpha-beta intraepithelial T cell", "double negative T regulatory cell",
        "central memory CD4-positive, alpha-beta T cell", "central memory CD8-positive, alpha-beta T cell",
        "effector memory CD4-positive, alpha-beta T cell", "effector memory CD8-positive, alpha-beta T cell",
        "effector memory CD8-positive, alpha-beta T cell, terminally differentiated",
        "activated CD4-positive, alpha-beta T cell", "activated CD8-positive, alpha-beta T cell",
        "double-positive, alpha-beta thymocyte", "double negative thymocyte"
    ],
    "NK cell related cells": [
        "natural killer cell", "mature NK T cell", "CD16-negative, CD56-bright natural killer cell",
        "CD16-positive, CD56-dim natural killer cell"
    ],
    "Other lymphocytes": [
        "innate lymphoid cell", "immature innate lymphoid cell", "lymphoid lineage restricted progenitor cell"
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
        "nasal mucosa goblet cell"
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

# Create a list of cell types in the desired order for plotting
cell_type_order = []
category_colors = {}
color_map = cm.get_cmap('tab20', len(cell_type_mapping))
for i, (category, cell_types) in enumerate(cell_type_mapping.items()):
    cell_type_order.extend(cell_types)
    for cell_type in cell_types:
        category_colors[cell_type] = color_map(i / len(cell_type_mapping))

# Ensure all cell types in the GSEA results are in the mapping
gsea_df_filtered = gsea_df[gsea_df['cell_type'].isin(cell_type_order)]

# --- Plotting ---
subtypes = gsea_df_filtered['subtype'].unique()
n_subtypes = len(subtypes)

# Create subplots for each subtype with space for a legend at the top
fig = plt.figure(figsize=(14, 3 * n_subtypes + 1.5))
gs = fig.add_gridspec(n_subtypes + 1, 1, height_ratios=[1] + [3] * n_subtypes)
axes = [fig.add_subplot(gs[i + 1, 0]) for i in range(n_subtypes)]

if n_subtypes == 1:
    axes = [axes]

# Generate a plot for each subtype
for i, subtype in enumerate(subtypes):
    ax = axes[i]
    subtype_data = gsea_df_filtered[gsea_df_filtered['subtype'] == subtype].copy() # Use .copy() to avoid SettingWithCopyWarning
    
    # Create the scatter plot
    for cell_type in cell_type_order:
        if cell_type in subtype_data['cell_type'].values:
            data_point = subtype_data[subtype_data['cell_type'] == cell_type]
            # MODIFICATION: Use 'NES' for the y-axis
            ax.scatter(
                cell_type,
                data_point['NES'].iloc[0],
                color=category_colors[cell_type],
                s=60
            )
            
    # Prepare data for text labels to avoid overlap
    present_cell_types = [ct for ct in cell_type_order if ct in subtype_data['cell_type'].values]
    ct_to_x = {ct: i for i, ct in enumerate(present_cell_types)}
    
    all_x = [ct_to_x[ct] for ct in present_cell_types]
    # MODIFICATION: Use 'NES' for y positions
    all_y = [subtype_data[subtype_data['cell_type'] == ct]['NES'].iloc[0] for ct in present_cell_types]
    
    # MODIFICATION: Select top N points by ABSOLUTE NES to label
    n_labels = 3
    subtype_data['abs_NES'] = subtype_data['NES'].astype(float).abs()
    top_n_data = subtype_data.nlargest(n_labels, 'abs_NES')
    
    label_x = [ct_to_x[ct] for ct in top_n_data['cell_type']]
    label_y = top_n_data['NES'].tolist() # Use the original NES for label positioning
    label_texts = top_n_data['cell_type'].tolist()

    # Find the maximum absolute NES to determine the symmetrical range
    max_abs_nes = subtype_data['NES'].astype(float).abs().max()
    
    # Set the y-axis limits to be symmetrical around 0 with 10% padding
    # y_limit = max_abs_nes * 1.2
    # ax.set_ylim(-y_limit, y_limit)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1.5)
    if subtype_data['NES'].min() < 0:
        ax.axhline(y=-1, color='gray', linestyle='--', linewidth=1.5)
    if subtype_data['NES'].max() > 0:
        ax.axhline(y=1, color='gray', linestyle='--', linewidth=1.5)

    # Use textalloc to intelligently place labels
    textalloc.allocate(
        ax=ax,
        x=label_x,
        y=label_y,
        text_list=label_texts,
        x_scatter=all_x,
        y_scatter=all_y,
        linecolor='black',
        priority_strategy=0,
    )

    
    # MODIFICATION: Change reference line to y=0 and update y-axis label
    
    ax.text(0.99, 0.05, f'Subtype {subtype}', transform=ax.transAxes, fontsize=14, verticalalignment='bottom', horizontalalignment='right', fontweight='bold')
    ax.set_ylabel('Normalized Enrichment Score (NES)')
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.grid(True, axis='y', linestyle='--', alpha=0.6)

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
plt.tight_layout(rect=[0, 0, 1, 0.96])
# MODIFICATION: Change the output filename
plt.savefig(exp_path + 'cell_type_gsea_nes_scatter.png', bbox_inches='tight', dpi=300)
print(f"Figure saved to {exp_path + 'cell_type_gsea_nes_scatter.png'}")

plt.show()