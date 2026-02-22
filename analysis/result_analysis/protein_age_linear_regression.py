import pandas as pd
import utils.utils as utils
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from venn import venn
import matplotlib.pyplot as plt

exp_name = 'ukb_MixtureGRBF_cv_nsubtype_biom17'
nsubtype = 5
top_n = 100
subtype_stage = utils.get_subtype_stage(exp_name, nsubtype=nsubtype).rename(columns={'PTID': 'eid'})

protein_table = pd.read_csv('input/ukb/ukb_covreg1_trans1_nanf1_biom0.csv')

n_covs = 7

features = protein_table.iloc[:, n_covs:].columns

results = []

for fea in features:
    y_all = protein_table[fea]
    for subtype in range(1, nsubtype + 1):
        subtype_ids = subtype_stage[subtype_stage['subtype'] == subtype]['eid']
        mask = protein_table['RID'].isin(subtype_ids)
        y = y_all[mask]
        x = protein_table.loc[mask, 'stage']
        # add constant
        X = sm.add_constant(x)
        model = sm.OLS(y, X).fit()

        # beta and p
        beta = model.params.iloc[1]
        pvalue = model.pvalues.iloc[1]
        results.append({'protein': fea, 'subtype': subtype, 'beta': beta, 'pvalue': pvalue})

df = pd.DataFrame(results)

# Apply multiple testing correction per subtype
df['qvalue'] = 0.0
for subtype in range(1, nsubtype + 1):
    sub_df = df[df['subtype'] == subtype]
    if len(sub_df) > 0:
        pvals = sub_df['pvalue'].values
        _, p_corrected, _, _ = multipletests(pvals, method='fdr_bh', alpha=0.05)
        df.loc[df['subtype'] == subtype, 'qvalue'] = p_corrected
# Bonf

df['p_corrected'] = 0.0
for subtype in range(1, nsubtype + 1):
    sub_df = df[df['subtype'] == subtype]
    if len(sub_df) > 0:
        pvals = sub_df['pvalue'].values
        _, p_corrected, _, _ = multipletests(pvals, method='bonferroni', alpha=0.05)
        df.loc[df['subtype'] == subtype, 'p_corrected'] = p_corrected

output_dir = f'output/result_analysis/{exp_name}/{nsubtype}_subtypes'
df.to_csv(f'{output_dir}/protein_age_linear_regression_results.csv', index=False)

# Select top_n proteins per subtype
top_proteins = {}
for subtype in range(1, nsubtype + 1):
    sub_df = df[df['subtype'] == subtype].copy()
    if len(sub_df) >= top_n:
        top_prots = sub_df.nsmallest(top_n, 'qvalue')['protein'].tolist()
    else:
        top_prots = sub_df['protein'].tolist()
    top_proteins[f'Subtype {subtype}'] = set(top_prots)

# Draw Venn diagram
venn(top_proteins)
plt.savefig(f'{output_dir}/protein_age_linear_regression_venn.png', dpi=300)