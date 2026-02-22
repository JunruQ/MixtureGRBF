import pandas as pd
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np

def bonferroni_correction(p_values):
    n = len(p_values)
    corrected_p_values = np.array(p_values) * n
    return corrected_p_values

nsubtype = 4
INPUT_TABLE_PATH = 'input/ukb/ukb_covreg1_trans1_nanf1_biom0.csv'
SUBTYPE_STAGE_PATH = f'output/ukb_MixtureGRBF_cv_nsubtype_biom6/{nsubtype}_subtypes/subtype_stage.csv'
CLASSIFICATION_PATH = 'analysis/result_analysis/protein_classification_form.csv'

df = pd.read_csv(INPUT_TABLE_PATH)
df.rename(columns={'RID':'PTID'}, inplace=True)
subtype_stage = pd.read_csv(SUBTYPE_STAGE_PATH)
classification = pd.read_csv(CLASSIFICATION_PATH)
biomarker_names = df.iloc[:, 7:].columns.tolist()
df = pd.merge(df, subtype_stage, how='inner', on=['PTID', 'stage'])

for i in range(1, nsubtype+1):
    ps = []
    for biom in biomarker_names:
        group_1 = df.loc[df['subtype'] == i,biom]
        group_2 = df.loc[~(df['subtype'] == i),biom]
        _, p_value = stats.ttest_ind(group_1, group_2, equal_var=False)
        ps.append(p_value)
    ps = bonferroni_correction(ps)
    is_significant = ps <= 0.05
    significant_biom = np.array(biomarker_names)[is_significant].tolist()
    with open(f'analysis/result_analysis/output/subtype{i}_significant_biom.txt', 'w') as f:
        for biom in significant_biom:
            f.write(biom + '\n')
