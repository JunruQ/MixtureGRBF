import json
import pandas as pd
import os
from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt
import numpy as np
import utils.utils as utils
from matplotlib.font_manager import FontProperties

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 9
font_bold = FontProperties(family='Arial', weight='bold', size=9)

def parse_target_field(df: pd.DataFrame) -> pd.DataFrame:
    df_result = pd.DataFrame()
    df_result['eid'] = df['eid']
    if 'Field' in df.columns:
        df_result['field'] = df['Field'].apply(lambda x: x.split(' ')[1] if not pd.isna(x) else x)
    elif 'target_cancer' in df.columns:
        df_result['field'] = df['target_cancer']
    else:
        raise ValueError('Field not found in dataframe')
    df_result['bl2t'] = df['BL2Target_yrs']
    return df_result

def kmf_subtype(survival_time, event, subtype, label, ax, colors):
    kmf = KaplanMeierFitter()
    unique_subtypes = sorted(set(subtype.dropna()))
    for idx, sub in enumerate(unique_subtypes):
        mask = (subtype == sub)
        kmf.fit(durations=survival_time[mask], event_observed=event[mask], label=f'Subtype {sub}')
        kmf.plot_survival_function(ax=ax, color=colors[idx], linewidth=1)

    ax.set_title(label, fontproperties=font_bold)
    ax.set_xlim(left=0)
    ax.set_xlabel(None)

    legend = ax.get_legend()
    if legend:
        legend.remove()
    x_label = ax.get_xlabel()
    if x_label:
        x_label.remove()

'''
IO
'''
with open('preprocess/data/important_disease_cancer.json', 'r') as f:
    important_disease = json.load(f)

nsubtype = 5
result_folder = 'ukb_MixtureGRBF_cv_nsubtype_biom17'
output_dir = f'./output/result_analysis/{result_folder}/{nsubtype}_subtypes'
subtype_stage = utils.get_subtype_stage(result_folder, nsubtype)
os.makedirs(output_dir, exist_ok=True)
colors = utils.subtype_colors[:nsubtype]

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(24 / 2.54, 15 / 2.54))
axes = axes.flatten()

lines_for_legend = []

'''
Survival Analysis
'''
for i, (disease_name, disease_code) in enumerate(important_disease.items()):
    if i >= len(axes): 
        break
    disease_upper_level_code = disease_code[0][0]
    disease_info = pd.read_csv(f'./input/disease_info/{disease_upper_level_code}0.csv')
    df_s = pd.merge(subtype_stage, parse_target_field(disease_info), left_on='PTID', right_on='eid', how='left')

    event = df_s['field'].apply(lambda x: x in disease_code if not pd.isna(x) else False)
    survival_time = df_s['bl2t'] + df_s['stage']
    subtype = df_s['subtype']
    kmf_subtype(survival_time, event, subtype, disease_name, axes[i], colors)

    if i == 0:
        for idx, sub in enumerate(sorted(set(subtype.dropna()))):
            lines_for_legend.append(plt.Line2D([0], [0], color=colors[idx], lw=1, label=f'Subtype {sub}'))

for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

fig.legend(handles=lines_for_legend, title='Subtype', loc='upper center',
           ncol=nsubtype, fontsize=9, title_fontproperties=font_bold, prop=font_bold, frameon=False)

plt.savefig(f'{output_dir}/survival_curves_cancer.png', dpi=500)
