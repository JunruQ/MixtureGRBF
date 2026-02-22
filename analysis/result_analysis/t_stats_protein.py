import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import textalloc
import statsmodels.api as sm
import tqdm
from adjustText import adjust_text
from statsmodels.stats.multitest import multipletests

def bonferroni_correction(p_values):
    n = len(p_values)
    corrected_p_values = np.array(p_values) * n
    return corrected_p_values

# Style settings: MATLAB-like style with 12pt Arial for all text
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.labelsize'] = 11  # Axis labels: 12pt
plt.rcParams['axes.titlesize'] = 11  # Axis titles: 12pt
plt.rcParams['xtick.labelsize'] = 9  # X-axis tick labels: 12pt
plt.rcParams['ytick.labelsize'] = 9  # Y-axis tick labels: 12pt
plt.rcParams['axes.labelcolor'] = 'black'
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['axes.linewidth'] = 1.0

nsubtype = 5
exp_name = 'ukb_MixtureGRBF_cv_nsubtype_biom17'
INPUT_TABLE_PATH = 'input/ukb/ukb_covreg1_trans1_nanf1_biom0.csv'
OUTPUT_DIR = f'output/result_analysis/{exp_name}/{nsubtype}_subtypes'
CLASSIFICATION_PATH = 'analysis/result_analysis/protein_classification_form.csv'

# MATLAB-like color palette
palette = ['#206491', '#038db2', '#f9637c', '#fe7966', '#fbb45c', '#ffcb5d', '#81d0bb', '#45aab4']
gray = '#777777'


classification = pd.read_csv(CLASSIFICATION_PATH)

import utils.utils as utils
subtype_stage = utils.get_subtype_stage(exp_name, nsubtype)
protein_expr_difference_path = OUTPUT_DIR + '/t_stats_by_subtype.csv'


df = subtype_stage
subtype_cols = {f'subtype_{i}': (df['subtype'] == i).astype(int) for i in range(1, nsubtype + 1)}
subtype_df = pd.DataFrame(subtype_cols, index=df.index)
df = pd.concat([df, subtype_df], axis=1)
print(f'Loading {protein_expr_difference_path}')
results_df = pd.read_csv(protein_expr_difference_path)
biomarker_names = results_df.loc[results_df['subtype'] == 1, 'biom'].tolist()

def show_pos_neg_count(results_df):
    sig = results_df[results_df['corrected_p_value'] < 0.05]

    # 对每个 subtype 统计正向显著和负向显著的个数
    summary = sig.groupby('subtype').apply(
        lambda df: pd.Series({
            'pos_sig': (df['t_statistic'] > 0).sum(),
            'neg_sig': (df['t_statistic'] < 0).sum()
        })
    ).reset_index()

    print(summary)

alpha = 0.05

show_pos_neg_count(results_df)

# Create figure with 2x3 grid (for 5 subtypes, last subplot will be removed)
fig = plt.figure(figsize=(33/2.54, 16/2.54), dpi=300, facecolor='white')
fig.subplots_adjust(hspace=0.3, wspace=0.2, left=0.05, bottom=0.1, right=0.95, top=0.9)

xtick_labels_legend = classification.iloc[:, 0].tolist()
xtick_labels = ['Cardio-\nmetabolic', 'Inflam-\nmation', 'Neurology', 'Oncology']

strategy_priorities = [3,0,0,412,8]
for i in range(1, nsubtype + 1):
    k = i
    ax = plt.subplot(2, 3, i)  # 2x3 grid
    count = 0
    xtick_positions = []
    ts = results_df.loc[(results_df['subtype'] == k) & (results_df['biom'].isin(biomarker_names)), 't_statistic'].tolist()
    ps = results_df.loc[(results_df['subtype'] == k) & (results_df['biom'].isin(biomarker_names)), 'corrected_p_value'].tolist()
    case_group = df['subtype'] == k
    control_group = df['subtype'] != k

    # ps = bonferroni_correction(ps)
    is_significant = (np.array(ps) <= alpha).tolist()
    info_item = {biom: [ts[i], is_significant[i]] for i, biom in enumerate(biomarker_names)}

    x = []
    y = []
    bioms_new = []
    colors = []
    t_values_dict = {}
    class_t_values = {j: [] for j in range(classification.shape[0])}

    for j in range(classification.shape[0]):
        c = eval(classification.iloc[j, 1])
        c = sorted(list(set(c).intersection(biomarker_names)))
        for biom in c:
            count += 1
            x.append(count)
            bioms_new.append(biom)
            y.append(info_item[biom][0])
            t_val = info_item[biom][0]
            class_t_values[j].append((abs(t_val), t_val, biom, count-1, info_item[biom][1]))
            t_values_dict[count-1] = (t_val, biom)
            if info_item[biom][1]:
                colors.append(palette[j % len(palette)])
            else:
                colors.append(gray)
        xtick_positions.append(count)

    xtick_positions = [xtick_positions[2*i] for i in range(len(xtick_positions)//2)]
    tick_positions = np.array(xtick_positions)
    sig_count = sum(is_significant)

    # Scatter plot
    ax.scatter(x, y, marker='o', s=5, c=colors)
    ax.set_ylabel('t-statistic', fontsize=11)

    # Annotate top 2 biomarkers per category
    top_n = 1
    text_x = []
    text_y = []
    text_biom = []
    for j in range(classification.shape[0]):
        sorted_class = sorted(class_t_values[j], reverse=True, key=lambda x: x[1])[:top_n] + sorted(class_t_values[j], reverse=False, key=lambda x: x[1])[:top_n]
        for _, t_val, biom, idx, sig in sorted_class:
            if sig:
                text_x.append(x[idx])
                text_y.append(t_val)
                text_biom.append(biom)

    import random
    random.seed(1)

    if k in [1,3,4,5]:
        text_x = text_x[::-1]
        text_y = text_y[::-1]
        text_biom = text_biom[::-1]
    elif k == 2:
        def element_swap(lst, i, j):
            lst[i], lst[j] = lst[j], lst[i]

        for lst in [text_x, text_y, text_biom]:
            element_swap(lst, 9, 7)


    # Text allocation with 12pt font
    # textalloc.allocate(ax, text_x, text_y, text_biom,
    #                    x_scatter=x, y_scatter=y, x_lines=[np.array([min(x), max(x)])], y_lines=[np.array([0, 0])],
    #                    textsize=9, linecolor='black', priority_strategy=1, 
    #                    draw_all=True, xlims=(min(x), max(x)), margin=0.01, max_distance=0.1, auto_ha=True,
    #                    avoid_crossing_label_lines=True, avoid_label_lines_overlap=True)
    
    # ax.text(min(x) + 0.03 * (max(x) - min(x)), 135, f'Subtype {k} vs. rest, {sig_count} significant proteins', fontsize=12, color='black', ha='left', va='center')
    ax.set_title(f'Subtype {k} vs. rest, {sig_count} significant proteins')
    # ax.set_ylabel('t-statistic', fontsize=12, labelpad=5, color='black')
    # ax.set_title(f'Subtype {i} vs. rest\n{sig_count} significant proteins', fontsize=12, pad=10, color='black')
    ax.set_xlim(-5, count + 5)
    ax.set_ylim(-150, 150)
    if k in [3, 4, 5]:
        textalloc.allocate(ax, text_x, text_y, text_biom,
                        x_scatter=x, y_scatter=y, x_lines=[np.array([min(x), max(x)])], y_lines=[np.array([0, 0])],
                        textsize=9, linecolor='black', priority_strategy=strategy_priorities[k-1], 
                        draw_all=True, xlims=(min(x), max(x)), margin=0.01, max_distance=0.12, auto_ha=False,
                        avoid_crossing_label_lines=True, avoid_label_lines_overlap=True)
    elif k in [2]:
        TEXTS = []
        for i in range(len(text_x)):
            TEXTS.append(ax.text(text_x[i], text_y[i], text_biom[i], color='black', fontsize=9))
        adjust_text(
            TEXTS, 
            x=x,
            y=y,
            expand=(1.3, 1.3),
            min_arrow_len=0,
            force_explode=(2, 2),
            ensure_inside_axes=True,
            arrowprops=dict(
                arrowstyle="-", 
                color="black", 
                lw=1
            ),
            ax=ax,
        )
    elif k in [1, 4]:
        TEXTS = []
        for i in range(len(text_x)):
            TEXTS.append(ax.text(text_x[i], text_y[i], text_biom[i], color='black', fontsize=9))
        adjust_text(
            TEXTS, 
            x=x,
            y=y,
            min_arrow_len=0,
            force_text=(1, 1.5),
            force_static=(1, 1.5),
            force_explode=(1.5, 1.5),
            ensure_inside_axes=True,
            arrowprops=dict(
                arrowstyle="-", 
                color="black", 
                lw=1
            ),
            ax=ax,
        )

    ax.grid(False)
    ax.tick_params(direction='in')
    ax.set_xticks([])
    # ax.set_xticks(tick_positions)
    # ax.set_xticklabels(xtick_labels, rotation=0, ha='center', fontsize=10, color='black')
    ax.axhline(0, color='black', linewidth=1.0)

    if i == nsubtype:
        xtick_labels_legend = [i.replace('_II', ' II') if i.endswith('_II') else i for i in xtick_labels_legend]
        handles = [plt.Line2D([0], [0], marker='o', color=palette[j % len(palette)], markersize=6, linestyle='', label=label) for j, label in enumerate(xtick_labels_legend)]
        ax.legend(handles=handles, loc='upper right', bbox_to_anchor=(1.8, 1),
                  frameon=False, framealpha=1, edgecolor='black',
                  title='Biomarker Categories', title_fontsize=10, fontsize=10)

# Remove the empty subplot (position 6 in 2x3 grid)
# fig.delaxes(fig.axes[-1])  # Remove the last subplot (6th)

# Save the figure
plt.savefig(OUTPUT_DIR+'/t_stats_protein.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print(f'T-statistic plots saved to "{OUTPUT_DIR}/t_stats_protein.png"')