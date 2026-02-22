import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import utils.utils as utils

# Experiment parameters
exp_name = 'ukb_MixtureGRBF_cv_nsubtype_biom17'
nsubtype = 5
blood_chem_path = 'data/ClinicalLabData.csv'

# Load and merge data
blood_chem_df = pd.read_csv(blood_chem_path)
subtype_stage = utils.get_subtype_stage(exp_name, nsubtype)
subtype_stage = subtype_stage.rename(columns={'PTID': 'eid'})
subtype_stage = subtype_stage.merge(blood_chem_df, on='eid', how='left')

# Prepare features and target
X = subtype_stage.drop(columns=['eid', 'subtype', 'stage'])
X = X.fillna(X.mean())
y = subtype_stage['stage']

# Linear regression model
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X, y)

# Calculate residuals
subtype_stage['residuals'] = model.predict(X) - y

# Set plot style
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 10,
})

# Convert cm to inches for figure size (7cm x 7cm)
cm_to_inch = 1 / 2.54
fig, ax = plt.subplots(figsize=(7 * cm_to_inch, 7 * cm_to_inch))

# Define colors
colors = utils.subtype_colors[:nsubtype]

sns.boxplot(
    x='subtype',
    y='residuals',
    data=subtype_stage,
    width=0.3,
    notch=True,
    palette=colors,
    boxprops={'edgecolor': 'black', 'linewidth': 1},
    medianprops={'color': 'black', 'linewidth': 1.5},
    whiskerprops={'color': 'black', 'linewidth': 1},
    capprops={'color': 'black', 'linewidth': 1},
    flierprops={'marker': '+', 'markerfacecolor': 'black', 'markersize': 4, 'alpha': 0.5},
    ax=ax
)

ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5)

# Set title and labels
ax.set_title('Blood routin age gap', fontsize=10, pad=10)
ax.set_xlabel('Subtype')
ax.set_ylabel('Age gap (years)')

# Set tick parameters
ax.tick_params(axis='both', labelsize=10, width=0.5)

# Set spine linewidth
for spine in ax.spines.values():
    spine.set_linewidth(0.5)

# Adjust layout
plt.subplots_adjust(left=0.2, right=0.95, top=0.85, bottom=0.2)

# Save figure
output_path = f'output/result_analysis/{exp_name}/{nsubtype}_subtypes/blood_chem_predicted_age_delta_boxplot.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to '{output_path}'")