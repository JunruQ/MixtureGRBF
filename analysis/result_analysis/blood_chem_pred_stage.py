import pandas as pd
import utils.utils as utils
import numpy as np
import seaborn as sns
exp_name = 'ukb_MixtureGRBF_cv_nsubtype_biom17'
nsubtype = 5

blood_chem_path = 'data/ClinicalLabData.csv'

blood_chem_df = pd.read_csv(blood_chem_path)

subtype_stage = utils.get_subtype_stage(exp_name, nsubtype)

subtype_stage = subtype_stage.rename(columns={'PTID': 'eid'})

subtype_stage = subtype_stage.merge(blood_chem_df, on='eid', how='left')

X = subtype_stage.drop(columns=['eid', 'subtype', 'stage'])
# fill with mean for simplicity
X = X.fillna(X.mean())
y = subtype_stage['stage']

# linear regression model
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

# residual boxplot over subtypes
import seaborn as sns
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.labelsize': 10,
    'axes.titlesize': 12,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'axes.spines.top': False,
    'axes.spines.right': False
})

# sns.set(style="whitegrid")
plt.figure(figsize=(5, 5))
subtype_stage['residuals'] = model.predict(X) - y
sns.boxplot(
    x='subtype', 
    y='residuals', 
    data=subtype_stage,
    width=0.35,
    notch=True,
    palette=utils.subtype_colors[:nsubtype],  # Use custom colors for subtypes
    boxprops={'edgecolor': 'black', 'linewidth': 1},
    medianprops={'color': 'black', 'linewidth': 1.5},
    whiskerprops={'color': 'black', 'linewidth': 1},
    capprops={'color': 'black', 'linewidth': 1},
    flierprops={'marker': '+', 'markerfacecolor': 'black', 'markersize': 4, 'alpha': 0.5}
)

plt.title('Age Delta by Subtype\n(Pred from Linear Regression)', fontsize=12, pad=15)
plt.xlabel('Subtype', fontsize=10, labelpad=8)
plt.ylabel('Age Delta (Pred - True)', fontsize=10, labelpad=8)
plt.grid(True, linestyle='--', alpha=0.2)
plt.tight_layout()
plt.savefig('residuals_boxplot_by_subtype.png', dpi=300)
plt.close()

# Set modern style parameters
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.labelsize': 10,
    'axes.titlesize': 12,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.linestyle': '--',
    'grid.alpha': 0.2
})

# Create a figure with subplots for each subtype
fig, axes = plt.subplots(1, nsubtype, figsize=(5 * nsubtype, 5), sharex=True, sharey=True)

# Ensure axes is a list even for a single subplot
if nsubtype == 1:
    axes = [axes]

# Define jitter amount (adjust as needed for visual effect)
jitter_strength = 0.5  # Small random noise to spread points

# Plot scatter and trend line for each subtype
for i, subtype in enumerate(range(1, nsubtype+1)):
    # Filter data for the current subtype
    mask = subtype_stage['subtype'] == subtype
    x_sub = y[mask]
    y_sub = y_pred[mask]

    # Add jitter to x and y coordinates
    np.random.seed(42)  # For reproducibility
    x_jitter = x_sub + np.random.uniform(-jitter_strength, jitter_strength, size=x_sub.shape)
    y_jitter = y_sub + np.random.uniform(-jitter_strength, jitter_strength, size=y_sub.shape)

    # Scatter plot with jitter
    sns.scatterplot(
        x=x_jitter,
        y=y_jitter,
        ax=axes[i],
        color=utils.subtype_colors[subtype-1],
        s=5,
        alpha=0.6,
        edgecolor='black',
        linewidth=0.5
    )

    # Add trend line (using original data, not jittered)
    z = np.polyfit(x_sub, y_sub, 1)
    p = np.poly1d(z)
    x_trend = np.linspace(x_sub.min(), x_sub.max(), 100)
    axes[i].plot(x_trend, p(x_trend), color='black', linestyle='--', linewidth=1.5)

    # Customize subplot
    axes[i].set_title(f'Subtype {subtype}', fontsize=12, pad=15)
    axes[i].set_xlabel('True Age', fontsize=10, labelpad=8)
    axes[i].set_ylabel('Predicted Age', fontsize=10, labelpad=8)

    # Add diagonal line for reference
    axes[i].plot([x_sub.min(), x_sub.max()], [x_sub.min(), x_sub.max()], 
                 color='gray', linestyle=':', linewidth=1, alpha=0.5)

# Adjust layout to prevent overlap
plt.tight_layout()

# Save the plot
plt.savefig('tmp.png', dpi=300)
plt.close()