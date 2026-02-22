import utils.utils as utils
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.stats.multitest import multipletests

exp_name = 'ukb_MixtureGRBF_cv_nsubtype_biom17'
nsubtype = 5

fig, ax = plt.subplots(figsize=(16/2.54, 4/2.54))

plt.rcParams.update({
    'font.family': 'sans-serif',  # 指定字体家族为无衬线
    'font.sans-serif': ['Arial'], # 在无衬线字体列表中首选 Arial
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 8
})


result_df = pd.read_csv(f'output/result_analysis/{exp_name}/{nsubtype}_subtypes/mh_difference.csv')


vmax = result_df['signed_log10FDR'].abs().max()

print(vmax)
threshold_star = np.log10(1 / 0.05)
threshold_dstar = np.log10(1 / 0.01)
threshold_tstar = np.log10(1 / 0.001)

pivot_df = result_df.pivot(index='feature', columns='subtype', values='signed_log10FDR')

im = ax.imshow(pivot_df.values.T, cmap=utils.custom_rdbu_r, aspect='auto', vmin=-vmax, vmax=vmax)

# Add significance annotations
for i in range(len(pivot_df.index)):
    for j in range(len(pivot_df.columns)):
        val = np.abs(pivot_df.iloc[i, j])
        if val > threshold_tstar:
            symbol = '***'
        elif val > threshold_dstar:
            symbol = '**'
        elif val > threshold_star:
            symbol = '*'
        else:
            continue
        ax.text(i, j+0.16, symbol, ha='center', va='center', 
                color='white' if val > np.max(pivot_df)/2 else 'black', 
                fontsize=12, weight='bold')

# Add colorbar using divider
# if idx == 1:
#     divider = make_axes_locatable(ax)
#     cax = divider.append_axes("right", size="5%", pad=0.1)
#     cbar = fig.colorbar(im, cax=cax)
#     cbar.set_label('Signed -log10(FDR)', rotation=270, labelpad=13)

# Set labels
ax.set_yticks(np.arange(len(pivot_df.columns)))
ax.set_xticks(np.arange(len(pivot_df.index)))
ax.set_yticklabels(pivot_df.columns, ha='right')
ax.set_xticklabels(pivot_df.index, rotation=45, ha='right')
ax.set_ylabel('Subtype')

# Add colorbar to the dedicated axis
# cbar = fig.colorbar(im, cax=cbar_ax, shrink=0.5)
# cbar.set_label('Signed -log10(FDR)', rotation=270, labelpad=13)

output_path = f'output/result_analysis/{exp_name}/{nsubtype}_subtypes/mh_difference.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f'Figure saved to {output_path}')
plt.close()
