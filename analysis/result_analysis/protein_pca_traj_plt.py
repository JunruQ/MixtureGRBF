# ==============================================================================
# 0. SETUP: LIBRARIES AND CONFIGURATION
# ==============================================================================
import pandas as pd
import numpy as np
import warnings
import sys
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import os

# --- [!] 用户配置: 文件路径 ---
nsubtype = 5
result_folder = 'ukb_MixtureGRBF_cv_nsubtype_biom17'
output_dir = f'./output/result_analysis/{result_folder}/{nsubtype}_subtypes'

# --- 输入文件路径 ---
SUBTYPE_FILE_PATH = f'./output/{result_folder}/{nsubtype}_subtypes/subtype_stage.csv'
PROTEIN_FILE_PATH = 'input/ukb/ukb_covreg1_trans1_nanf1_biom17.csv'
SUBTYPE_ORDER_FILE_PATH = f'output/result_analysis/{result_folder}/{nsubtype}_subtypes/all_cause_mortality_order.csv'
# --- 新增: 轨迹文件路径模板 ---
TRAJECTORY_FILE_TEMPLATE = f'./output/{result_folder}/{nsubtype}_subtypes/trajectory{{}}.csv'


# --- 输出文件路径 ---
OUTPUT_IMAGE_PATH = output_dir + '/protein_pca_3d_trajectory_with_paths.png'

# --- 定义非蛋白质数据列 (元数据列) ---
NON_PROTEIN_COLUMNS = ['eid', 'stage', 'sex', 'education', 'centre', 'Ethnic', 'years']


# ==============================================================================
# 1. DATA LOADING AND PREPARATION (与之前版本相同)
# ==============================================================================
print("--- Step 1: Loading and Preparing Data ---")
# ... (此部分代码与您提供的版本完全相同，为简洁起见此处省略) ...
# --- 加载亚型数据 ---
try:
    subtype_df = pd.read_csv(SUBTYPE_FILE_PATH)
    print(f"✅ Successfully loaded subtype data from: {SUBTYPE_FILE_PATH}")
except FileNotFoundError:
    print(f"❌ ERROR: Subtype file not found at '{SUBTYPE_FILE_PATH}'.")
    sys.exit()

# --- 加载蛋白质数据 ---
try:
    protein_df = pd.read_csv(PROTEIN_FILE_PATH)
    print(f"✅ Successfully loaded protein data from: {PROTEIN_FILE_PATH}")
except FileNotFoundError:
    print(f"❌ ERROR: Protein data file not found at '{PROTEIN_FILE_PATH}'.")
    sys.exit()

# --- 统一ID列名以便合并 ---
if 'PTID' in subtype_df.columns and 'eid' not in subtype_df.columns:
    subtype_df.rename(columns={'PTID': 'eid'}, inplace=True)
    print("Info: Renamed 'PTID' column to 'eid' in subtype data for merging.")
if 'RID' in protein_df.columns and 'eid' not in protein_df.columns:
    protein_df.rename(columns={'RID': 'eid'}, inplace=True)
    print("Info: Renamed 'RID' column to 'eid' in subtype data for merging.")

# --- 应用亚型排序 ---
n_subtypes = nsubtype
try:
    subtype_order_df = pd.read_csv(SUBTYPE_ORDER_FILE_PATH, header=None)
    subtype_order = subtype_order_df.iloc[:, 0].tolist()
    print(f"Info: Applying custom subtype order: {subtype_order}")
    subtype_mapping = {original: new_order for new_order, original in enumerate(subtype_order, 1)}
    subtype_df['subtype_ordered'] = subtype_df['subtype'].map(subtype_mapping)
except FileNotFoundError:
    warnings.warn(f"⚠️ Warning: Subtype ordering file not found at '{SUBTYPE_ORDER_FILE_PATH}'. Using default order.")
    subtype_df['subtype_ordered'] = subtype_df['subtype']

subtype_df = subtype_df.dropna(subset=['subtype_ordered'])
subtype_df['subtype_ordered'] = subtype_df['subtype_ordered'].astype(int)

# --- 合并数据 ---
merged_df = pd.merge(subtype_df[['eid', 'subtype_ordered']], protein_df, on='eid', how='inner')

if merged_df.empty:
    print("❌ ERROR: Merged DataFrame is empty.")
    sys.exit()
else:
    print(f"✅ Data merged successfully. Found {len(merged_df)} samples.")


# ==============================================================================
# 2. PCA DIMENSIONALITY REDUCTION (与之前版本相同)
# ==============================================================================
print("\n--- Step 2: Performing PCA on Patient Data ---")
# ... (此部分代码与您提供的版本完全相同，为简洁起见此处省略) ...
existing_meta_cols = [col for col in NON_PROTEIN_COLUMNS if col in merged_df.columns]
protein_columns = merged_df.columns.drop(existing_meta_cols + ['subtype_ordered'])

if len(protein_columns) < 3:
    print(f"❌ ERROR: Found only {len(protein_columns)} protein columns. Need at least 3.")
    sys.exit()
else:
    print(f"Info: Identified {len(protein_columns)} protein features for PCA.")

protein_data = merged_df[protein_columns]
rows_before = len(protein_data)
cleaned_indices = protein_data.dropna().index
protein_data_cleaned = protein_data.loc[cleaned_indices]
rows_after = len(protein_data_cleaned)
print(f"Info: Handling missing values. {rows_before - rows_after} rows removed. {rows_after} rows remain.")

if rows_after == 0:
    print("❌ ERROR: No data remains after handling missing values.")
    sys.exit()

analysis_df = merged_df.loc[cleaned_indices].copy()
scaler = StandardScaler()
protein_scaled = scaler.fit_transform(protein_data_cleaned)
pca = PCA(n_components=3)
principal_components = pca.fit_transform(protein_scaled)
analysis_df['PC1'] = principal_components[:, 0]
analysis_df['PC2'] = principal_components[:, 1]
analysis_df['PC3'] = principal_components[:, 2]

print("\n✅ PCA analysis complete.")
print(f"  - Total variance explained by 3 components: {np.sum(pca.explained_variance_ratio_):.2%}")


# ==============================================================================
# 3. 新增: 加载并转换轨迹数据
# ==============================================================================
print("\n--- Step 3: Loading and Transforming Trajectory Data ---")
transformed_trajectories = []

# 从真实数据中获取stage范围
if 'stage' not in analysis_df.columns:
    print("❌ ERROR: 'stage' column not found in data, cannot create stage axis for trajectories.")
    sys.exit()
    
min_stage = int(analysis_df['stage'].min())
max_stage = int(analysis_df['stage'].max())
stage_range = np.arange(min_stage, max_stage + 1)
print(f"Info: Trajectories will be mapped to stage range: {min_stage} to {max_stage}.")

for i in subtype_order:  # 使用 subtype_order 替代 range(1, n_subtypes + 1)
    traj_path = TRAJECTORY_FILE_TEMPLATE.format(i)
    try:
        # 读取无表头的轨迹数据
        traj_df = pd.read_csv(traj_path)
        
        # 确保轨迹数据的列数与蛋白质数据匹配
        if traj_df.shape[1] != len(protein_columns):
            warnings.warn(f"⚠️ Warning: Trajectory {i} has {traj_df.shape[1]} columns, but expected {len(protein_columns)}. Skipping.")
            continue
            
        # 确保轨迹点数量与stage范围匹配
        if traj_df.shape[0] != len(stage_range):
            warnings.warn(f"⚠️ Warning: Trajectory {i} has {traj_df.shape[0]} points, but stage range has {len(stage_range)}. Skipping.")
            continue

        # 将轨迹数据的列名设置为与原始数据一致，以保证转换的准确性
        traj_df.columns = protein_columns
        
        # 使用已经拟合好的scaler和pca来转换轨迹数据
        scaled_traj = scaler.transform(traj_df)
        transformed_traj = pca.transform(scaled_traj)
        
        transformed_trajectories.append({
            "subtype": subtype_mapping[i],  # 使用 subtype_mapping 将原始 subtype 映射到排序后的 subtype
            "path": transformed_traj
        })
        print(f"✅ Successfully loaded and transformed trajectory for Subtype {i} (Ordered: {subtype_mapping[i]}).")
        
    except FileNotFoundError:
        warnings.warn(f"⚠️ Warning: Trajectory file not found at '{traj_path}'. Skipping.")
    except Exception as e:
        warnings.warn(f"⚠️ Warning: Failed to process trajectory {i} due to an error: {e}")

# ==============================================================================
# 4. 3D VISUALIZATION (解决轨迹遮挡问题)
# ==============================================================================
# ==============================================================================
# 4. 3D VISUALIZATION (批量生成多视角版本)
# ==============================================================================
print("\n--- Step 4: Generating Multiple Advanced 3D Visualizations ---")

from matplotlib.colors import LinearSegmentedColormap

# --- [!] 用户配置: 在此定义您想生成的多个视角和缩放级别 ---
# name: 用于生成图片的文件名
# elev: 仰角 (从侧面看是0, 从正上方看是90)
# azim: 方位角/旋转角度 (0-360)
# zoom: 缩放级别 (1.0代表正常大小, 0.5代表放大一倍, 2.0代表缩小)
views_to_generate = [
    # {'name': 'default_view', 'elev': 25, 'azim': -60, 'zoom': 1.0},
    {'name': 'zoomed_in_view', 'elev': 20, 'azim': -150, 'zoom': 0.5},
]

# --- 获取数据的整体范围 ---
x_min, x_max = analysis_df['PC1'].min(), analysis_df['PC1'].max()
y_min, y_max = analysis_df['PC2'].min(), analysis_df['PC2'].max()
z_min, z_max = analysis_df['PC3'].min(), analysis_df['PC3'].max()
x_center, y_center, z_center = np.mean([x_min, x_max]), np.mean([y_min, y_max]), np.mean([z_min, z_max])
x_range, y_range, z_range = (x_max - x_min), (y_max - y_min), (z_max - z_min)

# --- 定义基础颜色和轨迹颜色图 ---
# 1. 定义自定义的 "蓝红黄紫绿" 散点颜色
import utils.utils as utils
custom_scatter_colors = utils.subtype_colors[:nsubtype]

# 2. <-- 修改核心: 根据上面的颜色列表，为每个轨迹动态创建自定义颜色图 ---
custom_trajectory_cmaps = []
for i, color_hex in enumerate(custom_scatter_colors):
    cmap_name = f'custom_traj_cmap_{i}'
    # 创建一个从浅灰色 (#E0E0E0) 到指定颜色的平滑渐变
    colors_for_cmap = ['#E0E0E0', color_hex]
    custom_cmap = LinearSegmentedColormap.from_list(cmap_name, colors_for_cmap)
    custom_trajectory_cmaps.append(custom_cmap)


# --- 循环生成每个视图 ---
for view in views_to_generate:
    print(f"\n--- Generating view: '{view['name']}' ---")
    
    fig = plt.figure(figsize=(15, 12))
    ax = fig.add_subplot(111, projection='3d')

    # --- 绘制散点 ---
    ordered_subtypes = sorted(analysis_df['subtype_ordered'].unique())
    scatter_colors_dict = {subtype: custom_scatter_colors[i] for i, subtype in enumerate(ordered_subtypes)}

    for subtype_val, data in analysis_df.groupby('subtype_ordered'):
        if len(data) > 1:
            plot_data = data.sample(frac=0.3)
        else:
            plot_data = data
        ax.scatter(plot_data['PC1'], plot_data['PC2'], plot_data['PC3'], label=f'Subtype {subtype_val} samples',
                   c=[scatter_colors_dict[subtype_val]], s=5, alpha=0.7, edgecolors='none')

    # --- 绘制轨迹 ---
    main_lw = 5
    norm = Normalize(vmin=min_stage, vmax=max_stage)

    for traj_data in transformed_trajectories:
        subtype_idx, path = traj_data["subtype"] - 1, traj_data["path"]
        # <-- 修改核心: 使用我们新创建的自定义颜色图对象列表 ---
        cmap = custom_trajectory_cmaps[subtype_idx]
        
        for j in range(len(path) - 1):
            p1, p2 = path[j], path[j+1]
            segment_color = cmap(norm(stage_range[j]))
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color=segment_color, linewidth=main_lw)

    # --- 设置图表属性和图例 ---
    ax.set_title(f"3D PCA - View: {view['name']}", fontsize=20, pad=20)
    ax.set_xlabel('PC1', fontsize=12, labelpad=10)
    ax.set_ylabel('PC2', fontsize=12, labelpad=10)
    ax.set_zlabel('PC3', fontsize=12, labelpad=10)

    from matplotlib.lines import Line2D
    handles, labels = ax.get_legend_handles_labels()
    for i in range(len(transformed_trajectories)):
        subtype_val = transformed_trajectories[i]['subtype']
        # <-- 修改核心: 从自定义颜色图中获取代表色 ---
        rep_color = custom_trajectory_cmaps[i](0.9) # 用0.9取一个饱和度高的颜色
        proxy = Line2D([0], [0], linestyle="-", lw=main_lw, color=rep_color, label=f'Subtype {subtype_val} Trajectory')
        handles.append(proxy)
    ax.legend(handles=handles, title='Data Legend', loc='best', markerscale=2)

    cbar_cmap = plt.get_cmap('Grays')
    sm = plt.cm.ScalarMappable(cmap=cbar_cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.6, aspect=20, pad=0.08)
    cbar.set_label('Stage Progression', fontsize=12, rotation=270, labelpad=20)
    ax.grid(True)

    # --- 应用视角和缩放 ---
    ax.view_init(elev=view['elev'], azim=view['azim'])
    zoom_factor = view['zoom']
    ax.set_xlim3d(x_center - (x_range / 2 * zoom_factor), x_center + (x_range / 2 * zoom_factor))
    ax.set_ylim3d(y_center - (y_range / 2 * zoom_factor), y_center + (y_range / 2 * zoom_factor))
    ax.set_zlim3d(z_center - (z_range / 2 * zoom_factor), z_center + (z_range / 2 * zoom_factor))
    
    # --- 保存图像 ---
    import os
    base_path, extension = os.path.splitext(OUTPUT_IMAGE_PATH)
    view_save_path = f"{base_path}_{view['name']}{extension}"
    
    plt.tight_layout()
    plt.savefig(view_save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Visualization for view '{view['name']}' saved to: {view_save_path}")
    
    plt.close(fig)