import pandas as pd

def subtype_order_map(df, exp_name, nsubtype=5):
    """
    根据实验名称和亚型顺序文件，创建亚型映射。
    
    参数:
    df (DataFrame): 包含亚型信息的DataFrame。
    exp_name (str): 实验名称，用于确定亚型顺序文件路径。
    nsubtype (int, optional): 亚型数量，默认为5。
    
    返回:
    DataFrame: 更新后的DataFrame，包含新的亚型顺序列。
    """
    subtype_order_file = f'output/survival_analysis/{exp_name}/{nsubtype}_subtypes/all_cause_mortality_order.csv'
    
    try:
        subtype_order_df = pd.read_csv(subtype_order_file, header=None)
        subtype_order = subtype_order_df.iloc[:, 0].tolist()
        print(f"Info: Applying custom subtype order: {subtype_order}")
        subtype_mapping = {original: new_order for new_order, original in enumerate(subtype_order, 1)}
        df['subtype'] = df['subtype'].map(subtype_mapping)
    except FileNotFoundError:
        print(f"⚠️ Warning: Subtype ordering file not found at '{subtype_order_file}'. Using default order.")
        
    return df

def get_subtype_stage(exp_name, nsubtype=5, subtype_order=True):
    """
    获取包含亚型和阶段信息的DataFrame，并应用亚型顺序映射。
    
    参数:
    exp_name (str): 实验名称，用于确定亚型文件路径。
    nsubtype (int, optional): 亚型数量，默认为5。
    subtype_order (bool, optional): 是否应用亚型顺序映射，默认为True。
    
    返回:
    DataFrame: 包含亚型和阶段信息的DataFrame，其ID列为PTID。
    """
    subtype_file_path = f'./output/{exp_name}/{nsubtype}_subtypes/subtype_stage.csv'
    
    try:
        subtype_df = pd.read_csv(subtype_file_path)
        print(f"✅ Successfully loaded subtype data from: {subtype_file_path}")
    except FileNotFoundError:
        print(f"❌ ERROR: Subtype file not found at '{subtype_file_path}'.")
        raise FileNotFoundError(f"Subtype file not found at '{subtype_file_path}'")
    if subtype_order:
        subtype_df = subtype_order_map(subtype_df, exp_name, nsubtype=nsubtype)
    return subtype_df

def get_subtype_stage_with_cov(exp_name, nsubtype=5, subtype_order=True):
    """
    获取包含亚型和阶段信息的DataFrame，并应用亚型顺序映射。
    
    参数:
    exp_name (str): 实验名称，用于确定亚型文件路径。
    nsubtype (int, optional): 亚型数量，默认为5。
    subtype_order (bool, optional): 是否应用亚型顺序映射，默认为True。
    
    返回:
    DataFrame: 包含亚型和阶段信息的DataFrame，其ID列为PTID。
    """
    subtype_df = get_subtype_stage(exp_name, nsubtype=nsubtype, subtype_order=subtype_order)
    cov_path = 'input/ukb/ukb_covreg1_trans1_nanf1_biom9.csv'
    cov_columns = ['RID', 'sex', 'education', 'centre', 'Ethnic']
    cov_df = pd.read_csv(cov_path, usecols=cov_columns).rename(columns={'RID': 'PTID'})
    subtype_df = subtype_df.merge(cov_df, on='PTID', how='left')
    return subtype_df

# subtype_colors = [
#     '#0072BD',  # 1-蓝色
#     '#D95319',  # 2-红色
#     '#EDB120',  # 3-黄色
#     '#7E2F8E',  # 4-紫色
#     '#77AC30'   # 5-绿色
# ]
subtype_colors = [
    '#0072BD',
    '#EDB120',
    '#77AC30',
    '#D95319',  
    '#A2142F'
]

# subtype_colors = [
#     '#132157',  # 1-蓝色
#     '#FFAD0A',  # 2-红色
#     '#1BB6AF',  # 3-黄色
#     '#EE6100',  # 4-紫色
#     '#D72000'   # 5-绿色
# ]

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# 自定义调色板
subtype_cmap = ListedColormap(subtype_colors)

from matplotlib.colors import LinearSegmentedColormap
colors = ['#2980b9', '#ffffff', '#c0392b']
n_bins = 256
custom_rdbu_r = LinearSegmentedColormap.from_list('custom_diverging', colors, N=n_bins)

categorical_palette = ['#6E8FB2', '#7DA494', '#EAB67A', '#E5A79A', '#C16E71', '#ABC8E5', '#D8A0C1', '#9F8DB8', '#D0D08A']
