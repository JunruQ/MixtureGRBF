import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# 读取数据
df = pd.read_csv('input/ukb/ukb_regress1_MDB_DRS_2_proteomics.csv')

# 读取生物标志物列表
with open('preprocess/significant_biomarker/MDB_DRS_2_proteomics.txt', 'r') as f:
    biomarker_list = list(map(lambda x: x.strip(), f.readlines()))

df = df[df['stage'] >= 50]

# Step 1: 提取需要聚类的两列数据
X = df[biomarker_list].values  # 提取为numpy数组

# Step 2: 使用KMeans进行聚类
kmeans = KMeans(n_clusters=2)  # 假设我们聚成2类
kmeans.fit(X)

# Step 3: 可视化聚类结果
fig = plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, cmap='rainbow')
plt.savefig('output/kmeans_cluster_result.png')