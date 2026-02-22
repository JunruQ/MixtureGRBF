import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 输入目录
input_dir = "output/ukb_MixtureGRBF_search_lambda_biom17"

colors = plt.cm.tab10(np.linspace(0, 1, 8))
linestyles = ['-', '--', '-.', ':', '-', '--', '-.', ':']

plt.figure(figsize=(10/2.54, 6.76/2.54))

plt.rcParams.update({
    'font.family': 'sans-serif',  # 指定字体家族为无衬线
    'font.sans-serif': ['Arial'], # 在无衬线字体列表中首选 Arial
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 8
})

plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['axes.linewidth'] = 0.8

yticks = []
ytlabels = []
all_lambdas = []

# --- 1. 设置想要提取的指数 ---
exponent = 6             # 这里设置为 6，即 10^6
scale_factor = 10 ** exponent

for k in range(1, 9):
    file_path = os.path.join(input_dir, f"lambda_loglik_{k}subtypes.csv")
    
    # 为了演示，如果文件不存在生成假数据 (实际使用请保留原本的 check)
    if not os.path.exists(file_path):
        # print(f"⚠️ 文件不存在: {file_path}")
        # continue
        # --- 模拟数据 (仅用于演示，请删除此块并恢复上面的 continue) ---
        mock_lambda = [2**i for i in range(10)]
        mock_loglik = [-1.5e6 + k * 10000 - (x - 100)**2 for x in mock_lambda]
        df = pd.DataFrame({'lambda': mock_lambda, 'loglik': mock_loglik})
    else:
        df = pd.read_csv(file_path)

    all_lambdas.extend(df['lambda'].tolist())

    lo = df['loglik'].min()
    hi = df['loglik'].max()
    
    if hi == lo:
        normalized = np.zeros_like(df['loglik'])
    else:
        normalized = (df['loglik'] - lo) / (hi - lo)

    base = 2 * (k - 1)
    mapped = normalized + base

    plt.plot(
        df['lambda'],
        mapped,
        label=f'{k} subtypes',
        color=colors[k-1],
        linestyle=linestyles[k-1],
        linewidth=1.5
    )

    yticks.extend([base, base + 1])
    
    # --- 2. 修改标签格式 ---
    # 将数值除以 scale_factor，并保留2位小数(根据需要调整 .2f)
    ytlabels.extend([f"{lo/scale_factor:.4f}", f"{hi/scale_factor:.4f}"])

# x 轴设置
plt.xscale("log", base=2)
if all_lambdas:
    all_lambdas = sorted(set(all_lambdas))
    plt.xticks(all_lambdas, [str(int(l)) if float(l).is_integer() else str(l) for l in all_lambdas])

# y 轴设置
plt.ylim(0 - 0.5, 2 * 8 - 1 + 0.5)
plt.yticks(yticks, ytlabels)

plt.text(0, 1.02, f"$\\times 10^{{{exponent}}}$", transform=plt.gca().transAxes, ha='left', va='bottom', fontsize=10)
plt.tight_layout()

plt.savefig("trajectory_smoothness.png", dpi=300)
plt.show()