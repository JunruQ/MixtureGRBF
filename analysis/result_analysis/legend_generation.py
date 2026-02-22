import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib as mpl

# 设置字体为 Arial，大小为9pt
mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['font.size'] = 12

# 获取颜色
import utils.utils as utils
nsubtype = 5
colors = utils.subtype_colors[:nsubtype]

# 创建图例元素
# legend_elements = [Line2D([0], [0], color=colors[i], lw=2, label=f'Subtype {i+1}') for i in range(nsubtype)]

fig, ax = plt.subplots(figsize=(13/2.54, 0.3))  # 宽15cm，约等于15/2.54 inch；高可调
legend_elements = [Line2D([0], [0], color=colors[i], marker='o', linestyle='None', markersize=4, label=f'Subtype {i+1}') for i in range(nsubtype)]
# 绘图（仅用于图例）

ax.axis('off')  # 不显示坐标轴

# 添加图例，ncol=5 保证一行显示
legend = ax.legend(handles=legend_elements, loc='center', frameon=False, ncol=5, handletextpad=0.3, columnspacing=0.5)
# legend = ax.legend(handles=legend_elements, loc='center', frameon=False, ncol=5)
plt.tight_layout()
plt.savefig('tmp.png', dpi=300)
