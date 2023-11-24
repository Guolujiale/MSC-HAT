import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# 从CSV文件加载数据
data = pd.read_csv('clustered_embeddings_with_xy.csv')

# 获取x、y和cluster_label列的数据
x = data['x'].tolist()
y = data['y'].tolist()
cluster_label = data['cluster_label'].tolist()

# 创建一个字典来映射标签到颜色，假设有8种标签
label_to_color = {
    0: 'red',
    1: 'blue',
    2: 'green',
    3: 'orange',
    4: 'purple',
    5: 'cyan',
    6: 'magenta',
    7: 'yellow'
}

# 创建一个新的图像
plt.figure(figsize=(8, 6))

# 创建一个空的图例对象
legend_elements = []

# 遍历数据点，根据标签选择颜色并绘制点
for i in range(len(x)):
    color = label_to_color[cluster_label[i]]
    plt.scatter(x[i], y[i], color=color)
    # 添加标签到图例对象，确保每个标签只添加一次
    if cluster_label[i] not in [le[1] for le in legend_elements]:
        legend_elements.append((Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10), f'Cluster {cluster_label[i]}'))

# 添加图例
plt.legend(*zip(*legend_elements), title='Legend')

# 设置坐标轴标签
plt.xlabel('X坐标')
plt.ylabel('Y坐标')

plt.savefig('cluster_plot.png')
# 显示图像
plt.show()
