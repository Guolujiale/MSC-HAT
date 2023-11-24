import pandas as pd
import numpy as np
# 读取 CSV 文件
df = pd.read_csv('data.csv')

# 先按照 'x' 列排序，然后在相同 'x' 值的子组内按 'y' 列排序
df_sorted = df.sort_values(by=['x', 'y'])

df_sorted.to_csv('sorted_data.csv', index=False)

# 读取并排序 CSV 文件
df = pd.read_csv('sorted_data.csv')

# 确定矩形区域的边界
x_min, x_max = df['x'].min(), df['x'].max()
y_min, y_max = df['y'].min(), df['y'].max()

# 计算窗口大小和步长
window_width = (x_max - x_min) / 4
window_height = (y_max - y_min) / 4
step_x = window_width / 2
step_y = window_height / 2

# 初始化 'region' 列为字符串类型
df['region'] = pd.Series(dtype='object')

# 遍历每个窗口，分配区域标签，n 表示沿 x 轴的次数，m 表示沿 y 轴的次数
n = 0
for start_x in np.arange(x_min, x_max, step_x):
    n += 1
    m = 0
    for start_y in np.arange(y_min, y_max, step_y):
        m += 1
        end_x = start_x + window_width
        end_y = start_y + window_height

        # 如果接近边界，确保窗口不会超出边界
        if end_x > x_max:
            end_x = x_max
        if end_y > y_max:
            end_y = y_max

        # 对每个点进行区域分配，包括边界上的点
        mask = (df['x'] >= start_x) & (df['x'] <= end_x) & (df['y'] >= start_y) & (df['y'] <= end_y)
        df.loc[mask, 'region'] = f'region({n},{m})'

# 如果您决定不分配任何区域，只需取消注释下面这行代码
df['region'] = 0

df.to_csv('data_with_regions.csv', index=False)
