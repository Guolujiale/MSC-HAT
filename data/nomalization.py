import pandas as pd
from sklearn.preprocessing import RobustScaler

# 加载数据
file_path = 'data_with_regions.csv'  # 请将此路径替换为您的文件实际路径
df = pd.read_csv(file_path)

# 选择前2000列基因标签对应的数据进行归一化
genes_data = df.iloc[:, 0:2000]

# 初始化 RobustScaler
scaler = RobustScaler()

# 对基因表达数据进行归一化处理
genes_scaled = scaler.fit_transform(genes_data)

# 将归一化后的数据转换回DataFrame，并将列名设置回去
genes_scaled_df = pd.DataFrame(genes_scaled, columns=genes_data.columns)

# 将归一化后的基因表达数据与原始DataFrame的其他列合并
df.iloc[:, 0:2000] = genes_scaled_df

# 现在 df 包含了归一化后的基因表达数据和其他未变化的列
# 如果需要，可以将其保存为新的 CSV 文件
df.to_csv('final_data.csv', index=False)
