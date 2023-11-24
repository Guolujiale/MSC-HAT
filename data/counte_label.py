import pandas as pd

# 加载数据
file_path = 'data.csv'  # 请将此路径替换为您的文件实际路径
df = pd.read_csv(file_path)

# 查看 'celltype' 列中有多少种不同的标签类型
unique_celltypes = df['celltype'].unique()

# 打印不同标签的数量
num_unique_celltypes = len(unique_celltypes)
print(f'There are {num_unique_celltypes} unique cell types.')

# 如果你想看具体都有哪些不同的标签类型，也可以打印出来
print(unique_celltypes)
