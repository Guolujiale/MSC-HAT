import pandas as pd

# 读取文件，修改第一列，并保存
filename = 'B004_training_dryad.csv'
new_filename = 'single_cell_all_48feature.csv'
df = pd.read_csv(filename, encoding='ISO-8859-1')
# 修改第一列的值
df.iloc[:, 0] = range(len(df))
# 保存修改后的数据
#df.to_csv(filename, index=False, encoding='ISO-8859-1')


#df = pd.read_csv(filename, nrows=100000)
#df.to_csv(filename, index=False)

# 不重复地随机选取n个行
sampled_df = df.sample(n=len(df), replace=False)

# 保存随机选取后的结果到文件
sampled_df.to_csv(new_filename, index=False)

#我暂定用以下的数据格式来构建了哈:
#数据集包括st空转非图像数据以及RGB像素值表示的图像数据,且两种数据都在一个表格中.
# 该表格中st空转非图像数据来自对于细胞组织切片的空间转录,RGB像素值来自于对细胞组织切片
# 的RGB三通道的像素值提取.即该表格每一行代表一个细胞,前30列代表该细胞空转后的gene或
# protein的表达量,第31列与32列代表细胞的坐标,第33,34,35列代表该坐标的R,G,B的值
# ,第36列代表该细胞属于哪个组织切片,第37列代表该细胞属于哪一种生物组织.且表格中所
# 有元素都是浮点数.