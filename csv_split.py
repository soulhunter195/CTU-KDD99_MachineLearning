import pandas as pd

df_features1 = pd.read_csv('features1.csv')

df_features2 = pd.read_csv('features2.csv')

df_features3 = pd.read_csv('features3.csv')

df_label1 = pd.read_csv('labels1.csv')

df_label2 = pd.read_csv('labels2.csv')

df_label3 = pd.read_csv('labels3.csv')

# 拼接两个DataFrame
df_concatenated_features = pd.concat([df_features1, df_features2, df_features3], ignore_index=True)

df_concatenated_label = pd.concat([df_label1, df_label2, df_label3], ignore_index=True)

# 将拼接后的DataFrame保存到新的CSV文件
df_concatenated_features.to_csv('features.csv', index=False)

df_concatenated_label.to_csv('labels.csv', index=False)