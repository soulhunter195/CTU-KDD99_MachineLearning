import numpy as np
import matplotlib.pyplot as plt

# 模型名称
models = ['NB', 'C4.5-DT', 'KNN', 'NDK-EL']

# 模型指标和对应的值
indicators = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC']
nb_values = [73.75, 97.54, 55.28, 70.57, 77]
c45_values = [83.51, 96.82, 73.45, 83.53, 85]
knn_values = [80.34, 96.13, 68.21, 79.80, 82]
ndk_values = [86.77, 97.06, 79.16, 87.20, 88]

# 将模型指标值组合为一个数组
values = np.array([nb_values, c45_values, knn_values, ndk_values])

# 设置每个条形的宽度
width = 0.2

# 计算每个条形的位置
pos = [-2.5*width, -1.5*width, -0.5*width, 0.5*width]

# 绘制每个模型的条形图
fig, ax = plt.subplots()
for i, model in enumerate(models):
    bars = ax.bar(np.arange(len(indicators))+pos[i], values[i,:], width, label=model)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height+0.5, '%.2f' % height, ha='center', va='bottom', fontsize=14)

# 添加图例和设置图形标题
ax.legend(fontsize=18)
ax.set_title('Comparison of Evaluation Metrics for Base Classifiers and NDK-EL Classifier', fontsize=18)

# 设置X轴标签和刻度
ax.set_xticks(np.arange(len(indicators)))
ax.set_xticklabels(indicators, fontsize=20)

# 设置Y轴刻度和范围
ax.set_ylim([50, 110])
ax.set_yticks([50, 75, 85, 90, 95, 100])
ax.tick_params(axis='y', labelsize=14)

# 设置画布大小和分辨率
plt.figure(dpi=200, figsize=(6.2, 3))

# 显示图形
plt.tight_layout()
plt.show()