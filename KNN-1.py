import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

# 加载数据集
train_data = pd.read_csv('KDDTrain+.txt', header=None)
test_data = pd.read_csv('KDDTest+.txt', header=None)

# 定义分类标签映射字典
labels = {
    'normal': 0,
    'back': 1,
    'land': 1,
    'pod': 1,
    'neptune': 1,
    'smurf': 1,
    'teardrop': 1,
    'apache2': 1,
    'udpstorm': 1,
    'processtable': 1,
    'mailbomb': 1,
    'buffer_overflow': 1,
    'loadmodule': 1,
    'perl': 1,
    'rootkit': 1,
    'sqlattack': 1,
    'xterm': 1,
    'ps': 1,
    'httptunnel': 1,
    'named': 1,
    'sendmail': 1,
    'snmpgetattack': 1,
    'snmpguess': 1,
    'worm': 1,
    'xlock': 1,
    'xsnoop': 1,
    'imap': 1,
    'ftp_write': 1,
    'guess_passwd': 1,
    'multihop': 1,
    'phf': 1,
    'spy': 1,
    'warezclient': 1,
    'warezmaster': 1,
    'portsweep': 1,
    'ipsweep': 1,
    'nmap': 1,
    'satan': 1,
    'mscan': 1,
    'saint': 1
}

# 对分类标签进行映射
train_data[41] = train_data[41].apply(lambda x: labels[x])
test_data[41] = test_data[41].apply(lambda x: labels[x])

# 将非数值特征进行编码
train_data = pd.get_dummies(train_data, columns=[1, 2, 3])
test_data = pd.get_dummies(test_data, columns=[1, 2, 3])

# 对训练集和测试集的列数进行对齐
train_features, test_features = train_data.align(test_data, join='outer', axis=1, fill_value=0)

# 分离特征和标签
train_labels = train_features.pop(41)
test_labels = test_features.pop(41)

# 对特征进行标准化
scaler = StandardScaler()
train_features.columns = train_features.columns.astype(str)
test_features.columns = test_features.columns.astype(str)
train_features = scaler.fit_transform(train_features)
test_features = scaler.transform(test_features)

param_grid = {
    'n_neighbors': [5],
    'weights': ['distance'],
    'algorithm': ['kd_tree'],
    'p': [1, 2],
    'leaf_size': [30],
    'metric': ['manhattan', 'euclidean']
}

# 初始化K近邻分类器
knn = KNeighborsClassifier()

# 定义网格搜索器
grid_search = GridSearchCV(estimator=knn, param_grid=param_grid, cv=5, scoring='accuracy')

# 训练模型
grid_search.fit(train_features, train_labels)

# 输出最优超参数和对应的交叉验证得分
print('Best Parameters:', grid_search.best_params_)
print('Best CV Score:', grid_search.best_score_)

# 进行预测
test_pred = grid_search.predict(test_features)

# 计算分类性能指标
accuracy = accuracy_score(test_labels, test_pred)
precision = precision_score(test_labels, test_pred)
recall = recall_score(test_labels, test_pred)
f1 = f1_score(test_labels, test_pred)

print('Accuracy:', accuracy)
# print('Precision:', precision)
# print('Recall:', recall)
# print('F1 Score:', f1)

# # 绘制柱状图
# labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
# values = [accuracy, precision, recall, f1]
# plt.bar(labels, values)
# plt.title('Performance Metrics')
# plt.xlabel('Metric')
# plt.ylabel('Value')

# # 在每个条形上方添加数字（以百分比形式显示）
# for i, v in enumerate(values):
#     plt.text(i-0.1, v+0.01, '{:.2%}'.format(v), fontsize=10)

# # 添加标题
# plt.title('Experimental Results of KNN')

# plt.show()

# # 绘制ROC曲线和AUC值
# fpr, tpr, _ = roc_curve(test_labels, test_pred)
# roc_auc = auc(fpr, tpr)

# plt.plot(fpr, tpr, label='ROC Curve (AUC = %0.2f)' % roc_auc)
# plt.plot([0, 1], [0, 1], '--', color='gray', label='Random Guess')
# plt.title('Receiver Operating Characteristic')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.legend(loc='lower right')
# plt.show()