import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import warnings
import pickle
warnings.simplefilter(action='ignore', category=FutureWarning)

# Load datasets
train_data = pd.read_csv('KDDTrain+.txt', header=None)
test_data = pd.read_csv('KDDTest+.txt', header=None)

# Define classification label mapping dictionary
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

# Map classification labels
train_data[41] = train_data[41].apply(lambda x: labels[x])
test_data[41] = test_data[41].apply(lambda x: labels[x])

# Encode non-numeric features
train_data = pd.get_dummies(train_data, columns=[1, 2, 3])
test_data = pd.get_dummies(test_data, columns=[1, 2, 3])

# Align columns of training and test sets
train_features, test_features = train_data.align(test_data, join='outer', axis=1, fill_value=0)

# Separate features and labels
train_labels = train_features.pop(41)
test_labels = test_features.pop(41)

# Standardize features
scaler = StandardScaler()
train_features.columns = train_features.columns.astype(str)
test_features.columns = test_features.columns.astype(str)
train_features = scaler.fit_transform(train_features)
test_features = scaler.transform(test_features)

# 初始化基分类器模型
nb = GaussianNB()
dt = DecisionTreeClassifier()
knn = KNeighborsClassifier()

# 使用加权投票定义集成模型
ensemble_model = VotingClassifier(estimators=[('nb', nb), ('dt', dt), ('knn', knn)], voting='soft')

# 依据准确率设置各模型权重
param_grid = {'weights': [[0.31, 0.35, 0.34]]}

# Perform grid search to find the best weights
grid_search = GridSearchCV(ensemble_model, param_grid, cv=5)
grid_search.fit(train_features, train_labels)

# Print the best hyperparameters and the corresponding mean cross-validated score
print('Best Parameters:', grid_search.best_params_)
print('Best CV Score:', grid_search.best_score_)
# print("Best hyperparameters: ", grid_search.best_params_)
# print("Best mean cross-validated score: {:.2f}".format(grid_search.best_score_))

# Train ensemble model with the best hyperparameters
best_weights = grid_search.best_params_['weights']
ensemble_model = VotingClassifier(estimators=[('nb', nb), ('dt', dt), ('knn', knn)], voting='soft', weights=best_weights)
ensemble_model.fit(train_features, train_labels)

# Make predictions
test_pred = ensemble_model.predict(test_features)

# Compute classification metrics
accuracy = accuracy_score(test_labels, test_pred)
precision = precision_score(test_labels, test_pred)
recall = recall_score(test_labels, test_pred)
f1 = f1_score(test_labels, test_pred)

print('Accuracy:', accuracy)
print('Precision:', precision)
print('Recall:', recall)
print('F1 Score:', f1)

# # Plot bar chart
# labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
# values = [accuracy, precision, recall, f1]
# plt.bar(labels, values)

# # Add numbers above bars (as percentages)
# for i, v in enumerate(values):
#     plt.text(i-0.1, v+0.01, '{:.2%}'.format(v), fontsize=10)

# # Add title
# plt.title('Ensemble Learning Results (Weighted Voting)')
# plt.show()

# # Compute ROC curve and AUC
# fpr, tpr, threshold = roc_curve(test_labels, test_pred)
# roc_auc = auc(fpr, tpr)
# print('ROC AUC:', roc_auc)

# # Plot ROC curve
# plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
# plt.legend(loc='lower right')
# plt.plot([0, 1], [0, 1], 'r--')
# plt.xlim([0, 1])
# plt.ylim([0, 1])
# plt.ylabel('True Positive Rate')
# plt.xlabel('False Positive Rate')
# plt.title('Receiver Operating Characteristic')
# plt.show()
