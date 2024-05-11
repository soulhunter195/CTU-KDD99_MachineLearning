import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.model_selection import train_test_split

# 读取数据， 第一行是列名
labels = pd.read_csv('./labels.csv', header=0)
featuress = pd.read_csv('./features.csv', header=0)

# 将数据转换为numpy数组
# 如果label为yes，转换为1，否则转换为0
for i in range(len(labels)):
    if labels['yes/no'][i] == 'yes':
        labels['yes/no'][i] = 1
    else:
        labels['yes/no'][i] = 0
labels = np.array(labels).astype(int)
featuress = np.array(featuress).astype(float)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(featuress, labels, test_size=0.3, random_state=0)

# 将X_train 增加一个维度，值为1
X_train = np.expand_dims(X_train, axis=1)
X_test = np.expand_dims(X_test, axis=1)
# print(X_trian.shape)

 # 导入所需库
import torch
import torch.nn as nn 
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F 

# 定义超参数
input_dim = 1 # 输入数据维度 
num_classes = 2 # 分类数
batch_size = 64 # 批处理数
learning_rate = 1e-3
epochs = 100

# 定义CNN模型
class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Conv1d(input_dim, 32, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
            nn.MaxPool1d(kernel_size=2, stride=2))
            
        self.fc2 = nn.Sequential(
            nn.Linear(128, 1024),
            nn.Sigmoid(),
            nn.Linear(1024, num_classes))
        
    def forward(self, x):
        out = self.layer1(x)
        out = out.reshape(out.size(0), -1)
        out = self.fc2(out)
        # 使用dropout技术
        out = nn.Dropout(p=0.5)(out)
        return out

# 实例化模型,判断GPU是否可用
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = CNN().to(device) 
train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long())
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = torch.utils.data.TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).long())
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# 定义损失函数和优化器 损失函数为交叉熵
criterion = nn.CrossEntropyLoss()

# 训练
for epoch in range(epochs):
    for i, (x, y) in enumerate(train_loader):
        x = x.to(device) # 数据到GPU
        y = y.to(device) 
        #使用随机梯度下降作为优化器，学习率衰减
        optimizer = optim.SGD(model.parameters(), lr=learning_rate*0.96)
        optimizer.zero_grad()
        #print(x.shape)
        outputs = model(x)   
        # 将y变成0D的tensor
        y = y.squeeze(1)
        outputs = F.log_softmax(outputs, dim=1)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        if (i+1) % 10 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, epochs, i+1, len(train_loader), loss.item()))
# 测试
model.eval()

# 计算测试集准确率
with torch.no_grad():
    correct = 0
    total = 0
    for x, y in test_loader:
        x = x.to(device)
        y = y.to(device)
        outputs = model(x)
        y = y.squeeze(1)
        _, predicted = torch.max(outputs.data, 1)
        total += y.size(0)
        correct += (predicted == y).sum().item()
    print('Test Accuracy of the model on the test images: {} %'.format(100 * correct / total))