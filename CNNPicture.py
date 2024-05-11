import matplotlib.pyplot as plt

# 定义迭代轮数和准确率数据
iters = [150, 200, 250, 300, 350, 400, 450, 500]
train_accs = [97.45, 97.85, 97.95, 98.00, 98.03, 98.06, 98.06, 98.06]
test_accs = [96.92, 98.10, 97.82, 97.84, 98.00, 97.65, 97.92, 97.87]

# 绘制折线图
plt.plot(iters, train_accs, label='Train Accuracy')
plt.plot(iters, test_accs, label='Test Accuracy')
plt.xlabel('Iterations')
plt.ylabel('Accuracy')
plt.title('Accuracy vs. Iterations')
plt.xticks(iters) # 设置横轴刻度
plt.yticks([97.5, 97.8, 98.0, 98.2]) # 设置纵轴刻度
plt.legend()
plt.show()