import sklearn.naive_bayes as nb
import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('../data/multiple1.txt', delimiter=',', usecols=(0, 1, 2), unpack=False)
print(data.shape)       # (400, 3)
x = data[:, :2]
y = data[:, 2]
model = nb.GaussianNB()     # 创建高斯朴素贝叶斯分类器
model.fit(x, y)
# 绘制背景颜色，显示分类边界线
# 把整个空间分为500*500的网格化矩阵
l, r = x[:, 0].min() - 1, x[:, 0].max() + 1
b, t = x[:, 1].min() - 1, x[:, 1].max() + 1
grid_x, grid_y = np.meshgrid(np.linspace(l, r, 500), np.linspace(b, t, 500))
# 把grid_x与grid_y抻平并在一起成两列，作为测试集x
mesh_x = np.column_stack((grid_x.ravel(), grid_y.ravel()))
mesh_y = model.predict(mesh_x)
grid_z = mesh_y.reshape(grid_x.shape)

# 把所有样本点都绘制出来
plt.figure('Naive Bayes Classification', facecolor='lightgray')
plt.title('Naive Bayes Classification', fontsize=20)
plt.xlabel('x', fontsize=14)
plt.ylabel('y', fontsize=14)
plt.tick_params(labelsize=10)
plt.pcolormesh(grid_x, grid_y, grid_z, cmap='gray')
plt.scatter(x[:, 0], x[:, 1], c=y, cmap='brg', s=80)
plt.show()