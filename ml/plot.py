import numpy as np
import matplotlib.pyplot as plt

n = 1000
# 生成网格化坐标矩阵
x, y = np.meshgrid(np.linspace(-3, 3, n), np.linspace(-3, 3, n))
# 根据每个网格点坐标，通过某个公式计算z高度坐标
z = (2 - x/2 + x**2 + y**3) * np.exp(-x**2 - y**2)
plt.figure('Contour', facecolor='lightgray')
plt.title('Contour', fontsize=20)
plt.xlabel('x', fontsize=14)
plt.ylabel('y', fontsize=14)
plt.tick_params(labelsize=10)
plt.grid(linestyle='-')
plt.contourf(x, y, z, 8, cmap='jet')
cntr = plt.contour(x, y, z, 8, colors='black', linewidths=0.5)
plt.clabel(cntr, inline_spacing=1, fmt='%.1f', fontsize=10) 
plt.show()

from mpl_toolkits.mplot3d import axes3d

n = 500
x = np.random.normal(0, 1, n)
y = np.random.normal(0, 1, n)
z = np.random.normal(0, 1, n)
plt.figure('3D scatter')
ax3d = plt.gca(projection='3d')
d = x ** 2 + y ** 2 + z ** 2
ax3d.scatter(x, y, z, s=70, c=d, alpha=0.7, cmap='jet')
ax3d.set_xlabel('x')
ax3d.set_ylabel('y')
ax3d.set_zlabel('z')
plt.tight_layout()
plt.show()