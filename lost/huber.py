import numpy as np
import matplotlib.pyplot as plt

# 数据
x = np.array([50, 60, 70, 80, 90])
y = np.array([100, 120, 140, 160, 1000])  # 最后一个是异常点

# 假设我们尝试用 y = a * x + b 回归
# 简单线性拟合（最小二乘）
from sklearn.linear_model import LinearRegression, HuberRegressor

x_reshape = x.reshape(-1, 1)

# 用普通回归（等于用MSE）
lr = LinearRegression().fit(x_reshape, y)

# 用鲁棒回归（Huber）
huber = HuberRegressor().fit(x_reshape, y)

# 预测线
x_line = np.linspace(50, 90, 100).reshape(-1, 1)
y_lr = lr.predict(x_line)
y_huber = huber.predict(x_line)

# 画图
plt.scatter(x, y, color='black', label='数据点（含异常）')
plt.plot(x_line, y_lr, color='red', label='普通回归（MSE）')
plt.plot(x_line, y_huber, color='green', label='Huber回归')
plt.xlabel('面积（㎡）')
plt.ylabel('价格（万元）')
plt.title('异常点对回归线的影响')
plt.legend()
plt.grid(True)
plt.show()
