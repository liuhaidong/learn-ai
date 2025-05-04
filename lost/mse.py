import numpy as np
import matplotlib.pyplot as plt

# 假设 x=2, y=4
x = 2
y = 4

# 在不同的 w 取值下计算损失 L(w)
w_values = np.linspace(0, 4, 100)
loss_values = 0.5 * (x * w_values - y)**2

plt.plot(w_values, loss_values)
plt.xlabel("w")
plt.ylabel("Loss")
plt.title("均方误差 MSE 损失函数曲线")
plt.grid(True)
plt.show()
