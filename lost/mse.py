import numpy as np
import matplotlib.pyplot as plt

# 设置真实值y=0，预测值在-3到3之间变化
y = 0
y_pred = np.linspace(-3, 3, 1000)

# 计算MSE损失和导数
mse_loss = (y_pred - y)**2
mse_derivative = 2 * (y_pred - y)

# 绘图
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(y_pred, mse_loss)
plt.axvline(x=y, color='r', linestyle='--')
plt.title('MSE Loss')
plt.xlabel('Prediction')
plt.ylabel('Loss')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(y_pred, mse_derivative)
plt.axvline(x=y, color='r', linestyle='--')
plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.title('MSE Derivative')
plt.xlabel('Prediction')
plt.ylabel('Gradient')
plt.grid(True)

plt.tight_layout()
plt.show()