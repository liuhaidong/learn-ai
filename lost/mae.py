import numpy as np
import matplotlib.pyplot as plt

# 设置真实值y=0，预测值在-3到3之间变化
y = 0
y_pred = np.linspace(-3, 3, 1000)

# 计算MAE损失和导数
mae_loss = np.abs(y_pred - y)
mae_derivative = np.sign(y_pred - y)

# 绘图
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(y_pred, mae_loss)
plt.axvline(x=y, color='r', linestyle='--')
plt.title('MAE Loss')
plt.xlabel('Prediction')
plt.ylabel('Loss')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(y_pred, mae_derivative)
plt.axvline(x=y, color='r', linestyle='--')
plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.title('MAE Derivative')
plt.xlabel('Prediction')
plt.ylabel('Gradient')
plt.grid(True)
plt.ylim(-1.5, 1.5)

plt.tight_layout()
plt.show()