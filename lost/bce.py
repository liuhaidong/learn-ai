import numpy as np
import matplotlib.pyplot as plt

# 设置两种情况：y=1和y=0
y_pred = np.linspace(0.01, 0.99, 1000)  # 避免0和1导致的数值问题

# 计算y=1时的交叉熵损失和导数
bce_loss_y1 = -np.log(y_pred)
bce_derivative_y1 = -1/y_pred

# 计算y=0时的交叉熵损失和导数
bce_loss_y0 = -np.log(1 - y_pred)
bce_derivative_y0 = 1/(1 - y_pred)

# 绘图
plt.figure(figsize=(12, 10))

plt.subplot(2, 2, 1)
plt.plot(y_pred, bce_loss_y1)
plt.title('Binary Cross-Entropy Loss (y=1)')
plt.xlabel('Prediction')
plt.ylabel('Loss')
plt.grid(True)

plt.subplot(2, 2, 2)
plt.plot(y_pred, bce_derivative_y1)
plt.title('Binary Cross-Entropy Derivative (y=1)')
plt.xlabel('Prediction')
plt.ylabel('Gradient')
plt.grid(True)

plt.subplot(2, 2, 3)
plt.plot(y_pred, bce_loss_y0)
plt.title('Binary Cross-Entropy Loss (y=0)')
plt.xlabel('Prediction')
plt.ylabel('Loss')
plt.grid(True)

plt.subplot(2, 2, 4)
plt.plot(y_pred, bce_derivative_y0)
plt.title('Binary Cross-Entropy Derivative (y=0)')
plt.xlabel('Prediction')
plt.ylabel('Gradient')
plt.grid(True)

plt.tight_layout()
plt.show()
