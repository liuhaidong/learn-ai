import numpy as np
import matplotlib.pyplot as plt

# 设置真实值y=0，预测值在-3到3之间变化
y = 0
y_pred = np.linspace(-3, 3, 1000)
delta = 1  # Huber损失的delta参数

# 计算Huber损失
def huber_loss(y_true, y_pred, delta):
    error = y_pred - y_true
    is_small_error = np.abs(error) <= delta
    squared_loss = 0.5 * error**2
    linear_loss = delta * (np.abs(error) - 0.5 * delta)
    return np.where(is_small_error, squared_loss, linear_loss)

# 计算Huber损失的导数
def huber_derivative(y_true, y_pred, delta):
    error = y_pred - y_true
    is_small_error = np.abs(error) <= delta
    return np.where(is_small_error, error, delta * np.sign(error))

# 计算损失和导数
huber = huber_loss(y, y_pred, delta)
huber_deriv = huber_derivative(y, y_pred, delta)

# 绘图
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(y_pred, huber)
plt.axvline(x=y, color='r', linestyle='--')
plt.title(f'Huber Loss (δ={delta})')
plt.xlabel('Prediction')
plt.ylabel('Loss')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(y_pred, huber_deriv)
plt.axvline(x=y, color='r', linestyle='--')
plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.title(f'Huber Derivative (δ={delta})')
plt.xlabel('Prediction')
plt.ylabel('Gradient')
plt.grid(True)

plt.tight_layout()
plt.show()
