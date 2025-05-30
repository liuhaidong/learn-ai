import numpy as np
import matplotlib.pyplot as plt

# 激活函数及导数
def sigmoid(z):
    s = 1 / (1 + np.exp(-z))
    return s, s * (1 - s)

def tanh(z):
    t = np.tanh(z)
    return t, 1 - t**2

def relu(z):
    z = np.asarray(z)  # 强制转换为 NumPy 数组（即便是标量）
    return np.maximum(0, z), (z > 0).astype(float)

# 损失函数及导数
def mse(y_pred, y_true):
    loss = 0.5 * (y_pred - y_true) ** 2
    grad = y_pred - y_true
    return loss, grad

def bce(y_pred, y_true):
    eps = 1e-8
    loss = - (y_true * np.log(y_pred + eps) + (1 - y_true) * np.log(1 - y_pred + eps))
    grad = (y_pred - y_true) / ((y_pred + eps) * (1 - y_pred + eps))
    return loss, grad

# 训练设置
combinations = [
    ("Sigmoid + BCE", sigmoid, bce),
    ("Sigmoid + MSE", sigmoid, mse),
    ("Tanh + MSE", tanh, mse),
    ("ReLU + MSE", relu, mse)
]

x = 1.0
y_true = 2.0
eta = 0.1
steps = 80

plt.figure(figsize=(15, 8))

for i, (name, activation, loss_fn) in enumerate(combinations, 1):
    w = -2.0  # 初始远离目标
    w_list, loss_list = [], []
    
    for _ in range(steps):
        z = w * x
        a, da_dz = activation(z)
        loss, dL_da = loss_fn(a, y_true)

        # 反向传播梯度
        grad = dL_da * da_dz * x
        w -= eta * grad

        w_list.append(w)
        loss_list.append(loss)

    # Plot loss and weight values in subplots
    plt.subplot(4, 2, 2*i-1)
    plt.plot(loss_list, marker='o', color='r')
    plt.title(f"{name} - Loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.grid(True)
    
    plt.subplot(4, 2, 2*i)
    plt.plot(w_list, marker='o', color='b')
    plt.title(f"{name} - Weight Value")
    plt.xlabel("Step")
    plt.ylabel("Weight (w)")
    plt.grid(True)

plt.suptitle("激活函数 + 损失函数 组合对梯度下降的影响 (Loss and Weight Values)", fontsize=14)
plt.tight_layout()
plt.show()