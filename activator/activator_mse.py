import numpy as np
import matplotlib.pyplot as plt

# 输入和标签
x = 1.0
y_true = 2.0
lr = 0.1
steps = 30

# 激活函数定义
def relu(z): return np.maximum(0, z)
def relu_grad(z): return 1.0 if z > 0 else 0.0

def sigmoid(z): return 1 / (1 + np.exp(-z))
def sigmoid_grad(z): s = sigmoid(z); return s * (1 - s)

# 训练函数
def train(activation, activation_grad, name):
    w = 0.5
    w_list, loss_list = [], []

    for step in range(steps):
        z = w * x
        y_pred = activation(z)
        loss = 0.5 * (y_pred - y_true) ** 2

        # 反向传播链式法则
        dL_dy = y_pred - y_true
        dy_dz = activation_grad(z)
        dz_dw = x
        grad = dL_dy * dy_dz * dz_dw

        # 权重更新
        w -= lr * grad

        w_list.append(w)
        loss_list.append(loss)

    return w_list, loss_list

# 对比训练
relu_w, relu_loss = train(relu, relu_grad, 'ReLU')
sigmoid_w, sigmoid_loss = train(sigmoid, sigmoid_grad, 'Sigmoid')

# 绘图
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(relu_loss, label="ReLU")
plt.plot(sigmoid_loss, label="Sigmoid")
plt.title("Loss Curve")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1,2,2)
plt.plot(relu_w, label="ReLU")
plt.plot(sigmoid_w, label="Sigmoid")
plt.title("Weight Update Curve")
plt.xlabel("Step")
plt.ylabel("Weight")
plt.legend()

plt.tight_layout()
plt.show()
