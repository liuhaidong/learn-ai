import numpy as np
import matplotlib.pyplot as plt

# 输入范围
x = np.linspace(-6, 6, 400)

# 定义激活函数
def sigmoid(x): return 1 / (1 + np.exp(-x))
def tanh(x): return np.tanh(x)
def relu(x): return np.maximum(0, x)
def leaky_relu(x): return np.where(x > 0, x, 0.01 * x)
def swish(x): return x * sigmoid(x)
def gelu(x): return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

# 激活函数集合
activations = {
    'Sigmoid': sigmoid,
    'Tanh': tanh,
    'ReLU': relu,
    'Leaky ReLU': leaky_relu,
    'Swish': swish,
    'GELU': gelu
}

# 绘图
plt.figure(figsize=(12, 8))
for i, (name, func) in enumerate(activations.items(), 1):
    plt.subplot(2, 3, i)
    plt.plot(x, func(x), label=name, color='blue')
    plt.title(name)
    plt.axhline(0, color='gray', lw=0.5, ls='--')
    plt.axvline(0, color='gray', lw=0.5, ls='--')
    plt.grid(True)
    plt.legend()
plt.tight_layout()
plt.suptitle("激活函数曲线比较", fontsize=16, y=1.02)
plt.show()