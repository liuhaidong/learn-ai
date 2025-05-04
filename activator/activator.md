# 激活函数

---

## 🔢 一、常见激活函数：

| 名称         | 数学形式                                           | 常见用途                 |
| ---------- | ---------------------------------------------- | -------------------- |
| Sigmoid    | $\sigma(x) = \frac{1}{1 + e^{-x}}$             | 二分类输出                |
| Tanh       | $\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$ | 情感、归一化 \[-1, 1]      |
| ReLU       | $\text{ReLU}(x) = \max(0, x)$                  | 卷积神经网络中最常用           |
| Leaky ReLU | $\text{LReLU}(x) = \max(0.01x, x)$             | 防止 ReLU 死亡           |
| Swish      | $\text{Swish}(x) = x \cdot \sigma(x)$          | 更平滑的 ReLU 替代         |
| GELU       | 近似 $x \cdot \Phi(x)$，神经网络中近年流行                 | Transformer / BERT 等 |

---

## 🧪 二、代码：绘制激活函数曲线

```python
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
```

---

## 🔍 三、结果分析（你会看到）：

| 函数             | 特点                                       |
| -------------- | ---------------------------------------- |
| **Sigmoid**    | 输出范围 $(0, 1)$，容易饱和（梯度消失）                 |
| **Tanh**       | 输出范围 $(-1, 1)$，居中但也容易饱和                  |
| **ReLU**       | 0 以下恒为 0，速度快，稀疏性强（但可能死亡）                 |
| **Leaky ReLU** | 小于 0 区域有“微弱”梯度，解决 ReLU 死亡问题              |
| **Swish**      | 平滑 ReLU，性能更强（Google 提出）                  |
| **GELU**       | 类似 Swish，但更自然地模拟神经行为（Transformer/BERT使用） |

---

## ✅ 四、总结一句话：

> 激活函数的选择对模型性能影响极大，不同函数的**非线性、平滑性、梯度传播**等性质，直接决定了网络的训练效果。

# 激活函数 + 损失函数



---

## 🧪 一、实验设定

我们来训练一个最简单的神经元：

* 输入：固定 $x = 1$
* 权重 $w$：需要学习
* 输出 $\hat{y} = \text{激活}(w \cdot x)$
* 标签 $y = 1$（正类）

我们尝试以下组合：

| 激活函数    | 损失函数        | 常见搭配理由                 |
| ------- | ----------- | ---------------------- |
| Sigmoid | BCE（二分类交叉熵） | 常用于二分类输出层              |
| Tanh    | MSE         | \[-1, 1] 范围，适用于回归或对称分类 |
| ReLU    | MSE         | 可用于回归                  |

---

## 🧠 二、反向传播原理

对于权重 $w$，我们更新公式：

$$
w := w - \eta \cdot \frac{dL}{dw}
$$

而这个梯度是链式法则：

$$
\frac{dL}{dw} = \frac{dL}{d\hat{y}} \cdot \frac{d\hat{y}}{dz} \cdot \frac{dz}{dw}
$$

其中 $z = w \cdot x$，所以我们主要关注的是：

> 激活函数（$\hat{y} = \text{act}(z)$） 和 损失函数（$L(\hat{y}, y)$） 的组合影响了梯度。

---

## 💻 三、Python 演示代码：3种组合比较

```python
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
    ("Tanh + MSE", tanh, mse),
    ("ReLU + MSE", relu, mse)
]

x = 1.0
y_true = 1.0
eta = 0.1
steps = 30

plt.figure(figsize=(12, 4))

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

    plt.subplot(1, 3, i)
    plt.plot(loss_list, marker='o')
    plt.title(name)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.grid(True)

plt.suptitle("激活函数 + 损失函数 组合对梯度下降的影响", fontsize=14)
plt.tight_layout()
plt.show()
```

---

## 📊 四、你会看到什么？

| 组合                | 训练表现                    |
| ----------------- | ----------------------- |
| **Sigmoid + BCE** | 收敛稳定且较快，适合二分类           |
| **Tanh + MSE**    | 收敛较慢，但最终可收敛（可能有梯度消失）    |
| **ReLU + MSE**    | 训练前期如果 w < 0 梯度为 0，可能卡死 |

---

## ✅ 五、结论总结

| 激活函数    | 输出范围    | 导数行为      | 最佳搭配损失函数             |
| ------- | ------- | --------- | -------------------- |
| Sigmoid | (0, 1)  | 容易饱和、梯度变小 | Binary Cross Entropy |
| Tanh    | (-1, 1) | 也容易饱和     | MSE                  |
| ReLU    | \[0, ∞) | 左边梯度为 0   | MSE（适用于正值）           |

> 📌 **激活函数决定梯度流动是否顺畅，损失函数决定误差对梯度的敏感程度。它们的组合会直接决定梯度是否容易消失/爆炸、是否可以稳定学习。**


# 推导 **Sigmoid 函数的导数**



### 🔹 一、Sigmoid 函数定义

Sigmoid 函数是一种常用的 S 形激活函数，定义如下：

$$
\sigma(x) = \frac{1}{1 + e^{-x}}
$$

---

### 🔹 二、求导目标

我们要计算的是它对 $x$ 的导数：

$$
\frac{d}{dx} \sigma(x) = ?
$$

---

### 🔹 三、推导过程

我们令：

$$
y = \sigma(x) = \frac{1}{1 + e^{-x}}
$$

对这个式子求导，使用**链式法则和商法则**：

---

#### 【第一种推导方式：用链式法则和商法则】

$$
\frac{dy}{dx} = \frac{d}{dx} \left( \frac{1}{1 + e^{-x}} \right)
$$

设：

* 分母：$u(x) = 1 + e^{-x}$
* 所以：$y = \frac{1}{u(x)}$

利用复合函数求导法则：

$$
\frac{dy}{dx} = -\frac{1}{(u(x))^2} \cdot \frac{du}{dx}
$$

计算 $\frac{du}{dx}$：

$$
\frac{du}{dx} = \frac{d}{dx}(1 + e^{-x}) = -e^{-x}
$$

所以：

$$
\frac{dy}{dx} = -\frac{1}{(1 + e^{-x})^2} \cdot (-e^{-x}) = \frac{e^{-x}}{(1 + e^{-x})^2}
$$

---

#### 【进一步化简】

我们用 $\sigma(x)$ 本身来表示这个结果。

记住：

$$
\sigma(x) = \frac{1}{1 + e^{-x}} \Rightarrow 1 - \sigma(x) = \frac{e^{-x}}{1 + e^{-x}}
$$

于是：

$$
\sigma'(x) = \frac{e^{-x}}{(1 + e^{-x})^2} = \sigma(x)(1 - \sigma(x))
$$

---

### 🔹 四、最终结论

$$
\boxed{ \frac{d}{dx} \sigma(x) = \sigma(x)(1 - \sigma(x)) }
$$

这个结论非常优雅，实际中我们经常直接使用它。它说明：

* 当 $\sigma(x)$ 接近 0 或 1 时，导数接近 0（梯度消失）
* 最大导数值出现在 $\sigma(x) = 0.5$ 时（值为 0.25）




