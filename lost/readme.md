# 损失函数分析与可视化

下面我将详细分析每种损失函数的公式、求导过程、导数结果，并用Python代码绘制相应的函数曲线和导数曲线。

## 1. 均方误差 (MSE)

### 公式
$$\text{MSE} = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$$

对于单个样本：
$$\text{MSE} = (y - \hat{y})^2$$

### 求导过程
$$\frac{d\text{MSE}}{d\hat{y}} = \frac{d}{d\hat{y}}(y - \hat{y})^2 = 2(y - \hat{y}) \cdot (-1) = -2(y - \hat{y})$$

### 导数结果
$$\frac{d\text{MSE}}{d\hat{y}} = -2(y - \hat{y}) = 2(\hat{y} - y)$$

### 可视化代码

```python
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
```

## 2. 绝对误差 (MAE)

### 公式
$$\text{MAE} = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|$$

对于单个样本：
$$\text{MAE} = |y - \hat{y}|$$

### 求导过程
$$\frac{d\text{MAE}}{d\hat{y}} = \frac{d}{d\hat{y}}|y - \hat{y}| = 
\begin{cases} 
-1, & \text{if } y > \hat{y} \\
\text{undefined}, & \text{if } y = \hat{y} \\
1, & \text{if } y < \hat{y}
\end{cases}$$

也可以写为：
$$\frac{d\text{MAE}}{d\hat{y}} = \text{sign}(\hat{y} - y)$$

### 导数结果
$$\frac{d\text{MAE}}{d\hat{y}} = 
\begin{cases} 
-1, & \text{if } y > \hat{y} \\
\text{undefined}, & \text{if } y = \hat{y} \\
1, & \text{if } y < \hat{y}
\end{cases}$$

### 可视化代码

```python
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
```

## 3. Huber 损失

### 公式
$$\text{Huber}(y, \hat{y}) = 
\begin{cases} 
\frac{1}{2}(y - \hat{y})^2, & \text{if } |y - \hat{y}| \leq \delta \\
\delta|y - \hat{y}| - \frac{1}{2}\delta^2, & \text{otherwise}
\end{cases}$$

### 求导过程
$$\frac{d\text{Huber}}{d\hat{y}} = 
\begin{cases} 
\frac{d}{d\hat{y}}\frac{1}{2}(y - \hat{y})^2 = -(y - \hat{y}), & \text{if } |y - \hat{y}| \leq \delta \\
\frac{d}{d\hat{y}}(\delta|y - \hat{y}| - \frac{1}{2}\delta^2) = -\delta \cdot \text{sign}(y - \hat{y}), & \text{otherwise}
\end{cases}$$

### 导数结果
$$\frac{d\text{Huber}}{d\hat{y}} = 
\begin{cases} 
-(y - \hat{y}) = \hat{y} - y, & \text{if } |y - \hat{y}| \leq \delta \\
-\delta \cdot \text{sign}(y - \hat{y}) = \delta \cdot \text{sign}(\hat{y} - y), & \text{otherwise}
\end{cases}$$

### 可视化代码

```python
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
```

## 4. 交叉熵损失

### 二分类交叉熵公式
$$\text{BCE}(y, \hat{y}) = -[y \log(\hat{y}) + (1-y) \log(1-\hat{y})]$$

### 多分类交叉熵公式
$$\text{CE}(y, \hat{y}) = -\sum_{i=1}^{C} y_i \log(\hat{y}_i)$$

### 求导过程（二分类）
$$\frac{d\text{BCE}}{d\hat{y}} = -\left[\frac{y}{\hat{y}} + (1-y) \cdot \frac{-1}{1-\hat{y}}\right] = -\frac{y}{\hat{y}} + \frac{1-y}{1-\hat{y}}$$

### 导数结果（二分类）
$$\frac{d\text{BCE}}{d\hat{y}} = \frac{\hat{y} - y}{\hat{y}(1-\hat{y})}$$

### 可视化代码

```python
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
```

## 5. Hinge 损失 (SVM)

### 公式
$$\text{Hinge}(y, \hat{y}) = \max(0, 1 - y \cdot \hat{y})$$

其中，y ∈ {-1, 1}是真实标签，$\hat{y}$是模型预测的分数。

### 求导过程
$$\frac{d\text{Hinge}}{d\hat{y}} = 
\begin{cases} 
0, & \text{if } y \cdot \hat{y} > 1 \\
-y, & \text{if } y \cdot \hat{y} < 1
\end{cases}$$

### 导数结果
$$\frac{d\text{Hinge}}{d\hat{y}} = 
\begin{cases} 
0, & \text{if } y \cdot \hat{y} > 1 \\
-y, & \text{if } y \cdot \hat{y} < 1
\end{cases}$$

### 可视化代码

```python
import numpy as np
import matplotlib.pyplot as plt

# 设置预测值在-3到3之间变化
y_pred = np.linspace(-3, 3, 1000)

# 计算y=1和y=-1时的hinge损失和导数
def hinge_loss(y, y_pred):
    return np.maximum(0, 1 - y * y_pred)

def hinge_derivative(y, y_pred):
    return np.where(y * y_pred >= 1, 0, -y)

# 计算损失和导数
hinge_loss_y1 = hinge_loss(1, y_pred)
hinge_derivative_y1 = hinge_derivative(1, y_pred)

hinge_loss_ym1 = hinge_loss(-1, y_pred)
hinge_derivative_ym1 = hinge_derivative(-1, y_pred)

# 绘图
plt.figure(figsize=(12, 10))

plt.subplot(2, 2, 1)
plt.plot(y_pred, hinge_loss_y1)
plt.axvline(x=1, color='r', linestyle='--')
plt.title('Hinge Loss (y=1)')
plt.xlabel('Prediction')
plt.ylabel('Loss')
plt.grid(True)

plt.subplot(2, 2, 2)
plt.plot(y_pred, hinge_derivative_y1)
plt.axvline(x=1, color='r', linestyle='--')
plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.title('Hinge Derivative (y=1)')
plt.xlabel('Prediction')
plt.ylabel('Gradient')
plt.grid(True)
plt.ylim(-1.5, 1.5)

plt.subplot(2, 2, 3)
plt.plot(y_pred, hinge_loss_ym1)
plt.axvline(x=-1, color='r', linestyle='--')
plt.title('Hinge Loss (y=-1)')
plt.xlabel('Prediction')
plt.ylabel('Loss')
plt.grid(True)

plt.subplot(2, 2, 4)
plt.plot(y_pred, hinge_derivative_ym1)
plt.axvline(x=-1, color='r', linestyle='--')
plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.title('Hinge Derivative (y=-1)')
plt.xlabel('Prediction')
plt.ylabel('Gradient')
plt.grid(True)
plt.ylim(-1.5, 1.5)

plt.tight_layout()
plt.show()
```

## 总结

| 损失函数 | 公式 | 导数 | 特点 |
|---------|------|------|------|
| MSE | $(y - \hat{y})^2$ | $2(\hat{y} - y)$ | 平滑，对异常值敏感 |
| MAE | $\|y - \hat{y}\|$ | $\text{sign}(\hat{y} - y)$ | 在0处不可导，对异常值较稳健 |
| Huber | $\begin{cases} \frac{1}{2}(y - \hat{y})^2, & \text{if } \|y - \hat{y}\| \leq \delta \\ \delta\|y - \hat{y}\| - \frac{1}{2}\delta^2, & \text{otherwise} \end{cases}$ | $\begin{cases} \hat{y} - y, & \text{if } \|y - \hat{y}\| \leq \delta \\ \delta \cdot \text{sign}(\hat{y} - y), & \text{otherwise} \end{cases}$ | 结合MSE和MAE的优点，平滑且对异常值稳健 |
| 交叉熵 | $-[y \log(\hat{y}) + (1-y) \log(1-\hat{y})]$ | $\frac{\hat{y} - y}{\hat{y}(1-\hat{y})}$ | 适用于分类问题，在正确标签处梯度较大 |
| Hinge | $\max(0, 1 - y \cdot \hat{y})$ | $\begin{cases} 0, & \text{if } y \cdot \hat{y} > 1 \\ -y, & \text{if } y \cdot \hat{y} < 1 \end{cases}$ | 用于SVM，在正确分类且超过边界时梯度为0 |