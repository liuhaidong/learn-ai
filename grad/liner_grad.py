import torch
import matplotlib.pyplot as plt

# 固定随机种子保证可复现
torch.manual_seed(42)

# 模拟数据：输入X (2 samples, 3 features), 权重W (3x2), 偏置b (2)
X = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=False)
W = torch.randn(3, 2, requires_grad=True)  # 需要梯度
b = torch.randn(2, requires_grad=True)      # 需要梯度

# 前向传播
Y = X @ W + b
loss = Y.sum()  # 假设损失函数为Y的元素和（简化示例）

# 反向传播
loss.backward()

# === 打印数值 ===
print("===== 线性层（全连接层） =====")
print("输入 X:\n", X)
print("\n权重 W:\n", W.detach())
print("\n偏置 b:\n", b.detach())
print("\n前向输出 Y:\n", Y.detach())
print("\n梯度 ∂L/∂W:\n", W.grad)
print("\n梯度 ∂L/∂b:\n", b.grad)

# === 可视化 ===
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
ax[0].imshow(X.detach().numpy(), cmap='Blues')
ax[0].set_title("Input X\n(2x3)")
ax[0].axis('off')

grad_W = W.grad.detach().numpy()
im = ax[1].imshow(grad_W, cmap='Reds')
ax[1].set_title(f"Gradient of W\n(3x2)\nSum: {grad_W.sum():.2f}")
ax[1].axis('off')
plt.colorbar(im, ax=ax[1])

grad_b = b.grad.detach().numpy()
ax[2].bar(range(len(grad_b)), grad_b, color='orange')
ax[2].set_title(f"Gradient of b\n(2,)\nSum: {grad_b.sum():.2f}")

plt.suptitle("Linear Layer (FC) Gradient Calculation", fontsize=14)
plt.tight_layout()
plt.show()