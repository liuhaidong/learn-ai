import torch
import matplotlib.pyplot as plt

# 生成输入数据
x = torch.linspace(-2, 2, 10, requires_grad=True)  # 减少点数便于显示
y = torch.relu(x)  # 前向传播
y.sum().backward()  # 假设损失函数为y的元素和

# === 打印数值 ===
print("\n===== 非线性层（ReLU） =====")
print("输入 x:\n", x.detach().numpy().round(2))
print("\n前向输出 y:\n", y.detach().numpy().round(2))
print("\n梯度 ∂y/∂x:\n", x.grad.numpy().round(2))

# === 可视化 ===
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(x.detach().numpy(), y.detach().numpy(), 'o-', label='ReLU(x)')
plt.title("ReLU Forward Pass")
plt.xlabel("Input x")
plt.ylabel("Output ReLU(x)")
plt.grid(True)

plt.subplot(1, 2, 2)
plt.stem(x.detach().numpy(), x.grad.detach().numpy(), linefmt='r-', markerfmt='ro')
plt.title("ReLU Gradient (Backward Pass)")
plt.xlabel("Input x")
plt.ylabel("Gradient")
plt.ylim(-0.1, 1.1)
plt.grid(True)

plt.suptitle("Nonlinearity (ReLU) Gradient Calculation", fontsize=14)
plt.tight_layout()
plt.show()