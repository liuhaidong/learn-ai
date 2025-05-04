# 人工训练：用 MAE 损失训练一个最简单的 y = wx 模型

# 初始化
w = 0.0
x = 1.0
y_true = 2.0
lr = 0.1

loss_list = []
w_list = []

# 训练 20 步
for step in range(20):
    y_pred = w * x
    error = y_pred - y_true
    loss = abs(error)
    
    # 梯度计算（符号导数）
    grad = 1.0 if error > 0 else -1.0 if error < 0 else 0.0
    
    # 权重更新
    w -= lr * grad * x

    # 记录
    loss_list.append(loss)
    w_list.append(w)

    print(f"Step {step}: w={w:.2f}, loss={loss:.2f}")

# 可视化
import matplotlib.pyplot as plt
plt.plot(w_list, loss_list, marker='o')
plt.xlabel("权重 w")
plt.ylabel("损失 MAE")
plt.title("MAE 训练过程中 w 和 Loss 的变化")
plt.grid(True)
plt.show()
