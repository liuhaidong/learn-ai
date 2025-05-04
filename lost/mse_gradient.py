import matplotlib.pyplot as plt

# 初始值
w = 0.0
x = 1.0
y_true = 2.0
lr = 0.1

w_list = []
loss_list = []

# 训练20步
for step in range(20):
    y_pred = w * x
    error = y_pred - y_true
    loss = 0.5 * error ** 2
    grad = error * x  # dL/dw

    # 梯度更新
    w = w - lr * grad

    w_list.append(w)
    loss_list.append(loss)

    print(f"Step {step}: w = {w:.3f}, loss = {loss:.3f}")

# 可视化
plt.plot(w_list, loss_list, marker='o')
plt.title("MSE 梯度下降过程")
plt.xlabel("权重 w")
plt.ylabel("损失 Loss")
plt.grid(True)
plt.show()
