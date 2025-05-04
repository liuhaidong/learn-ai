import torch

A = torch.randn(2, 3, requires_grad=True)
B = torch.randn(3, 4, requires_grad=True)
C = A @ B
loss = C.sum()  # 或其他标量损失
loss.backward()

# 手动计算梯度
dL_dC = torch.ones_like(C)
dL_dA = dL_dC @ B.T
dL_dB = A.T @ dL_dC

print(A.grad)
print(B.grad)
# 验证
print(torch.allclose(A.grad, dL_dA))  # True
print(torch.allclose(B.grad, dL_dB))  # True