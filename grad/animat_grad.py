import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import colors

# 设置矩阵
A = np.array([[1, 2], [3, 4]], dtype=float)
B = np.array([[5, 6], [7, 8]], dtype=float)
C = A @ B
L = np.sum(C)

# 计算梯度
dL_dA = np.array([[np.sum(B[0,:]), np.sum(B[1,:])], 
                  [np.sum(B[0,:]), np.sum(B[1,:])]])
dL_dB = np.array([[np.sum(A[:,0]), np.sum(A[:,0])], 
                  [np.sum(A[:,1]), np.sum(A[:,1])]])

# 创建画布
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Matrix Grad Calc", fontsize=16)

# 颜色标准化
norm = colors.Normalize(vmin=0, vmax=15)

# 存储绘图对象
text_objects = []
image_objects = []

def init():
    global text_objects, image_objects
    text_objects = []
    image_objects = []
    
    for ax in (ax1, ax2, ax3):
        ax.clear()
        ax.set_xticks([])
        ax.set_yticks([])
    
    # 绘制初始矩阵
    ax1.set_title("Matrix A")
    img1 = ax1.imshow(A, cmap='Blues', norm=norm)
    image_objects.append(img1)
    for i in range(2):
        for j in range(2):
            t = ax1.text(j, i, f"{A[i,j]}", ha='center', va='center', color='white' if A[i,j] > 7 else 'black')
            text_objects.append(t)
    
    ax2.set_title("Matrix B")
    img2 = ax2.imshow(B, cmap='Greens', norm=norm)
    image_objects.append(img2)
    for i in range(2):
        for j in range(2):
            t = ax2.text(j, i, f"{B[i,j]}", ha='center', va='center', color='white' if B[i,j] > 7 else 'black')
            text_objects.append(t)
    
    ax3.set_title("Grad Calc")
    return image_objects + text_objects

def update(frame):
    global text_objects, image_objects
    for ax in (ax1, ax2, ax3):
        ax.clear()
        ax.set_xticks([])
        ax.set_yticks([])
    
    text_objects = []
    image_objects = []
    
    # 分步演示
    if frame < 5:
        ax1.set_title("Matrix A")
        img1 = ax1.imshow(A, cmap='Blues', norm=norm)
        image_objects.append(img1)
        for i in range(2):
            for j in range(2):
                t = ax1.text(j, i, f"{A[i,j]}", ha='center', va='center', color='white' if A[i,j] > 7 else 'black')
                text_objects.append(t)
        
        ax2.set_title("Matrix B")
        img2 = ax2.imshow(B, cmap='Greens', norm=norm)
        image_objects.append(img2)
        for i in range(2):
            for j in range(2):
                t = ax2.text(j, i, f"{B[i,j]}", ha='center', va='center', color='white' if B[i,j] > 7 else 'black')
                text_objects.append(t)
        
        ax3.set_title("Step 1:  C = A @ B")
        C_vis = np.zeros_like(C)
        progress = min(1.0, frame/4)
        for i in range(2):
            for j in range(2):
                C_vis[i,j] = progress * C[i,j]
                t = ax3.text(j, i, f"{C[i,j]:.1f}" if progress>0 else "", ha='center', va='center')
                text_objects.append(t)
        img3 = ax3.imshow(C_vis, cmap='Reds', norm=norm)
        image_objects.append(img3)
    
    elif frame < 10:
        ax3.set_title("Step 2:  L = sum(C)")
        img3 = ax3.imshow(C, cmap='Reds', norm=norm)
        image_objects.append(img3)
        for i in range(2):
            for j in range(2):
                t = ax3.text(j, i, f"{C[i,j]}", ha='center', va='center', color='white' if C[i,j] > 7 else 'black')
                text_objects.append(t)
        if frame == 9:
            t = ax3.text(1.5, 2.2, f"L = {L}", ha='center', fontsize=12)
            text_objects.append(t)
    
    elif frame < 15:
        ax1.set_title("Calc ∂L/∂A")
        ax2.set_title("B col sum")
        
        img2 = ax2.imshow(B, cmap='Greens', norm=norm)
        image_objects.append(img2)
        col_sums = np.sum(B, axis=1)
        for j in range(2):
            t = ax2.text(j, 0.5, f"Sum={col_sums[j]}", ha='center', va='center')
            text_objects.append(t)
            if frame > 12:
                t = ax2.text(j, 1.5, f"→ A grad col{j}", ha='center', color='red')
                text_objects.append(t)
        
        dA_vis = np.zeros_like(dL_dA)
        progress = min(1.0, (frame-10)/4)
        for i in range(2):
            for j in range(2):
                dA_vis[i,j] = progress * dL_dA[i,j]
                t = ax1.text(j, i, f"{dL_dA[i,j]:.1f}" if progress>0 else "", ha='center', va='center')
                text_objects.append(t)
        img1 = ax1.imshow(dA_vis, cmap='Oranges', norm=norm)
        image_objects.append(img1)
    
    else:
        ax1.set_title("∂L/∂A Result")
        img1 = ax1.imshow(dL_dA, cmap='Oranges', norm=norm)
        image_objects.append(img1)
        for i in range(2):
            for j in range(2):
                t = ax1.text(j, i, f"{dL_dA[i,j]}", ha='center', va='center')
                text_objects.append(t)
        
        ax2.set_title("Calc ∂L/∂B")
        ax3.set_title("A row sum")
        
        img3 = ax3.imshow(A, cmap='Blues', norm=norm)
        image_objects.append(img3)
        row_sums = np.sum(A, axis=0)
        for i in range(2):
            t = ax3.text(0.5, i, f"Sum={row_sums[i]}", ha='center', va='center')
            text_objects.append(t)
            if frame > 17:
                t = ax3.text(1.5, i, f"→ B grad {i}", ha='center', color='red')
                text_objects.append(t)
        
        dB_vis = np.zeros_like(dL_dB)
        progress = min(1.0, (frame-15)/4)
        for i in range(2):
            for j in range(2):
                dB_vis[i,j] = progress * dL_dB[i,j]
                t = ax2.text(j, i, f"{dL_dB[i,j]:.1f}" if progress>0 else "", ha='center', va='center')
                text_objects.append(t)
        img2 = ax2.imshow(dB_vis, cmap='Purples', norm=norm)
        image_objects.append(img2)
    
    return image_objects + text_objects

# 创建动画
ani = FuncAnimation(fig, update, frames=20, init_func=init, interval=1000, blit=True)
plt.tight_layout()
plt.show()