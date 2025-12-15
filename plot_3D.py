import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle
import itertools

# ----------------------------
# 1. 重新定义傅里叶基类 (用于计算特征)
# ----------------------------
# 必须与训练时保持一致，或者直接从 problem2_solution.py 导入
class FourierBasis:
    def __init__(self, state_dim, order):
        self.order = order
        self.eta = np.array(list(itertools.product(range(order + 1), repeat=state_dim)))

    def get_features(self, state):
        # state 必须归一化到 [0, 1]
        return np.cos(np.pi * np.dot(self.eta, state))

# ----------------------------
# 2. 加载权重
# ----------------------------
try:
    with open('weights.pkl', 'rb') as f:
        data = pickle.load(f)
    W = data['W']
    eta_loaded = data['N']
    print("Weights loaded successfully.")
except FileNotFoundError:
    print("Error: weights.pkl not found. Please train the agent first.")
    exit()

# 检查阶数 p (根据 eta 的大小推断)
# MountainCar state_dim=2. eta 长度为 (p+1)^2
# 如果 len=16 -> p=3; len=9 -> p=2
p_calculated = int(np.sqrt(len(eta_loaded))) - 1
basis = FourierBasis(state_dim=2, order=p_calculated)

# ----------------------------
# 3. 生成网格数据
# ----------------------------
# MountainCar State: position in [-1.2, 0.6], velocity in [-0.07, 0.07]
x = np.linspace(-1.2, 0.6, 50)
y = np.linspace(-0.07, 0.07, 50)
X, Y = np.meshgrid(x, y)

Z_V = np.zeros_like(X)    # 存储 Value Function
Z_Pi = np.zeros_like(X)   # 存储 Policy

low = np.array([-1.2, -0.07])
high = np.array([0.6, 0.07])

print("Computing 3D surfaces...")
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        # 1. 获取原始状态
        pos = X[i, j]
        vel = Y[i, j]
        s_raw = np.array([pos, vel])
        
        # 2. 归一化
        s_scaled = (s_raw - low) / (high - low)
        
        # 3. 计算特征
        phi = basis.get_features(s_scaled)
        
        # 4. 计算 Q 值: W * phi
        # W shape: [n_actions, n_features]
        # phi shape: [n_features]
        # Q_values shape: [n_actions]
        Q_values = np.dot(W, phi)
        
        # 5. 提取 V(s) = max Q(s, a) 和 pi(s) = argmax Q(s, a)
        Z_V[i, j] = -np.max(Q_values) # 注意：通常我们画 -V 以方便看“代价”，或者直接画 max Q (它是负数)
                                      # 这里直接画 max Q，它应该是负值，接近 0 表示好
        Z_V[i, j] = np.max(Q_values) 
        Z_Pi[i, j] = np.argmax(Q_values)

# ----------------------------
# 4. 绘图
# ----------------------------

# Plot 1: Value Function (3D)
fig = plt.figure(figsize=(12, 5))

ax1 = fig.add_subplot(1, 2, 1, projection='3d')
surf1 = ax1.plot_surface(X, Y, Z_V, cmap='viridis', edgecolor='none')
ax1.set_title('Value Function V(s) = max Q(s,a)')
ax1.set_xlabel('Position')
ax1.set_ylabel('Velocity')
ax1.set_zlabel('Value (Negative)')
fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=5)

# Plot 2: Policy (3D / Contour)
# Policy 是离散的 (0, 1, 2)，用 3D 散点或颜色图可能更好看
ax2 = fig.add_subplot(1, 2, 2, projection='3d')
surf2 = ax2.plot_surface(X, Y, Z_Pi, cmap='coolwarm', edgecolor='none')
ax2.set_title('Policy pi(s) (Action 0, 1, 2)')
ax2.set_xlabel('Position')
ax2.set_ylabel('Velocity')
ax2.set_zlabel('Action')
ax2.set_zticks([0, 1, 2])

plt.tight_layout()
plt.show()

# 额外建议：对于 Policy，用 2D 热力图可能更清晰
plt.figure(figsize=(6, 5))
plt.contourf(X, Y, Z_Pi, levels=[-0.5, 0.5, 1.5, 2.5], cmap='coolwarm', alpha=0.8)
plt.colorbar(ticks=[0, 1, 2], label='Action (0:Left, 1:None, 2:Right)')
plt.title('Optimal Policy (2D View)')
plt.xlabel('Position')
plt.ylabel('Velocity')
plt.grid(True)
plt.show()