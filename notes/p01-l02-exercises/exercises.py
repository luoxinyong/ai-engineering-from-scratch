"""Phase 01 / Lesson 02 练习 1、2 + 两个补充实验"""
import numpy as np
from fractions import Fraction

np.set_printoptions(precision=6, suppress=True)


def det2(m):
    return m[0][0] * m[1][1] - m[0][1] * m[1][0]


def inv2(m):
    d = det2(m)
    if d == 0:
        raise ValueError("奇异矩阵，无逆")
    return [[m[1][1] / d, -m[0][1] / d],
            [-m[1][0] / d, m[0][0] / d]]


def matmul(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(len(b)))
             for j in range(len(b[0]))] for i in range(len(a))]


# ============================================================
# 练习 1：验证 A @ A⁻¹ = I，以及 det=0 时会怎样
# ============================================================
print("=" * 62)
print("练习 1：验证逆矩阵")
print("=" * 62)

cases = [
    ("A1", [[1, 2], [3, 4]]),
    ("A2", [[4, 7], [2, 6]]),
    ("A3", [[2, 0], [0, 5]]),
    ("A4 (奇异)", [[1, 2], [2, 4]]),      # 第二行 = 2×第一行
]
for name, M in cases:
    d = det2(M)
    print(f"\n{name} = {M}    det = {d}")
    if d == 0:
        print("  → det=0，矩阵奇异，逆不存在")
        print("  → 几何上：它把面积压成 0（2D 平面被压成一条直线）")
        print(f"  → rank = {np.linalg.matrix_rank(np.array(M, float))}（不满秩）")
        continue
    Minv = inv2(M)
    prod = matmul(M, Minv)
    print(f"  A⁻¹ = [[{Minv[0][0]:.4f}, {Minv[0][1]:.4f}], "
          f"[{Minv[1][0]:.4f}, {Minv[1][1]:.4f}]]")
    print(f"  A @ A⁻¹ = [[{prod[0][0]:.10f}, {prod[0][1]:.10f}], "
          f"[{prod[1][0]:.10f}, {prod[1][1]:.10f}]]")
    err = max(abs(prod[i][j] - (1 if i == j else 0)) for i in range(2) for j in range(2))
    print(f"  与单位阵最大偏差 = {err:.2e}  {'✅' if err < 1e-9 else '⚠️ 浮点误差'}")

# 病态矩阵：det 接近 0 但不等于 0
print("\n--- 补充：det 接近 0 会发生什么（病态） ---")
for eps in [1e-1, 1e-4, 1e-8, 1e-12]:
    M = np.array([[1.0, 1.0], [1.0, 1.0 + eps]])
    cond = np.linalg.cond(M)
    Minv = np.linalg.inv(M)
    err = np.abs(M @ Minv - np.eye(2)).max()
    print(f"  eps={eps:.0e}  det={np.linalg.det(M):+.2e}  "
          f"条件数={cond:.2e}  |A@A⁻¹-I| 误差={err:.2e}  逆元最大值={np.abs(Minv).max():.2e}")
print("  → det 越接近 0，逆的元素越爆炸，数值误差越大 = 病态(ill-conditioned)")

# ============================================================
# 练习 2：3×3 逆矩阵（伴随矩阵法），对拍 numpy
# ============================================================
print("\n" + "=" * 62)
print("练习 2：3×3 求逆（伴随矩阵法）")
print("=" * 62)


def minor(m, i, j):
    return [[m[r][c] for c in range(len(m)) if c != j]
            for r in range(len(m)) if r != i]


def det3(m):
    return sum((-1) ** j * m[0][j] * det2(minor(m, 0, j)) for j in range(3))


def inv3(m):
    d = det3(m)
    if d == 0:
        raise ValueError("奇异矩阵，无逆")
    # 余子式矩阵 → 代数余子式 → 转置 = 伴随矩阵 → 除以行列式
    cof = [[(-1) ** (i + j) * det2(minor(m, i, j)) for j in range(3)] for i in range(3)]
    adj = [[cof[j][i] for j in range(3)] for i in range(3)]     # 转置
    return [[adj[i][j] / d for j in range(3)] for i in range(3)]


M = [[2, -1, 0],
     [-1, 2, -1],
     [0, -1, 2]]
print(f"M = {M}")
print(f"det3(M)  = {det3(M)}      numpy: {np.linalg.det(np.array(M, float)):.6f}")

mine = np.array(inv3(M))
theirs = np.linalg.inv(np.array(M, float))
print(f"\n我的 inv3:\n{mine}")
print(f"\nnumpy inv:\n{theirs}")
print(f"\n最大差异 = {np.abs(mine - theirs).max():.2e}   "
      f"{'✅ 一致' if np.allclose(mine, theirs) else '❌ 不一致'}")
print(f"M @ M⁻¹ =\n{np.array(matmul(M, inv3(M)))}")

# 用分数看清楚精确解
print("\n精确解（分数形式）:")
frac = [[Fraction(int(round(x * det3(M))), det3(M)) for x in row] for row in inv3(M)]
for row in frac:
    print("  [" + ", ".join(str(f) for f in row) + "]")

# ============================================================
# 补充 1：逐元素乘 vs 矩阵乘 —— 结果完全不同
# ============================================================
print("\n" + "=" * 62)
print("补充：逐元素乘 vs 矩阵乘")
print("=" * 62)
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
print(f"A * B (逐元素) =\n{A * B}")
print(f"A @ B (矩阵乘) =\n{A @ B}")
print(f"B @ A (换序)   =\n{B @ A}      ← 矩阵乘不满足交换律！")
print(f"A * B == B * A ? {np.array_equal(A*B, B*A)}   ← 逐元素乘满足交换律")

# ============================================================
# 补充 2：转置为什么是反向传播的核心
# ============================================================
print("\n" + "=" * 62)
print("补充：转置在反向传播里的作用")
print("=" * 62)
W = np.random.default_rng(0).standard_normal((2, 3))
x = np.array([1.0, 0.5, -0.3])
y = W @ x
print(f"前向: W{W.shape} @ x{x.shape} -> y{y.shape}")
grad_y = np.array([1.0, -1.0])                 # 上游传来的梯度，形状同 y
grad_x = W.T @ grad_y                          # 反向：用 W 的转置
print(f"反向: W.T{W.T.shape} @ grad_y{grad_y.shape} -> grad_x{grad_x.shape}")
print(f"→ 前向把 3 维压成 2 维，反向把 2 维梯度摊回 3 维")
print(f"→ 形状必须对上，唯一的办法就是转置。这就是 backprop 里到处是 .T 的原因")

# ============================================================
# 补充 3：广播规则
# ============================================================
print("\n" + "=" * 62)
print("补充：广播 (broadcasting)")
print("=" * 62)
mat = np.array([[1, 2, 3], [4, 5, 6]])
bias_row = np.array([10, 20, 30])              # (3,)  沿行广播
bias_col = np.array([[100], [200]])            # (2,1) 沿列广播
print(f"mat {mat.shape} + bias_row {bias_row.shape} =\n{mat + bias_row}")
print(f"mat {mat.shape} + bias_col {bias_col.shape} =\n{mat + bias_col}")
try:
    mat + np.array([1, 2])
except ValueError as e:
    print(f"\nmat (2,3) + (2,) → 报错: {e}")
    print("→ 广播从【最右边】的维度开始对齐：3 vs 2 不匹配也不为 1，失败")
    print("→ 要沿列加，必须显式变成 (2,1)")
