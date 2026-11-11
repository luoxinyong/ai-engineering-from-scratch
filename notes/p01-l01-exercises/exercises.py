"""Phase 01 / Lesson 01 练习 2、3、5、6"""
import numpy as np

np.set_printoptions(precision=4, suppress=True)

# ============================================================
# 练习 2：2D 缩放矩阵，x 翻倍、y 三倍
# ============================================================
print("=" * 60)
print("练习 2：缩放矩阵")
print("=" * 60)

S = np.array([[2, 0],
              [0, 3]])
v = np.array([1, 1])
print(f"S =\n{S}")
print(f"S @ [1,1] = {S @ v}")

# 对角矩阵只沿坐标轴缩放，不旋转。看它对单位正方形做了什么：
square = np.array([[0, 1, 1, 0], [0, 0, 1, 1]])   # 四个角
print(f"\n单位正方形四角:\n{square}")
print(f"变换后:\n{S @ square}")
print("→ 1×1 的正方形变成 2×3 的矩形，面积 1 → 6")
print(f"→ 面积缩放倍数 = |det(S)| = {abs(np.linalg.det(S)):.0f}")

# ============================================================
# 练习 3：5 个 50 维"词向量"，找最相似的两个
# ============================================================
print("\n" + "=" * 60)
print("练习 3：余弦相似度找最相似的一对（RAG 的最小版本）")
print("=" * 60)

rng = np.random.default_rng(42)
D = 50
names = ["w1", "w2", "w3", "w4", "w5"]
V = rng.standard_normal((5, D))

def cosine_matrix(M):
    norms = np.linalg.norm(M, axis=1, keepdims=True)
    U = M / norms                      # 逐行归一化
    return U @ U.T                     # 归一化后，点积就是余弦

C = cosine_matrix(V)
print("余弦相似度矩阵（对角线是自己跟自己=1）:")
print("      " + "  ".join(f"{n:>7}" for n in names))
for i, n in enumerate(names):
    print(f"{n:>4}  " + "  ".join(f"{C[i,j]:7.4f}" for j in range(5)))

iu = np.triu_indices(5, k=1)           # 只看上三角，排除对角线和重复
best = np.argmax(C[iu])
i, j = iu[0][best], iu[1][best]
print(f"\n最相似的一对: {names[i]} 和 {names[j]}, 余弦 = {C[i,j]:.4f}, "
      f"夹角 = {np.degrees(np.arccos(np.clip(C[i,j],-1,1))):.1f}°")

# —— 高维的诡异之处 ——
print("\n--- 但注意这些数字有多小 ---")
offdiag = C[iu]
print(f"5 个随机向量两两余弦: 均值 {offdiag.mean():+.4f}, 绝对值最大 {np.abs(offdiag).max():.4f}")
for d in [2, 10, 50, 768, 4096]:
    A = rng.standard_normal((300, d))
    Cd = cosine_matrix(A)
    od = Cd[np.triu_indices(300, k=1)]
    print(f"  {d:5d} 维: |余弦| 均值 {np.abs(od).mean():.4f}   "
          f"理论值 1/√d = {1/np.sqrt(d):.4f}")
print("→ 维度越高，随机向量越趋近于互相垂直（余弦→0）")
print("→ 所以真实的词向量里，余弦 0.7 已经是「非常相似」了")

# ============================================================
# 练习 5：秩为 2 的 3×3 矩阵
# ============================================================
print("\n" + "=" * 60)
print("练习 5：造一个秩=2 的 3×3 矩阵")
print("=" * 60)

A = np.array([[1, 0, 2],
              [0, 1, 1],
              [1, 1, 3]], dtype=float)    # 第3列 = 2*第1列 + 1*第2列
print(f"A =\n{A}")
print(f"rank(A) = {np.linalg.matrix_rank(A)}   (3×3 但秩只有 2 → 秩亏)")
print(f"det(A)  = {np.linalg.det(A):.6f}      (秩亏 ⇒ 行列式=0 ⇒ 不可逆)")
print(f"第3列 - (2*第1列 + 第2列) = {A[:,2] - (2*A[:,0] + A[:,1])}")

n = np.cross(A[:, 0], A[:, 1])            # 前两列的法向量
print(f"\n列张成的几何对象：过原点的一个【平面】")
print(f"  平面法向量 n = col1 × col2 = {n}")
print(f"  验证三列都在平面上（与 n 点积应为 0）:")
for k in range(3):
    print(f"    col{k+1} · n = {np.dot(A[:,k], n):.6f}")
print(f"\n后果：A 把整个 3D 空间【压扁】到这个平面上，一个维度的信息永久丢失")
print(f"  例：[0,0,1] 和 [-2,-1,1] 都被映射到 {A @ np.array([0.,0.,1.])}")
print(f"     A@[0,0,1]     = {A @ np.array([0.,0.,1.])}")
print(f"     A@[-2,-1,1]+? 检查零空间：")
ns = np.array([-2., -1., 1.])
print(f"     A @ {ns} = {A @ ns}   ← 非零向量被映射到 0（零空间）")

# ============================================================
# 练习 6：把 [1,2,3] 投影到 [1,1,1]
# ============================================================
print("\n" + "=" * 60)
print("练习 6：投影 [1,2,3] → [1,1,1]")
print("=" * 60)

a = np.array([1., 2., 3.])
b = np.array([1., 1., 1.])
scalar = np.dot(a, b) / np.dot(b, b)
proj = scalar * b
resid = a - proj

print(f"a = {a}")
print(f"b = {b}")
print(f"a·b = {np.dot(a,b):.0f},  b·b = {np.dot(b,b):.0f},  系数 = {scalar:.4f}")
print(f"proj_b(a) = {proj}")
print(f"residual  = {resid}")
print(f"residual · b = {np.dot(resid, b):.6f}   ← 垂直")

print(f"\n【几何含义】")
print(f"  a 各分量的均值 = {a.mean():.4f}   ← 跟上面的系数完全一样！")
print(f"  投影到全1方向 = 提取「均值」")
print(f"  残差 {resid} = a 减掉均值 = 【中心化】后的数据")
print(f"  残差各分量之和 = {resid.sum():.6f}  (中心化后必然为0)")
print(f"\n→ 这就是 PCA / 标准化第一步「减均值」在几何上做的事：")
print(f"  把数据向量中平行于 [1,1,...,1] 的分量投影掉")
