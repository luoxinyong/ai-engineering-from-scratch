"""投影与均值的关系 —— 把边界划清楚"""
import numpy as np
np.set_printoptions(precision=4, suppress=True)

a = np.array([1., 2., 3.])
ones = np.array([1., 1., 1.])

print("=" * 64)
print("① [1,1,1] 是单位向量吗？")
print("=" * 64)
print(f"|[1,1,1]| = {np.linalg.norm(ones):.6f}  = √3   → 不是单位向量")
u = ones / np.linalg.norm(ones)
print(f"归一化后 û = {u}   |û| = {np.linalg.norm(u):.6f}")

print("\n" + "=" * 64)
print("② 用 [1,1,1] 和用 û 投影，结果一样吗？")
print("=" * 64)
proj_ones = (a @ ones) / (ones @ ones) * ones
coef_ones = (a @ ones) / (ones @ ones)
proj_u = (a @ u) * u                      # 单位向量时 b·b = 1，系数就是 a·û
coef_u = a @ u

print(f"用 [1,1,1]:  系数 = {coef_ones:.4f}      投影向量 = {proj_ones}")
print(f"用 û      :  系数 = {coef_u:.4f}      投影向量 = {proj_u}")
print(f"投影向量相同吗? {np.allclose(proj_ones, proj_u)}   ← 相同！方向一样，投影就一样")
print(f"系数相同吗?     {np.isclose(coef_ones, coef_u)}   ← 不同！")
print(f"\na 的均值 = {a.mean():.4f}")
print(f"  用 [1,1,1] 的系数 = {coef_ones:.4f}  = 均值        ✅")
print(f"  用 û 的系数      = {coef_u:.4f}  = 均值×√3 = {a.mean()*np.sqrt(3):.4f}  ❌不是均值")
print("→ 【结论】'系数=均值' 依赖于用【未归一化的全1向量】，不是'投影到单位向量'的性质")

print("\n" + "=" * 64)
print("③ 投影到别的方向呢？还是均值吗？")
print("=" * 64)
for name, b in [("[1,1,1] 全1", ones),
                ("[1,0,0] x轴", np.array([1., 0., 0.])),
                ("[1,2,3] 自己", a.copy()),
                ("[1,-1,0]  ", np.array([1., -1., 0.])),
                ("[2,2,2] 全1的倍数", np.array([2., 2., 2.]))]:
    coef = (a @ b) / (b @ b)
    proj = coef * b
    print(f"  投影到 {name:18s} 系数={coef:+8.4f}  投影向量={proj}")
print(f"\na 的均值 = {a.mean()}")
print("→ 只有【全1方向】(以及它的任意倍数) 给出均值。[2,2,2] 也行，因为方向相同")
print("→ 投影到 x 轴给的是【第一个分量】，投影到自己给的是它自己 —— 都不是均值")

print("\n" + "=" * 64)
print("④ 为什么偏偏是全1方向给出均值？—— 均值就是最小二乘解")
print("=" * 64)
print("问题：找一个常数 c，让 [c,c,c] 离 a 最近（平方距离最小）")
cs = np.linspace(0, 4, 9)
print(f"  {'c':>6}  {'||a - c*[1,1,1]||²':>20}")
for c in cs:
    d = np.sum((a - c * ones) ** 2)
    mark = "  ← 最小" if np.isclose(c, a.mean()) else ""
    print(f"  {c:6.2f}  {d:20.4f}{mark}")
print(f"\n解析解: c* = argmin Σ(aᵢ-c)² , 令导数=0 → -2Σ(aᵢ-c)=0 → c = Σaᵢ/n = 均值")
print("→ 【均值 = 用一个常数逼近数据的最小二乘解】")
print("→ 【最小二乘 = 投影】(上一课讲过)")
print("→ 所以: 投影到常数方向(全1) 必然给出均值。这不是巧合，是同一件事")

print("\n" + "=" * 64)
print("⑤ 勾股分解：总能量 = 均值部分 + 波动部分")
print("=" * 64)
proj = proj_ones
resid = a - proj
print(f"a        = {a}          |a|²     = {a@a:.4f}")
print(f"proj     = {proj}          |proj|²  = {proj@proj:.4f}   ← 均值贡献")
print(f"resid    = {resid}          |resid|² = {resid@resid:.4f}   ← 波动贡献")
print(f"                                    合计    = {proj@proj + resid@resid:.4f}")
print(f"proj · resid = {proj@resid:.6f}   ← 垂直，所以勾股定理成立")
print(f"\n|resid|² / n = {(resid@resid)/len(a):.4f}   = 方差 {a.var():.4f}   ✅")
print("→ 【方差就是残差长度的平方除以 n】= 数据在'垂直于全1方向'的那个子空间里的长度")

print("\n" + "=" * 64)
print("⑥ 推广到矩阵：给数据集做中心化")
print("=" * 64)
rng = np.random.default_rng(0)
X = rng.integers(0, 10, size=(5, 3)).astype(float)   # 5 个样本, 3 个特征
print(f"X (5样本×3特征) =\n{X}")
n = X.shape[0]
one = np.ones((n, 1))
P = one @ one.T / n                    # 投影到全1方向的投影矩阵 (n×n)
print(f"\n投影矩阵 P = 1·1ᵀ/n  ({P.shape}) —— 每个元素都是 1/n = {1/n}")
print(f"P @ X (每列的均值，复制 n 遍) =\n{P @ X}")
print(f"列均值 = {X.mean(axis=0)}   ✅ 一致")
Xc = (np.eye(n) - P) @ X               # 中心化 = 投影到全1的正交补
print(f"\n(I - P) @ X  = 中心化后的数据 =\n{Xc}")
print(f"X - X.mean(0) =\n{X - X.mean(axis=0)}")
print(f"两者相同? {np.allclose(Xc, X - X.mean(axis=0))}   ✅")
print(f"\nP 是投影矩阵的证据: P@P == P ? {np.allclose(P@P, P)}  (幂等)")
print(f"                    P == P.T ? {np.allclose(P, P.T)}  (对称)")
print("→ 中心化不是'减个数'，是【把数据投影到全1方向的正交补子空间】")
print("→ (I-P) 也是投影矩阵，投到 n-1 维的子空间 —— 这就是统计里'自由度 n-1'的来源")
