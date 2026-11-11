"""Phase 01 / Lesson 03 练习 1、2、3"""
import numpy as np

np.set_printoptions(precision=4, suppress=True)

# ============================================================
# 练习 1：对单位正方形做旋转/缩放/切变，验证旋转保距离
# ============================================================
print("=" * 62)
print("练习 1：单位正方形变换 + 旋转保距离验证")
print("=" * 62)

square = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)  # 四角
names = ["原点", "右", "右上", "上"]

def rot(t):   return np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
def scale(sx, sy): return np.array([[sx, 0], [0, sy]])
def shear(k):  return np.array([[1, k], [0, 1]])

def corner_dists(pts):
    """相邻角之间的边长 + 对角线"""
    d = []
    for i in range(4):
        d.append(np.linalg.norm(pts[i] - pts[(i+1) % 4]))
    return np.array(d)

orig_d = corner_dists(square)
print(f"原正方形四条边长: {orig_d}")

for label, M in [("旋转30°", rot(np.radians(30))),
                 ("缩放(2,0.5)", scale(2, 0.5)),
                 ("切变k=1", shear(1))]:
    out = (M @ square.T).T
    d = corner_dists(out)
    same = np.allclose(d, orig_d)
    print(f"\n{label}: det={np.linalg.det(M):.2f}")
    for n, p in zip(names, out):
        print(f"    {n:>4}: [{p[0]:6.3f}, {p[1]:6.3f}]")
    print(f"    四条边长: {d}   保持原边长? {same}")
print("\n→ 只有旋转保持所有边长不变(等距变换);缩放/切变会改变边长")

# ============================================================
# 练习 2：手算 [[4,2],[1,3]] 的特征值,再对拍
# ============================================================
print("\n" + "=" * 62)
print("练习 2：手算特征值 [[4,2],[1,3]]")
print("=" * 62)

a, b, c, d = 4, 2, 1, 3
trace = a + d          # 7
det = a*d - b*c        # 12 - 2 = 10
print(f"矩阵 [[4,2],[1,3]]")
print(f"特征方程: λ² − (a+d)λ + (ad−bc) = 0")
print(f"        = λ² − {trace}λ + {det} = 0")
disc = trace**2 - 4*det
print(f"判别式 = {trace}² − 4×{det} = {trace**2} − {4*det} = {disc}")
l1 = (trace + disc**0.5) / 2
l2 = (trace - disc**0.5) / 2
print(f"λ = ({trace} ± √{disc}) / 2 = ({trace} ± {disc**0.5:.4f}) / 2")
print(f"手算特征值: λ1 = {l1:.4f},  λ2 = {l2:.4f}")

# 验算：和=trace, 积=det
print(f"\n验算: λ1+λ2 = {l1+l2:.4f} (应=trace {trace}),  λ1·λ2 = {l1*l2:.4f} (应=det {det})")

# numpy 对拍
A = np.array([[4, 2], [1, 3]], dtype=float)
vals, vecs = np.linalg.eig(A)
print(f"\nnumpy 特征值: {np.sort(vals)[::-1]}")
print(f"一致? {np.allclose(sorted([l1, l2]), sorted(vals.real))}")

# 验证 A@v = λ·v
print(f"\n验证 A@v = λ·v:")
for i in range(2):
    v, lam = vecs[:, i], vals[i]
    print(f"  λ={lam:.4f}: A@v = {A@v},  λ·v = {lam*v}   {'✓' if np.allclose(A@v, lam*v) else '✗'}")

# ============================================================
# 练习 3：三个变换复合,作用在圆上8个点;验证 det 相乘
# ============================================================
print("\n" + "=" * 62)
print("练习 3：复合变换(旋转30°→缩放(1.5,0.8)→切变kx=0.3)")
print("=" * 62)

R = rot(np.radians(30))
S = scale(1.5, 0.8)
Sh = shear(0.3)

# 复合:先转,再缩放,再切变 → 矩阵顺序反过来 Sh @ S @ R
M = Sh @ S @ R
print(f"复合矩阵 M = Sh @ S @ R =\n{M}")

# 圆上8个点
angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
circle = np.array([[np.cos(t), np.sin(t)] for t in angles])
transformed = (M @ circle.T).T

print(f"\n{'角度':>5}  {'原坐标':>18}   {'变换后':>18}")
for t, p, q in zip(np.degrees(angles), circle, transformed):
    print(f"{t:5.0f}°  [{p[0]:6.3f}, {p[1]:6.3f}]   [{q[0]:6.3f}, {q[1]:6.3f}]")

# 验证 det(复合) = det 连乘
dR, dS, dSh = np.linalg.det(R), np.linalg.det(S), np.linalg.det(Sh)
dM = np.linalg.det(M)
print(f"\ndet(R)={dR:.4f}  det(S)={dS:.4f}  det(Sh)={dSh:.4f}")
print(f"三者相乘 = {dR*dS*dSh:.4f}")
print(f"det(复合M) = {dM:.4f}")
print(f"相等? {np.isclose(dM, dR*dS*dSh)}   ← det(AB)=det(A)det(B)")
print(f"\n→ 圆被这个复合变换拉成了椭圆;面积缩放倍数 = {abs(dM):.4f} = 1.5×0.8×1×1")
