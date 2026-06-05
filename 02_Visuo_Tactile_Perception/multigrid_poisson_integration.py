"""
@file multigrid_poisson_integration.py
@brief 基于多重网格法（Multigrid）的二维泊松积分——与 FFT 方法对比

数学原理：
    泊松方程：∇²z = f，其中 f = ∂p/∂x + ∂q/∂y

    多重网格法核心思想：
        1. 在细网格上做几次松弛迭代（如 Gauss-Seidel），消除高频误差
        2. 将残差限制（restriction）到粗网格
        3. 在粗网格上求解修正量（递归或直接求解）
        4. 将修正量插值（prolongation）回细网格并校正
        5. 再做几次松弛迭代（后光滑）

    收敛性：O(N)，远优于 Gauss-Seidel 的 O(N²) 和 FFT 的 O(N log N)，
    但在常数因子上 FFT 更优。本实现用于对比验证。

    参考：Briggs, Henson, McCormick, "A Multigrid Tutorial", SIAM 2000
"""


import numpy as np
from fft_poisson_integration import fft_poisson_integration

# ════════════════════════════════════════════════
# 基础迭代算子
# ════════════════════════════════════════════════

def gauss_seidel(
    z: np.ndarray,
    f: np.ndarray,
    h: float = 1.0,
    n_iter: int = 50
) -> np.ndarray:
    """
    Gauss-Seidel 松弛迭代求解 ∇²z = f

    离散拉普拉斯：(z[i+1,j] + z[i-1,j] + z[i,j+1] + z[i,j-1] - 4z[i,j]) / h² = f[i,j]
    迭代更新：z[i,j] = (z[i+1,j] + z[i-1,j] + z[i,j+1] + z[i,j-1] - h²·f[i,j]) / 4

    Args:
        z: 当前高度场估计
        f: 泊松方程右端项
        h: 网格步长
        n_iter: 迭代次数
    """
    z = z.copy()
    h2 = h * h

    for _ in range(n_iter):
        # 向量化红黑 Gauss-Seidel（简化为 Jacobi 风格，实际工程中应分色迭代）
        z_new = np.zeros_like(z)
        z_new[1:-1, 1:-1] = (
            z[2:, 1:-1] + z[:-2, 1:-1] +
            z[1:-1, 2:] + z[1:-1, :-2] -
            h2 * f[1:-1, 1:-1]
        ) / 4.0
        z[1:-1, 1:-1] = z_new[1:-1, 1:-1]

    return z


def compute_residual(
    z: np.ndarray,
    f: np.ndarray,
    h: float = 1.0
) -> np.ndarray:
    """
    计算残差 r = f - ∇²z
    """
    h2 = h * h
    laplacian_z = np.zeros_like(z)
    laplacian_z[1:-1, 1:-1] = (
        z[2:, 1:-1] + z[:-2, 1:-1] +
        z[1:-1, 2:] + z[1:-1, :-2] -
        4 * z[1:-1, 1:-1]
    ) / h2
    return f - laplacian_z


# ════════════════════════════════════════════════
# 限制与插值算子
# ════════════════════════════════════════════════

def restrict(fine: np.ndarray) -> np.ndarray:
    """
    限制算子（Restriction）：细网格 → 粗网格
    使用全权重（full-weighting）方案

    coarse[i,j] = 1/16 * (
        fine[2i-1,2j-1] + fine[2i-1,2j+1] + fine[2i+1,2j-1] + fine[2i+1,2j+1] +
        2*(fine[2i-1,2j] + fine[2i+1,2j] + fine[2i,2j-1] + fine[2i,2j+1]) +
        4*fine[2i,2j]
    )
    """
    H, W = fine.shape
    h_coarse, w_coarse = (H - 1) // 2 + 1, (W - 1) // 2 + 1
    coarse = np.zeros((h_coarse, w_coarse))

    for i in range(1, h_coarse - 1):
        for j in range(1, w_coarse - 1):
            fi, fj = 2 * i, 2 * j
            coarse[i, j] = (
                fine[fi - 1, fj - 1] + fine[fi - 1, fj + 1] +
                fine[fi + 1, fj - 1] + fine[fi + 1, fj + 1] +
                2 * (fine[fi - 1, fj] + fine[fi + 1, fj] +
                     fine[fi, fj - 1] + fine[fi, fj + 1]) +
                4 * fine[fi, fj]
            ) / 16.0

    return coarse


def prolongate(coarse: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    """
    插值算子（Prolongation）：粗网格 → 细网格
    使用双线性插值（scipy.ndimage.zoom）
    """
    from scipy.ndimage import zoom

    h_c, w_c = coarse.shape
    scale_h = target_shape[0] / h_c
    scale_w = target_shape[1] / w_c

    fine = zoom(coarse, (scale_h, scale_w), order=1)
    # 裁剪或填充至精确目标尺寸
    result = np.zeros(target_shape)
    h_min = min(fine.shape[0], target_shape[0])
    w_min = min(fine.shape[1], target_shape[1])
    result[:h_min, :w_min] = fine[:h_min, :w_min]
    return result


# ════════════════════════════════════════════════
# V-Cycle 多重网格求解器
# ════════════════════════════════════════════════

def v_cycle(
    z: np.ndarray,
    f: np.ndarray,
    h: float = 1.0,
    n_pre: int = 3,
    n_post: int = 3,
    min_size: int = 8,
    level: int = 0
) -> np.ndarray:
    """
    V-Cycle 多重网格求解器

    Args:
        z: 当前高度场估计
        f: 泊松方程右端项
        h: 网格步长
        n_pre: 前光滑迭代次数
        n_post: 后光滑迭代次数
        min_size: 最粗网格尺寸（直接求解）
        level: 当前层级（用于调试）
    """
    # 前光滑
    z = gauss_seidel(z, f, h, n_iter=n_pre)

    # 计算残差
    r = compute_residual(z, f, h)

    # 判断是否到达最粗层
    if min(z.shape) <= min_size:
        # 最粗层：直接求解（多迭代几次）
        e = gauss_seidel(np.zeros_like(r), r, h, n_iter=50)
    else:
        # 限制残差到粗网格
        r_coarse = restrict(r)
        e_coarse = np.zeros_like(r_coarse)

        # 递归 V-Cycle
        e_coarse = v_cycle(e_coarse, r_coarse, h * 2, n_pre, n_post, min_size, level + 1)

        # 插值修正量回细网格
        e = prolongate(e_coarse, z.shape)

    # 校正
    z = z + e

    # 后光滑
    z = gauss_seidel(z, f, h, n_iter=n_post)

    return z


# ════════════════════════════════════════════════
# 完整多重网格求解
# ════════════════════════════════════════════════

def multigrid_poisson_integration(
    p: np.ndarray,
    q: np.ndarray,
    h: float = 1.0,
    n_vcycles: int = 5,
    tol: float = 1e-6
) -> np.ndarray:
    """
    基于多重网格法的二维泊松积分

    Args:
        p: (H, W) x 方向梯度 ∂z/∂x
        q: (H, W) y 方向梯度 ∂z/∂y
        h: 网格步长
        n_vcycles: V-Cycle 迭代次数
        tol: 收敛阈值（残差 L2 范数）

    Returns:
        z: (H, W) 重建的高度场
    """
    # 构造右端项 f = ∂p/∂x + ∂q/∂y
    f = np.zeros_like(p)
    f[1:-1, 1:-1] = (
        (p[1:-1, 2:] - p[1:-1, :-2]) / (2 * h) +
        (q[2:, 1:-1] - q[:-2, 1:-1]) / (2 * h)
    )

    # 初始猜测
    z = np.zeros_like(p)

    for i in range(n_vcycles):
        z = v_cycle(z, f, h)
        residual = compute_residual(z, f, h)
        res_norm = np.linalg.norm(residual)

        if res_norm < tol:
            print(f"  [V-Cycle {i+1}] Converged: residual = {res_norm:.2e}")
            break
        else:
            print(f"  [V-Cycle {i+1}] Residual = {res_norm:.2e}")

    return z


# ════════════════════════════════════════════════
# 对比验证：FFT vs Multigrid
# ════════════════════════════════════════════════

def compare_methods(
    p: np.ndarray,
    q: np.ndarray,
    h: float = 1.0
) -> dict:
    """
    对比 FFT 与 Multigrid 两种泊松积分方法

    Returns:
        dict: 包含两种方法的结果、耗时、误差对比
    """
    import time

    print("═══════════════════════════════════════════")
    print("  Poisson Integration: FFT vs Multigrid")
    print("═══════════════════════════════════════════")

    # FFT 方法
    t0 = time.perf_counter()
    z_fft = fft_poisson_integration(p, q)
    t_fft = time.perf_counter() - t0

    # Multigrid 方法
    t0 = time.perf_counter()
    z_mg = multigrid_poisson_integration(p, q, h)
    t_mg = time.perf_counter() - t0

    # 对比
    # FFT 结果作为参考（解析解），计算 Multigrid 的相对误差
    diff = z_mg - z_fft
    # 去除均值偏移（泊松解差一个常数）
    diff = diff - np.mean(diff)
    rel_error = np.linalg.norm(diff) / (np.linalg.norm(z_fft) + 1e-10)

    print(f"\n  FFT time:       {t_fft*1000:.2f} ms")
    print(f"  Multigrid time: {t_mg*1000:.2f} ms")
    print(f"  Relative error: {rel_error:.6f}")
    print("═══════════════════════════════════════════")

    return {
        "z_fft": z_fft,
        "z_mg": z_mg,
        "t_fft_ms": t_fft * 1000,
        "t_mg_ms": t_mg * 1000,
        "relative_error": rel_error
    }


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(
        description="Multigrid Poisson Integration (vs FFT)"
    )
    parser.add_argument(
        "--normals", type=str,
        help="法向量 .npy 文件路径 (H, W, 3)"
    )
    parser.add_argument(
        "--synthetic", action="store_true",
        help="使用合成数据测试"
    )
    args = parser.parse_args()

    if args.synthetic or args.normals is None:
        # 合成测试：已知高度场 → 法向量 → 重建 → 对比
        print("[INFO] Running synthetic test...")
        H, W = 128, 128
        x = np.linspace(-2, 2, W)
        y = np.linspace(-2, 2, H)
        X, Y = np.meshgrid(x, y)

        # 真实高度场：高斯峰
        z_true = np.exp(-(X**2 + Y**2) / 2.0)

        # 计算梯度
        p = np.gradient(z_true, axis=1)  # ∂z/∂x
        q = np.gradient(z_true, axis=0)  # ∂z/∂y

        # 对比
        results = compare_methods(p, q)

        # 可视化
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        axes[0].imshow(z_true, cmap='jet')
        axes[0].set_title("Ground Truth")
        axes[1].imshow(results["z_fft"], cmap='jet')
        axes[1].set_title(f"FFT ({results['t_fft_ms']:.1f}ms)")
        axes[2].imshow(results["z_mg"], cmap='jet')
        axes[2].set_title(f"Multigrid ({results['t_mg_ms']:.1f}ms)")
        plt.tight_layout()
        plt.savefig("multigrid_comparison.png", dpi=150)
        plt.show()
    else:
        from fft_poisson_integration import normals_to_gradients
        normals = np.load(args.normals)
        p, q = normals_to_gradients(normals)
        results = compare_methods(p, q)
