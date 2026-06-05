"""
@file fft_poisson_integration.py
@brief 基于频域 FFT 的二维泊松积分——从法向量梯度场重建亚毫米级 3D 表面形貌

数学原理：
    已知表面梯度场 (p, q) = (∂z/∂x, ∂z/∂y)，求解高度场 z 满足：
        ∇²z = ∂p/∂x + ∂q/∂y   （泊松方程）

    频域解法（Frankot-Chellappa）：
        Z(u,v) = (-j·u·P(u,v) - j·v·Q(u,v)) / (u² + v²)
    其中 P, Q 为梯度场的傅里叶变换，分母为零处（DC）置零。

    本实现用于具身智能触觉传感器（如 GelSight）的亚毫米级 3D 重建，
    将光度立体视觉输出的法向量转化为可度量的接触面高度场。
"""

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


def normals_to_gradients(normals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    从表面法向量 (nx, ny, nz) 恢复梯度场 (p, q)

    朗伯面约束：p = -nx/nz, q = -ny/nz

    Args:
        normals: (H, W, 3) 归一化法向量

    Returns:
        p: (H, W) ∂z/∂x 梯度
        q: (H, W) ∂z/∂y 梯度
    """
    nx = normals[:, :, 0]
    ny = normals[:, :, 1]
    nz = normals[:, :, 2].copy()
    nz[nz == 0] = 1e-10  # 避免除零

    p = -nx / nz
    q = -ny / nz
    return p, q


def fft_poisson_integration(
    p: np.ndarray,
    q: np.ndarray
) -> np.ndarray:
    """
    基于频域 FFT 的二维泊松积分（Frankot-Chellappa 方法）

    Args:
        p: (H, W) x 方向梯度 ∂z/∂x
        q: (H, W) y 方向梯度 ∂z/∂y

    Returns:
        z: (H, W) 重建的高度场
    """
    H, W = p.shape

    # 对梯度场做二维 FFT
    P = np.fft.fft2(p)
    Q = np.fft.fft2(q)

    # 频域坐标
    u = np.fft.fftfreq(W).reshape(1, -1) * 2 * np.pi  # 归一化角频率
    v = np.fft.fftfreq(H).reshape(-1, 1) * 2 * np.pi

    # 频域泊松解
    # Z = (-j*u*P - j*v*Q) / (u² + v²)
    denom = u ** 2 + v ** 2
    denom[0, 0] = 1.0  # DC 分量置零（避免除零）

    Z = (-1j * u * P - 1j * v * Q) / denom
    Z[0, 0] = 0  # 高度场均值为零

    # 逆 FFT 取实部
    z = np.real(np.fft.ifft2(Z))

    return z


def reconstruct_height_from_normals(
    normals: np.ndarray,
    pixel_size_mm: float = 0.01
) -> np.ndarray:
    """
    端到端：法向量 → 梯度场 → 泊松积分 → 高度场

    Args:
        normals:     (H, W, 3) 归一化法向量
        pixel_size_mm: 像素物理尺寸 [mm]，用于将梯度转换为物理高度

    Returns:
        height_mm: (H, W) 重建的物理高度场 [mm]
    """
    p, q = normals_to_gradients(normals)
    z = fft_poisson_integration(p, q)

    # 梯度是无量纲的（像素空间），乘以像素尺寸转换为物理高度
    height_mm = z * pixel_size_mm

    return height_mm


def visualize_height_map(
    height: np.ndarray,
    save_path: Optional[str] = None
) -> None:
    """
    可视化重建高度场：2D 伪彩色 + 3D 曲面
    """
    fig = plt.figure(figsize=(14, 5))

    # 2D 伪彩色
    ax1 = fig.add_subplot(131)
    im = ax1.imshow(height, cmap='jet')
    ax1.set_title("Height Map (Top View)")
    plt.colorbar(im, ax=ax1, label='Height [mm]')

    # 3D 曲面
    ax2 = fig.add_subplot(132, projection='3d')
    H, W = height.shape
    X, Y = np.meshgrid(np.arange(W), np.arange(H))
    step = max(1, min(H, W) // 80)  # 降采样加速渲染
    ax2.plot_surface(
        X[::step, ::step], Y[::step, ::step], height[::step, ::step],
        cmap='jet', alpha=0.9, rstride=1, cstride=1
    )
    ax2.set_title("3D Surface Reconstruction")
    ax2.set_xlabel("X [px]")
    ax2.set_ylabel("Y [px]")
    ax2.set_zlabel("Z [mm]")

    # 截面轮廓
    ax3 = fig.add_subplot(133)
    mid = height.shape[0] // 2
    ax3.plot(height[mid, :], 'b-', linewidth=1.5)
    ax3.set_title(f"Cross-section (row={mid})")
    ax3.set_xlabel("X [px]")
    ax3.set_ylabel("Height [mm]")
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[INFO] Height map visualization saved to: {save_path}")
    plt.show()


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="FFT Poisson Integration: Normals → Height Map"
    )
    parser.add_argument(
        "--normals", type=str, required=True,
        help="法向量 .npy 文件路径 (H, W, 3)"
    )
    parser.add_argument(
        "--pixel_size", type=float, default=0.01,
        help="像素物理尺寸 [mm]，默认 0.01mm (GelSight 典型值)"
    )
    parser.add_argument(
        "--output", type=str, default="height_map.png",
        help="高度场可视化输出路径"
    )
    args = parser.parse_args()

    normals = np.load(args.normals)
    height = reconstruct_height_from_normals(normals, pixel_size_mm=args.pixel_size)

    np.save("height_map.npy", height)
    print(f"[INFO] Height map shape: {height.shape}")
    print(f"[INFO] Height range: [{height.min():.4f}, {height.max():.4f}] mm")

    visualize_height_map(height, save_path=args.output)
