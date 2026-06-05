"""
@file photometric_stereo.py
@brief 光度立体视觉（Photometric Stereo）——从多方向光照图像中恢复表面法向量

物理原理：
    朗伯反射模型（Lambertian Reflectance）：
        I = ρ · n^T · L
    其中 I 为观测亮度，ρ 为反照率，n 为表面法向量，L 为光源方向。
    至少需要 3 个不同光照方向，构成线性方程组：
        I = L · n^T  →  n = (L^T L)^{-1} L^T · I

    本实现支持：
        - 最小二乘法（≥3 光源，鲁棒解算）
        - 法向量归一化与反照率分离
        - 可视化：法向量球面映射 / RGB 编码
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple, Optional


def load_images_and_lights(
    image_dir: str,
    light_file: str
) -> Tuple[np.ndarray, np.ndarray]:
    """
    加载多光照图像与对应光源方向矩阵

    Args:
        image_dir: 图像目录，按光照顺序命名 (e.g., light_0.png, light_1.png, ...)
        light_file: 光源方向文件，每行 [Lx, Ly, Lz]，对应每张图像

    Returns:
        images: (H, W, N) 图像堆叠，N 为光照数
        lights: (N, 3) 光源方向矩阵
    """
    # 加载光源方向
    lights = np.loadtxt(light_file)  # (N, 3)
    n_lights = lights.shape[0]

    # 归一化光源方向
    lights = lights / np.linalg.norm(lights, axis=1, keepdims=True)

    # 加载图像
    img_paths = sorted(Path(image_dir).glob("*.png"))
    if len(img_paths) < n_lights:
        img_paths = sorted(Path(image_dir).glob("*.jpg"))

    images = []
    for i in range(n_lights):
        img = cv2.imread(str(img_paths[i]), cv2.IMREAD_GRAYSCALE).astype(np.float64) / 255.0
        images.append(img)

    images = np.stack(images, axis=-1)  # (H, W, N)
    return images, lights


def photometric_stereo(
    images: np.ndarray,
    lights: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    光度立体视觉核心解算

    Args:
        images: (H, W, N) 多光照图像
        lights: (N, 3) 光源方向矩阵

    Returns:
        normals: (H, W, 3) 表面法向量
        albedo:  (H, W)    反照率图
    """
    H, W, N = images.shape

    # 将图像重塑为 (H*W, N)，每行一个像素的观测向量
    I = images.reshape(-1, N)  # (H*W, N)

    # 最小二乘解：n = (L^T L)^{-1} L^T · I^T
    L_pinv = np.linalg.pinv(lights)  # (3, N)

    # 批量解算：(3, N) @ (N, H*W) → (3, H*W)
    normals_flat = L_pinv @ I.T  # (3, H*W)

    # 反照率 = 法向量的模
    albedo_flat = np.linalg.norm(normals_flat, axis=0)  # (H*W,)

    # 归一化法向量
    normals_flat = normals_flat / (albedo_flat[np.newaxis, :] + 1e-10)

    normals = normals_flat.T.reshape(H, W, 3)
    albedo = albedo_flat.reshape(H, W)

    return normals, albedo


def visualize_normals(
    normals: np.ndarray,
    save_path: Optional[str] = None
) -> None:
    """
    可视化法向量：将法向量映射至 [0, 255] RGB 空间

    Convention: n ∈ [-1, 1] → pixel ∈ [0, 255]
    """
    normal_vis = ((normals + 1.0) / 2.0 * 255).astype(np.uint8)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(normal_vis)
    axes[0].set_title("Surface Normals (RGB Encoding)")
    axes[0].axis("off")

    # 球面映射可视化
    H, W = normals.shape[:2]
    y, x = np.mgrid[0:H, 0:W]
    axes[1].quiver(
        x[::10, ::10], y[::10, ::10],
        normals[::10, ::10, 0], normals[::10, ::10, 1],
        scale=30, color='b', alpha=0.7
    )
    axes[1].set_title("Normal Vector Field")
    axes[1].invert_yaxis()
    axes[1].set_aspect('equal')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[INFO] Normal visualization saved to: {save_path}")
    plt.show()


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Photometric Stereo Reconstruction")
    parser.add_argument("--image_dir", type=str, required=True, help="多光照图像目录")
    parser.add_argument("--light_file", type=str, required=True, help="光源方向文件 (.txt)")
    parser.add_argument("--output", type=str, default="normals.png", help="法向量可视化输出路径")
    args = parser.parse_args()

    images, lights = load_images_and_lights(args.image_dir, args.light_file)
    normals, albedo = photometric_stereo(images, lights)
    visualize_normals(normals, save_path=args.output)

    # 保存法向量与反照率为 .npy
    np.save("normals.npy", normals)
    np.save("albedo.npy", albedo)
    print(f"[INFO] Normals shape: {normals.shape}, Albedo shape: {albedo.shape}")
