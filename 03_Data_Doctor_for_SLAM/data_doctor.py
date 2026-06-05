"""
@file data_doctor.py
@brief SLAM 护航医生——光学退化拦截与极线约束监控

核心能力：
    1. 光学退化拦截：结合拉普拉斯算子与高斯统计，自动诊断卷帘快门下的
       运动模糊与极度曝光异常
    2. 极线约束监控：无需标定板，通过特征点到极线的垂直像素重投影误差，
       在线实时监控双目外参物理漂移（机械碰撞/热胀冷缩引起）

设计哲学：
    SLAM 系统的崩溃往往不是算法问题，而是数据问题。
    本工具在数据进入前端之前进行"体检"，拦截病态帧，监控硬件退化。
"""

import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field


# ════════════════════════════════════════════════
# 数据结构
# ════════════════════════════════════════════════

@dataclass
class DegradationReport:
    """单帧退化诊断报告"""
    frame_id: int
    blur_score: float           # 拉普拉斯方差（越低越模糊）
    is_blurry: bool             # 是否运动模糊
    exposure_score: float       # 曝光评分 [0,1]
    is_overexposed: bool        # 过曝
    is_underexposed: bool       # 欠曝
    is_healthy: bool            # 综合判定


@dataclass
class EpipolarDriftReport:
    """极线漂移监控报告"""
    frame_id: int
    mean_epipolar_error: float   # 平均极线重投影误差 [px]
    max_epipolar_error: float    # 最大极线重投影误差 [px]
    outlier_ratio: float         # 异常点比例
    drift_detected: bool         # 是否检测到外参漂移


# ════════════════════════════════════════════════
# Module 1: 光学退化拦截
# ════════════════════════════════════════════════

class OpticalDegradationDetector:
    """
    光学退化检测器

    方法：
        - 运动模糊：拉普拉斯方差法（Laplacian Variance）
          原理：清晰图像的高频能量远高于模糊图像
        - 曝光异常：基于直方图统计的高斯拟合
          过曝：亮区像素占比超过阈值
          欠曝：暗区像素占比超过阈值
    """

    def __init__(
        self,
        blur_threshold: float = 100.0,
        overexpose_ratio: float = 0.05,
        underexpose_ratio: float = 0.05,
        bright_pixel_value: int = 250,
        dark_pixel_value: int = 5
    ):
        self.blur_threshold = blur_threshold
        self.overexpose_ratio = overexpose_ratio
        self.underexpose_ratio = underexpose_ratio
        self.bright_pixel_value = bright_pixel_value
        self.dark_pixel_value = dark_pixel_value

    def detect_blur(self, gray: np.ndarray) -> Tuple[float, bool]:
        """
        拉普拉斯方差法检测运动模糊

        Returns:
            (laplacian_var, is_blurry)
        """
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        variance = lap.var()
        return variance, variance < self.blur_threshold

    def detect_exposure(self, gray: np.ndarray) -> Tuple[float, bool, bool]:
        """
        检测曝光异常

        Returns:
            (exposure_score, is_overexposed, is_underexposed)
        """
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
        total = gray.size

        bright_ratio = hist[self.bright_pixel_value:].sum() / total
        dark_ratio = hist[:self.dark_pixel_value].sum() / total

        # 曝光评分：基于直方图熵
        hist_norm = hist / (total + 1e-10)
        hist_norm = hist_norm[hist_norm > 0]
        entropy = -np.sum(hist_norm * np.log2(hist_norm))
        exposure_score = min(entropy / 8.0, 1.0)  # 归一化至 [0, 1]

        is_over = bright_ratio > self.overexpose_ratio
        is_under = dark_ratio > self.underexpose_ratio

        return exposure_score, is_over, is_under

    def diagnose(self, frame: np.ndarray, frame_id: int = 0) -> DegradationReport:
        """
        对单帧进行完整诊断
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame

        blur_score, is_blurry = self.detect_blur(gray)
        exposure_score, is_over, is_under = self.detect_exposure(gray)

        return DegradationReport(
            frame_id=frame_id,
            blur_score=blur_score,
            is_blurry=is_blurry,
            exposure_score=exposure_score,
            is_overexposed=is_over,
            is_underexposed=is_under,
            is_healthy=not (is_blurry or is_over or is_under)
        )


# ════════════════════════════════════════════════
# Module 2: 极线约束监控
# ════════════════════════════════════════════════

class EpipolarDriftMonitor:
    """
    双目外参漂移在线监控器

    原理：
        对于标定好的双目系统，特征点到对极线的距离应接近零。
        当外参因机械碰撞或热胀冷缩发生漂移时，该距离显著增大。
        无需标定板，利用场景纹理即可在线监控。

    数学：
        极线方程：l = F · x₁  （F 为基础矩阵）
        点到极线距离：d = |x₂^T · l| / √(l[0]² + l[1]²)
    """

    def __init__(
        self,
        F: np.ndarray,
        drift_threshold: float = 2.0,    # 平均极线误差阈值 [px]
        outlier_threshold: float = 5.0,  # 单点异常阈值 [px]
        outlier_ratio_threshold: float = 0.15
    ):
        self.F = F
        self.drift_threshold = drift_threshold
        self.outlier_threshold = outlier_threshold
        self.outlier_ratio_threshold = outlier_ratio_threshold
        self._history: List[EpipolarDriftReport] = []

    @staticmethod
    def point_to_epipolar_line_distance(
        pts2: np.ndarray,
        epilines: np.ndarray
    ) -> np.ndarray:
        """
        计算特征点到极线的垂直距离

        Args:
            pts2:     (N, 2) 右图特征点
            epilines: (N, 3) 极线 [a, b, c] 满足 ax + by + c = 0

        Returns:
            distances: (N,) 像素距离
        """
        # d = |a·x + b·y + c| / √(a² + b²)
        numerator = np.abs(
            epilines[:, 0] * pts2[:, 0] +
            epilines[:, 1] * pts2[:, 1] +
            epilines[:, 2]
        )
        denominator = np.sqrt(epilines[:, 0] ** 2 + epilines[:, 1] ** 2 + 1e-10)
        return numerator / denominator

    def check(
        self,
        pts1: np.ndarray,
        pts2: np.ndarray,
        frame_id: int = 0
    ) -> EpipolarDriftReport:
        """
        检测当前帧的极线约束偏差

        Args:
            pts1: (N, 2) 左图特征点
            pts2: (N, 2) 右图特征点
            frame_id: 帧序号
        """
        # 计算极线
        pts1_h = np.column_stack([pts1, np.ones(len(pts1))]).astype(np.float64)
        epilines = (self.F @ pts1_h.T).T  # (N, 3)

        # 计算距离
        distances = self.point_to_epipolar_line_distance(pts2, epilines)

        mean_err = np.mean(distances)
        max_err = np.max(distances)
        outlier_ratio = np.mean(distances > self.outlier_threshold)

        report = EpipolarDriftReport(
            frame_id=frame_id,
            mean_epipolar_error=mean_err,
            max_epipolar_error=max_err,
            outlier_ratio=outlier_ratio,
            drift_detected=(mean_err > self.drift_threshold or
                          outlier_ratio > self.outlier_ratio_threshold)
        )
        self._history.append(report)
        return report

    @property
    def history(self) -> List[EpipolarDriftReport]:
        return self._history


# ════════════════════════════════════════════════
# 便捷入口：批量诊断
# ════════════════════════════════════════════════

def batch_diagnose(
    image_dir: str,
    output_csv: str = "diagnosis_report.csv"
) -> List[DegradationReport]:
    """
    对目录下所有图像进行批量退化诊断
    """
    detector = OpticalDegradationDetector()
    reports = []

    img_paths = sorted(
        list(Path(image_dir).glob("*.png")) +
        list(Path(image_dir).glob("*.jpg"))
    )

    for i, p in enumerate(img_paths):
        img = cv2.imread(str(p))
        if img is None:
            continue
        report = detector.diagnose(img, frame_id=i)
        reports.append(report)
        status = "✓" if report.is_healthy else "✗"
        print(f"  [{status}] Frame {i:04d}: blur={report.blur_score:.1f} "
              f"exp={report.exposure_score:.3f}")

    # 导出 CSV
    if reports:
        with open(output_csv, 'w') as f:
            f.write("frame_id,blur_score,is_blurry,exposure_score,"
                    "is_overexposed,is_underexposed,is_healthy\n")
            for r in reports:
                f.write(f"{r.frame_id},{r.blur_score:.2f},{r.is_blurry},"
                        f"{r.exposure_score:.4f},{r.is_overexposed},"
                        f"{r.is_underexposed},{r.is_healthy}\n")
        print(f"\n[INFO] Diagnosis report saved to: {output_csv}")

    return reports


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SLAM Data Doctor")
    sub = parser.add_subparsers(dest="command")

    # 退化诊断子命令
    diag = sub.add_parser("diagnose", help="光学退化拦截")
    diag.add_argument("--image_dir", type=str, required=True)
    diag.add_argument("--output", type=str, default="diagnosis_report.csv")

    # 极线监控子命令
    epi = sub.add_parser("epipolar", help="极线约束监控")
    epi.add_argument("--left_dir", type=str, required=True)
    epi.add_argument("--right_dir", type=str, required=True)
    epi.add_argument("--F_file", type=str, required=True, help="基础矩阵 .npy")

    args = parser.parse_args()

    if args.command == "diagnose":
        batch_diagnose(args.image_dir, args.output)
    elif args.command == "epipolar":
        F = np.load(args.F_file)
        monitor = EpipolarDriftMonitor(F)
        # 简化示例：逐帧提取 ORB 特征并监控
        left_paths = sorted(Path(args.left_dir).glob("*.png"))
        right_paths = sorted(Path(args.right_dir).glob("*.png"))
        detector = cv2.ORB_create(nfeatures=2000)
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        for i, (lp, rp) in enumerate(zip(left_paths, right_paths)):
            l_img = cv2.imread(str(lp), cv2.IMREAD_GRAYSCALE)
            r_img = cv2.imread(str(rp), cv2.IMREAD_GRAYSCALE)
            kp1, des1 = detector.detectAndCompute(l_img, None)
            kp2, des2 = detector.detectAndCompute(r_img, None)
            matches = matcher.match(des1, des2)
            pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
            pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
            report = monitor.check(pts1, pts2, frame_id=i)
            status = "⚠ DRIFT" if report.drift_detected else "✓ OK"
            print(f"  Frame {i:04d}: {status}  "
                  f"mean_err={report.mean_epipolar_error:.2f}px  "
                  f"outlier={report.outlier_ratio:.2%}")
    else:
        parser.print_help()
