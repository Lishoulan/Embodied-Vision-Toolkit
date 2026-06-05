/**
 * @file double_sphere_undistort.cpp
 * @brief 基于 Double Sphere（双球面）模型的鱼眼去畸变
 *
 * 论文：V. Usenko, G. Demmel, D. Cremers,
 *       "The Double Sphere Camera Model", 3DV 2018
 *
 * 模型优势（相比 KB 等距模型）：
 *   - 仅需 4 个参数 (fx, fy, cx, cy) + 2 个形状参数 (ξ, α)
 *   - 解析闭式反投影（单位方向向量），无需迭代
 *   - 覆盖 FOV > 195°，且在边缘区域畸变建模精度优于 KB
 *
 * 投影公式：
 *   d = √(x² + y² + z²)
 *   dz = ξ·d + z
 *   rz² = x² + y² + z²·(2ξ·d + z + 2ξ²)
 *   r = √(x² + y²)
 *   valid = (α > 0 && rz² > α·rz²)
 *   u = fx · (x / (α·dz + (1-α)·√rz²)) + cx
 *   v = fy · (y / (α·dz + (1-α)·√rz²)) + cy
 *
 * 反投影公式（闭式）：
 *   mx = (u - cx) / fx
 *   my = (v - cy) / fy
 *   r² = mx² + my²
 *   mz = (1 - α²·r²) / (α·√(1 + (1-α²)·r²) + (1-α))
 *        — 当 α²·r² ≤ 1 时有效
 *   归一化方向向量: n = (mx, my, mz) / ||(mx, my, mz)||
 */

#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <iostream>
#include <string>
#include <cmath>

// ──────────────────────────────────────────────
// Double Sphere 内参
// ──────────────────────────────────────────────
struct DoubleSphereParams {
    double fx, fy;    // 焦距 [px]
    double cx, cy;    // 主点 [px]
    double xi;        // 球面间距参数（控制两球心距离）
    double alpha;     // 混合参数（0=针孔，1=单位球投影）

    // 从 4×1 参数向量构造
    static DoubleSphereParams fromVector(const Eigen::Vector4d& p) {
        return {p(0), p(1), p(2), p(3), p(4), p(5)};
    }
};

// ──────────────────────────────────────────────
// 反投影：像素 → 单位方向向量（闭式解）
// ──────────────────────────────────────────────

/**
 * @brief 将像素坐标反投影至单位方向向量
 *
 * @param u,v       像素坐标
 * @param params    Double Sphere 参数
 * @param direction  输出：3D 单位方向向量
 * @return true     反投影有效（在视场范围内）
 */
bool unproject(
    double u, double v,
    const DoubleSphereParams& params,
    Eigen::Vector3d& direction)
{
    const double mx = (u - params.cx) / params.fx;
    const double my = (v - params.cy) / params.fy;
    const double r2 = mx * mx + my * my;

    // 有效性检查：α²·r² ≤ 1
    if (params.alpha * params.alpha * r2 > 1.0) {
        return false;
    }

    // 闭式 mz 计算
    const double alpha2 = params.alpha * params.alpha;
    const double sqrt_term = std::sqrt(1.0 + (1.0 - alpha2) * r2);
    const double mz = (1.0 - alpha2 * r2) /
                      (params.alpha * sqrt_term + (1.0 - params.alpha));

    // 归一化
    direction = Eigen::Vector3d(mx, my, mz).normalized();
    return true;
}

// ──────────────────────────────────────────────
// 投影：3D 点 → 像素
// ──────────────────────────────────────────────

/**
 * @brief 将 3D 点投影至像素坐标
 *
 * @param point     3D 点（相机坐标系）
 * @param params    Double Sphere 参数
 * @param u,v       输出：像素坐标
 * @return true     投影有效
 */
bool project(
    const Eigen::Vector3d& point,
    const DoubleSphereParams& params,
    double& u, double& v)
{
    const double x = point.x(), y = point.y(), z = point.z();
    const double d = point.norm();  // √(x² + y² + z²)

    if (d < 1e-10) return false;

    const double dz = params.xi * d + z;
    const double r2 = x * x + y * y + z * z * (2.0 * params.xi * d + z + 2.0 * params.xi * params.xi);
    const double rz = std::sqrt(r2);

    // 有效性检查
    if (params.alpha > 0 && r2 <= params.alpha * params.alpha * r2) {
        return false;
    }

    const double denom = params.alpha * dz + (1.0 - params.alpha) * rz;
    if (std::abs(denom) < 1e-10) return false;

    u = params.fx * (x / denom) + params.cx;
    v = params.fy * (y / denom) + params.cy;
    return true;
}

// ──────────────────────────────────────────────
// 生成去畸变映射表（与 fisheye_undistort 对比的核心差异）
// ──────────────────────────────────────────────

/**
 * @brief 生成 Double Sphere 去畸变映射
 *
 * 核心思路：对目标针孔图像的每个像素，反投影至 3D 方向向量，
 *          再用 Double Sphere 正投影回源图像，得到映射关系。
 *          与 OpenCV fisheye::initUndistortRectifyMap 不同，
 *          这里完全从模型公式推导，无任何黑盒调用。
 *
 * @param src_size     源图像尺寸
 * @param ds_params    Double Sphere 参数
 * @param pinhole_K    目标针孔相机矩阵
 * @param output_size  输出图像尺寸
 * @param map1, map2   输出映射表
 */
void build_undistort_map(
    const cv::Size& src_size,
    const DoubleSphereParams& ds_params,
    const Eigen::Matrix3d& pinhole_K,
    const cv::Size& output_size,
    cv::Mat& map1, cv::Mat& map2)
{
    map1.create(output_size, CV_32FC1);
    map2.create(output_size, CV_32FC1);

    const double fx_p = pinhole_K(0, 0);
    const double fy_p = pinhole_K(1, 1);
    const double cx_p = pinhole_K(0, 2);
    const double cy_p = pinhole_K(1, 2);

    for (int v_out = 0; v_out < output_size.height; ++v_out) {
        for (int u_out = 0; u_out < output_size.width; ++u_out) {
            // 目标针孔图像像素 → 归一化平面坐标
            const double mx = (u_out - cx_p) / fx_p;
            const double my = (v_out - cy_p) / fy_p;

            // 归一化方向向量（针孔模型：方向 = (mx, my, 1)）
            Eigen::Vector3d dir(mx, my, 1.0);
            dir.normalize();

            // Double Sphere 正投影：方向 → 源图像像素
            double u_src, v_src;
            bool valid = project(dir, ds_params, u_src, v_src);

            if (valid && u_src >= 0 && u_src < src_size.width &&
                        v_src >= 0 && v_src < src_size.height) {
                map1.at<float>(v_out, u_out) = static_cast<float>(u_src);
                map2.at<float>(v_out, u_out) = static_cast<float>(v_src);
            } else {
                // 视场外区域：映射至 (-1, -1)，remap 时用 BORDER_CONSTANT 填黑
                map1.at<float>(v_out, u_out) = -1.0f;
                map2.at<float>(v_out, u_out) = -1.0f;
            }
        }
    }
}

// ──────────────────────────────────────────────
// Main
// ──────────────────────────────────────────────
int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: double_sphere_undistort <image> <params_yaml> <output>\n";
        std::cerr << "  params_yaml should contain: fx, fy, cx, cy, xi, alpha\n";
        return 1;
    }

    const std::string img_path    = argv[1];
    const std::string params_path = argv[2];
    const std::string out_path    = argv[3];

    // 加载图像
    cv::Mat src = cv::imread(img_path);
    if (src.empty()) {
        std::cerr << "[ERROR] Cannot load image: " << img_path << "\n";
        return 1;
    }

    // 加载 Double Sphere 参数
    cv::FileStorage fs(params_path, cv::FileStorage::READ);
    DoubleSphereParams params;
    params.fx    = fs["fx"];
    params.fy    = fs["fy"];
    params.cx    = fs["cx"];
    params.cy    = fs["cy"];
    params.xi    = fs["xi"];
    params.alpha = fs["alpha"];
    fs.release();

    std::cout << "═══════════════════════════════════════════\n";
    std::cout << "  Double Sphere Undistortion\n";
    std::cout << "═══════════════════════════════════════════\n";
    std::cout << "  fx=" << params.fx << " fy=" << params.fy << "\n";
    std::cout << "  cx=" << params.cx << " cy=" << params.cy << "\n";
    std::cout << "  xi=" << params.xi << " alpha=" << params.alpha << "\n";

    // 构造目标针孔相机矩阵（缩小焦距以容纳更大视场）
    const double scale = 0.6;  // 缩放因子
    Eigen::Matrix3d pinhole_K = Eigen::Matrix3d::Identity();
    pinhole_K(0, 0) = params.fx * scale;
    pinhole_K(1, 1) = params.fy * scale;
    pinhole_K(0, 2) = src.cols / 2.0;
    pinhole_K(1, 2) = src.rows / 2.0;

    // 生成映射表
    cv::Mat map1, map2;
    build_undistort_map(src.size(), params, pinhole_K, src.size(), map1, map2);

    // 应用映射
    cv::Mat dst;
    cv::remap(src, dst, map1, map2, cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

    cv::imwrite(out_path, dst);
    std::cout << "  Output saved to: " << out_path << "\n";
    std::cout << "═══════════════════════════════════════════\n";

    return 0;
}
