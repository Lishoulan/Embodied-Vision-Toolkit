/**
 * @file baseline_eigen.cpp
 * @brief 利用 Eigen3 解算 4×4 齐次变换矩阵，求取双目相机物理基线
 *
 * 物理背景：
 *   双目外参矩阵 T = [R | t]，其中 t 的二范数 ||t|| 即为物理基线长度。
 *   在机器人坐标系下，基线向量 b = R^T · t 给出基线在世界系下的方向。
 *   本程序从标定结果中提取 T，利用 Eigen 的 SVD/QR 分解进行鲁棒解算。
 */

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <iostream>
#include <iomanip>
#include <cmath>

// ──────────────────────────────────────────────
// 从 4×4 齐次变换矩阵提取基线信息
// ──────────────────────────────────────────────
struct BaselineResult {
    double length;                     // 基线长度 [mm]
    Eigen::Vector3d direction;        // 基线方向（归一化）
    Eigen::Vector3d translation;       // 平移向量 [mm]
    Eigen::Matrix3d rotation;          // 旋转矩阵
};

/**
 * @brief 从齐次变换矩阵解算基线
 *
 * @param T_lr  左相机到右相机的 4×4 齐次变换矩阵
 * @return BaselineResult 包含基线长度、方向、平移与旋转
 */
BaselineResult compute_baseline(const Eigen::Matrix4d& T_lr) {
    BaselineResult result;

    // 提取旋转与平移
    result.rotation = T_lr.block<3, 3>(0, 0);
    result.translation = T_lr.block<3, 1>(0, 3);

    // 对旋转矩阵做最近正交化（SVD 投影至 SO(3)），消除数值漂移
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(result.rotation,
        Eigen::ComputeFullU | Eigen::ComputeFullV);
    result.rotation = svd.matrixU() * svd.matrixV().transpose();

    // 确保行列式为 +1（排除反射）
    if (result.rotation.determinant() < 0) {
        Eigen::Matrix3d V = svd.matrixV();
        V.col(2) *= -1;
        result.rotation = svd.matrixU() * V.transpose();
    }

    // 基线长度 = ||t||
    result.length = result.translation.norm();

    // 基线方向 = t / ||t||
    result.direction = result.translation.normalized();

    return result;
}

// ──────────────────────────────────────────────
// 辅助：从 R|t 构造 4×4 齐次矩阵
// ──────────────────────────────────────────────
Eigen::Matrix4d make_transform(const Eigen::Matrix3d& R, const Eigen::Vector3d& t) {
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    T.block<3, 3>(0, 0) = R;
    T.block<3, 1>(0, 3) = t;
    return T;
}

int main() {
    // ── 示例：典型水平双目外参 ──
    // 基线 ~120mm，沿 X 轴偏移，含微小旋转
    Eigen::Matrix3d R;
    R = Eigen::AngleAxisd(0.002, Eigen::Vector3d::UnitY()).toRotationMatrix()
      * Eigen::AngleAxisd(-0.001, Eigen::Vector3d::UnitX()).toRotationMatrix();

    Eigen::Vector3d t(119.85, 0.32, -0.18);  // [mm]

    Eigen::Matrix4d T_lr = make_transform(R, t);

    // 解算基线
    BaselineResult baseline = compute_baseline(T_lr);

    // 输出
    std::cout << "═══════════════════════════════════════════\n";
    std::cout << "  Stereo Baseline Analysis (Eigen3)\n";
    std::cout << "═══════════════════════════════════════════\n";
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "  Baseline length : " << baseline.length << " mm\n";
    std::cout << "  Baseline direction: ["
              << baseline.direction.transpose() << "]\n";
    std::cout << "  Translation (t)  : ["
              << baseline.translation.transpose() << "] mm\n";
    std::cout << "  Rotation det(R)  : " << baseline.rotation.determinant() << "\n";
    std::cout << "═══════════════════════════════════════════\n";

    return 0;
}
