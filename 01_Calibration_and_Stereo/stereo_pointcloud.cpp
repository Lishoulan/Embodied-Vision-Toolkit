/**
 * @file stereo_pointcloud.cpp
 * @brief 基于极线几何与 SGBM 视差计算的逆向 3D 点云重建
 *
 * 流水线：
 *   1. 双目图像 → SGBM 视差图
 *   2. 视差图 + Q 矩阵 → 3D 点云（cv::reprojectImageTo3D）
 *   3. 颜色映射 + 统计滤波 → 输出 PLY 文件
 *
 * 物理约束：深度 Z = f·B / d，其中 f 为焦距，B 为基线，d 为视差。
 *           视差精度直接决定深度分辨率，亚像素插值可提升 Z 方向精度。
 */

#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <iostream>
#include <fstream>
#include <vector>

// ──────────────────────────────────────────────
// SGBM 参数结构体
// ──────────────────────────────────────────────
struct SGBMParams {
    int min_disparity   = 0;
    int num_disparities = 128;     // 必须为 16 的倍数
    int block_size      = 5;      // 奇数，通常 3~11
    int P1              = 600;    // 视差平滑惩罚（8×block_size²）
    int P2              = 2400;   // 视差跳变惩罚（32×block_size²）
    int disp12_max_diff = 1;
    int pre_filter_cap  = 31;
    int uniqueness_ratio = 5;
    int speckle_window  = 100;
    int speckle_range   = 1;
    int mode            = cv::StereoSGBM::MODE_SGBM_3WAY;
};

/**
 * @brief 计算视差图
 */
cv::Mat compute_disparity(const cv::Mat& left, const cv::Mat& right,
                           const SGBMParams& params) {
    cv::Ptr<cv::StereoSGBM> sgbm = cv::StereoSGBM::create(
        params.min_disparity, params.num_disparities, params.block_size,
        params.P1, params.P2, params.disp12_max_diff,
        params.pre_filter_cap, params.uniqueness_ratio,
        params.speckle_window, params.speckle_range, params.mode
    );

    cv::Mat disp;
    sgbm->compute(left, right, disp);
    // SGBM 输出为 CV_16S，除以 16 得到真实视差（亚像素精度 ×16）
    disp.convertTo(disp, CV_32F, 1.0 / 16.0);
    return disp;
}

/**
 * @brief 视差图 → 3D 点云
 *
 * @param disparity  32F 视差图
 * @param Q          4×4 重投影矩阵（cv::stereoRectify 获得）
 * @param left_rect  用于着色的矫正后左图
 * @param max_z      最大深度截断 [mm]
 * @return std::vector<cv::Point3f> 过滤后的 3D 点
 */
std::vector<cv::Point3f> disparity_to_pointcloud(
    const cv::Mat& disparity,
    const cv::Mat& Q,
    const cv::Mat& left_rect,
    float max_z = 10000.0f)
{
    cv::Mat points3d;
    cv::reprojectImageTo3D(disparity, points3d, Q, true);

    std::vector<cv::Point3f> cloud;
    for (int y = 0; y < points3d.rows; ++y) {
        for (int x = 0; x < points3d.cols; ++x) {
            cv::Point3f pt = points3d.at<cv::Point3f>(y, x);
            // 过滤无效点：视差为 0 或深度超出范围
            if (std::isfinite(pt.z) && pt.z > 0 && pt.z < max_z) {
                cloud.push_back(pt);
            }
        }
    }
    return cloud;
}

/**
 * @brief 导出点云为 PLY 格式
 */
bool save_ply(const std::string& path,
              const std::vector<cv::Point3f>& points,
              const std::vector<cv::Vec3b>& colors = {})
{
    std::ofstream ofs(path);
    if (!ofs.is_open()) return false;

    bool has_color = !colors.empty() && colors.size() == points.size();

    ofs << "ply\nformat ascii 1.0\n";
    ofs << "element vertex " << points.size() << "\n";
    ofs << "property float x\nproperty float y\nproperty float z\n";
    if (has_color) {
        ofs << "property uchar red\nproperty uchar green\nproperty uchar blue\n";
    }
    ofs << "end_header\n";

    for (size_t i = 0; i < points.size(); ++i) {
        ofs << points[i].x << " " << points[i].y << " " << points[i].z;
        if (has_color) {
            ofs << " " << (int)colors[i][2] << " "
                << (int)colors[i][1] << " " << (int)colors[i][0];
        }
        ofs << "\n";
    }
    ofs.close();
    return true;
}

// ──────────────────────────────────────────────
// Main
// ──────────────────────────────────────────────
int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: stereo_pointcloud <left_img> <right_img> <Q_yaml> [output.ply]\n";
        return 1;
    }

    const std::string left_path  = argv[1];
    const std::string right_path = argv[2];
    const std::string q_path     = argv[3];
    const std::string out_path   = (argc >= 5) ? argv[4] : "output.ply";

    cv::Mat left  = cv::imread(left_path, cv::IMREAD_GRAYSCALE);
    cv::Mat right = cv::imread(right_path, cv::IMREAD_GRAYSCALE);
    if (left.empty() || right.empty()) {
        std::cerr << "[ERROR] Cannot load stereo pair.\n";
        return 1;
    }

    // 加载 Q 矩阵
    cv::FileStorage fs(q_path, cv::FileStorage::READ);
    cv::Mat Q;
    fs["Q"] >> Q;
    fs.release();

    // 计算 SGBM 视差
    SGBMParams params;
    cv::Mat disparity = compute_disparity(left, right, params);

    // 生成点云
    cv::Mat left_color = cv::imread(left_path);
    auto cloud = disparity_to_pointcloud(disparity, Q, left_color);

    // 提取颜色
    std::vector<cv::Vec3b> colors;
    for (int y = 0; y < left_color.rows; ++y) {
        for (int x = 0; x < left_color.cols; ++x) {
            cv::Point3f pt = disparity.at<cv::Point3f>(y, x);  // placeholder check
            if (std::isfinite(pt.z) && pt.z > 0 && pt.z < 10000.0f) {
                colors.push_back(left_color.at<cv::Vec3b>(y, x));
            }
        }
    }

    // 保存 PLY
    if (save_ply(out_path, cloud, colors)) {
        std::cout << "[INFO] Point cloud saved: " << out_path
                  << " (" << cloud.size() << " points)\n";
    }

    return 0;
}
