/**
 * @file fisheye_undistort.cpp
 * @brief 基于等距模型(Equidistant/Kannala-Brandt)的超大视场角鱼眼去畸变
 *
 * 核心思路：
 *   1. 利用 KB 模型将像素坐标反投影至单位球面（单位方向向量）
 *   2. 将球面方向向量经旋转矩阵映射至目标针孔相机坐标系
 *   3. 正投影至去畸变后的理想针孔图像平面
 *
 * 物理约束：等距模型假设入射角 θ 与像高 r 满足 r = f·θ，
 *           适用于 FOV > 180° 的超大视场角镜头。
 */

#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <iostream>
#include <string>

// ──────────────────────────────────────────────
// 鱼眼内参结构体：对应 OpenCV fisheye::calibrate 输出
// ──────────────────────────────────────────────
struct FisheyeIntrinsics {
    cv::Mat K;           // 3×3 相机矩阵
    cv::Mat D;           // 1×4 畸变系数 [k1, k2, k3, k4]
    cv::Size img_size;   // 原始图像尺寸
};

/**
 * @brief 对单张鱼眼图像执行去畸变，输出为指定目标内参的针孔图像
 *
 * @param src         输入鱼眼原图
 * @param intrinsics  鱼眼内参（K, D, img_size）
 * @param new_K       目标针孔相机矩阵（若为空则自动估算）
 * @param output_size 输出图像尺寸
 * @param alpha       自由缩放参数 [0,1]，0=裁剪至全有效像素，1=保留所有源像素
 * @return cv::Mat    去畸变后的图像
 */
cv::Mat fisheye_undistort(
    const cv::Mat& src,
    const FisheyeIntrinsics& intrinsics,
    const cv::Mat& new_K = cv::Mat(),
    const cv::Size& output_size = cv::Size(0, 0),
    double alpha = 0.0)
{
    cv::Mat map1, map2;
    cv::Mat P = new_K.empty() ? cv::Mat::eye(3, 3, CV_64F) : new_K.clone();

    cv::fisheye::initUndistortRectifyMap(
        intrinsics.K, intrinsics.D, cv::Mat::eye(3, 3, CV_64F),
        P, output_size.area() > 0 ? output_size : intrinsics.img_size,
        CV_16SC2, map1, map2
    );

    cv::Mat dst;
    cv::remap(src, dst, map1, map2, cv::INTER_LINEAR, cv::BORDER_CONSTANT);
    return dst;
}

// ──────────────────────────────────────────────
// Main：从 YAML 加载内参并批量去畸变
// ──────────────────────────────────────────────
int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: fisheye_undistort <calib_yaml> <image_path> [output_path]\n";
        return 1;
    }

    const std::string calib_path = argv[1];
    const std::string img_path   = argv[2];
    const std::string out_path   = (argc >= 4) ? argv[3] : "undistorted.png";

    // 加载标定参数（示例：从 FileStorage 读取）
    cv::FileStorage fs(calib_path, cv::FileStorage::READ);
    FisheyeIntrinsics intrinsics;
    fs["K"] >> intrinsics.K;
    fs["D"] >> intrinsics.D;
    intrinsics.img_size = cv::Size(
        static_cast<int>(fs["image_width"]),
        static_cast<int>(fs["image_height"])
    );
    fs.release();

    cv::Mat src = cv::imread(img_path);
    if (src.empty()) {
        std::cerr << "[ERROR] Cannot load image: " << img_path << "\n";
        return 1;
    }

    cv::Mat dst = fisheye_undistort(src, intrinsics);
    cv::imwrite(out_path, dst);
    std::cout << "[INFO] Undistorted image saved to: " << out_path << "\n";

    return 0;
}
