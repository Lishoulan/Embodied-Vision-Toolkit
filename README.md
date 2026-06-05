<div align="center">

# 🔬 Embodied-Vision-Toolkit

**具身智能视觉感知与底层数据诊断工具箱**

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![C++](https://img.shields.io/badge/C%2B%2B-17-00599C?logo=c%2B%2B&logoColor=white)](https://isocpp.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org/)
[![Eigen](https://img.shields.io/badge/Eigen-3.4-00599C?logo=eigen&logoColor=white)](https://eigen.tuxfamily.org/)
[![SLAM](https://img.shields.io/badge/Domain-SLAM-FF6F00?logo=robot&logoColor=white)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

---

> **"从光子到位姿，从像素到形貌——用第一性物理填平纯软件算法与底层硬件之间的鸿沟。"**

本工具箱从**光学物理第一性原理**出发，针对机器人 SLAM 与具身感知中的三大底层痛点——**标定退化、触觉盲区、数据病态**——提供可编译、可复现的算法级解决方案。每个模块均包含完整的数学推导与工程实现，而非黑盒调用。

---

## 📐 核心模块

### 🎯 Module 1：相机底层几何与立体视觉

> **从镜头畸变物理模型到 3D 空间重建的完整链路**

| 文件 | 语言 | 核心能力 |
|------|------|----------|
| [`fisheye_undistort.cpp`](01_Calibration_and_Stereo/fisheye_undistort.cpp) | C++ | 基于等距模型（Equidistant / Kannala-Brandt）的超大视场角鱼眼去畸变 |
| [`baseline_eigen.cpp`](01_Calibration_and_Stereo/baseline_eigen.cpp) | C++ | 利用 Eigen3 解算 $4 \times 4$ 齐次变换矩阵，SVD 投影至 $SO(3)$ 后求取物理基线 |
| [`stereo_pointcloud.cpp`](01_Calibration_and_Stereo/stereo_pointcloud.cpp) | C++ | 基于极线几何与 SGBM 视差计算的逆向 3D 点云重建，输出 PLY |

<details>
<summary>📖 物理原理简述</summary>

- **鱼眼去畸变**：等距模型假设入射角 $\theta$ 与像高 $r$ 满足 $r = f \cdot \theta$，适用于 FOV > 180° 的超大视场角镜头。通过反投影至单位球面再正投影至针孔平面，实现畸变校正。
- **基线解算**：双目外参矩阵 $T = [R \| t]$ 中，$\|t\|_2$ 即物理基线长度。对 $R$ 做 SVD 最近正交投影至 $SO(3)$，消除数值漂移。
- **深度重建**：深度 $Z = \frac{f \cdot B}{d}$，其中 $f$ 为焦距、$B$ 为基线、$d$ 为视差。SGBM 输出 $\times 16$ 亚像素精度视差图。

</details>

---

### ✋ Module 2：微观触觉 3D 感知

> **从图像像素到亚毫米级 3D 接触面形貌的"无中生有"重建**

| 文件 | 语言 | 核心能力 |
|------|------|----------|
| [`photometric_stereo.py`](02_Visuo_Tactile_Perception/photometric_stereo.py) | Python | 光度立体视觉——解构朗伯反射模型，提取表面 2D 法向量场 |
| [`fft_poisson_integration.py`](02_Visuo_Tactile_Perception/fft_poisson_integration.py) | Python | 基于频域 FFT 的二维泊松积分，法向量梯度场 → 亚毫米级高度场 |

<details>
<summary>📖 物理原理简述</summary>

- **朗伯反射模型**：$I = \rho \cdot \mathbf{n}^T \cdot \mathbf{L}$，其中 $I$ 为观测亮度、$\rho$ 为反照率、$\mathbf{n}$ 为表面法向量、$\mathbf{L}$ 为光源方向。至少 3 个不同光照方向构成超定方程组，最小二乘解算法向量。
- **泊松积分**：已知梯度场 $(p, q) = (\frac{\partial z}{\partial x}, \frac{\partial z}{\partial y})$，求解高度场 $z$ 满足泊松方程 $\nabla^2 z = \frac{\partial p}{\partial x} + \frac{\partial q}{\partial y}$。频域解法（Frankot-Chellappa）：
$$Z(u,v) = \frac{-j \cdot u \cdot P(u,v) - j \cdot v \cdot Q(u,v)}{u^2 + v^2}$$

</details>

---

### 🩺 Module 3：SLAM 护航医生与 3DGS 预处理

> **数据进入算法之前，先给数据做体检**

| 文件 | 语言 | 核心能力 |
|------|------|----------|
| [`data_doctor.py`](03_Data_Doctor_for_SLAM/data_doctor.py) | Python | 光学退化拦截 + 极线约束漂移监控 |
| [`3dgs_data_pipeline.py`](03_Data_Doctor_for_SLAM/3dgs_data_pipeline.py) | Python | NeRF / 3DGS 自动化视点数据集清洗管线 |

#### 🩺 光学退化拦截

| 退化类型 | 检测方法 | 物理依据 |
|----------|----------|----------|
| 运动模糊 | 拉普拉斯方差法 | 清晰图像高频能量远高于模糊图像 |
| 过曝 | 直方图亮区统计 | 像素饱和导致信息不可逆丢失 |
| 欠曝 | 直方图暗区统计 | 暗电流噪声淹没信号 |

#### 📏 极线约束监控

无需标定板，利用场景纹理在线监控双目外参物理漂移：

$$d = \frac{|\mathbf{x}_2^T \cdot \mathbf{F} \cdot \mathbf{x}_1|}{\sqrt{l_0^2 + l_1^2}}$$

当机械碰撞或热胀冷缩导致外参漂移时，特征点到极线的距离 $d$ 显著增大，系统自动告警。

#### 🧹 3DGS 数据清洗管线

```
原始图像 → [退化过滤] → [去畸变] → [分辨率统一] → [COLMAP 位姿] → transforms.json
```

为 NeRF / 3D Gaussian Splatting 提供高质量、防糊抗畸变的标准化视点数据集。

---

## 🗂️ 项目结构

```
Embodied-Vision-Toolkit/
├── 01_Calibration_and_Stereo/          # 相机底层几何与立体视觉
│   ├── fisheye_undistort.cpp           # 鱼眼等距模型去畸变
│   ├── baseline_eigen.cpp              # Eigen3 齐次变换基线解算
│   └── stereo_pointcloud.cpp           # SGBM 视差 → 3D 点云
├── 02_Visuo_Tactile_Perception/        # 微观触觉 3D 感知
│   ├── photometric_stereo.py           # 光度立体视觉法向量提取
│   └── fft_poisson_integration.py      # FFT 泊松积分高度场重建
├── 03_Data_Doctor_for_SLAM/            # SLAM 护航医生与 3DGS 预处理
│   ├── data_doctor.py                  # 光学退化拦截 + 极线漂移监控
│   └── 3dgs_data_pipeline.py          # 3DGS/NeRF 数据清洗管线
├── README.md
└── requirements.txt
```

---

## 🚀 Quick Start

### 环境依赖

| 依赖 | 版本 | 用途 |
|------|------|------|
| Python | ≥ 3.9 | Module 2 & 3 |
| C++ Compiler | ≥ C++17 | Module 1 |
| CMake | ≥ 3.16 | 编译 C++ 模块 |
| OpenCV | ≥ 4.5 | 图像处理核心 |
| Eigen3 | ≥ 3.4 | 线性代数运算 |
| COLMAP | ≥ 3.8 | SfM 位姿估计（可选） |

### Python 模块

```bash
# 安装依赖
pip install -r requirements.txt

# Module 2: 光度立体视觉
python 02_Visuo_Tactile_Perception/photometric_stereo.py \
    --image_dir ./data/tactile_images \
    --light_file ./data/light_directions.txt \
    --output normals.png

# Module 2: 泊松积分
python 02_Visuo_Tactile_Perception/fft_poisson_integration.py \
    --normals normals.npy \
    --pixel_size 0.01 \
    --output height_map.png

# Module 3: 光学退化诊断
python 03_Data_Doctor_for_SLAM/data_doctor.py diagnose \
    --image_dir ./data/slam_frames \
    --output diagnosis_report.csv

# Module 3: 极线漂移监控
python 03_Data_Doctor_for_SLAM/data_doctor.py epipolar \
    --left_dir ./data/left \
    --right_dir ./data/right \
    --F_file ./data/F_matrix.npy

# Module 3: 3DGS 数据清洗
python 03_Data_Doctor_for_SLAM/3dgs_data_pipeline.py \
    --image_dir ./data/raw_images \
    --output_dir ./cleaned_dataset
```

### C++ 模块编译

```bash
# 编译全部 C++ 模块
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# 运行鱼眼去畸变
./fisheye_undistort calib.yaml fisheye_image.png output.png

# 运行基线解算
./baseline_eigen

# 运行立体匹配点云
./stereo_pointcloud left.png right.png Q.yaml output.ply
```

<details>
<summary>🔧 CMakeLists.txt 参考</summary>

```cmake
cmake_minimum_required(VERSION 3.16)
project(EmbodiedVisionToolkit LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

# fisheye_undistort
add_executable(fisheye_undistort 01_Calibration_and_Stereo/fisheye_undistort.cpp)
target_link_libraries(fisheye_undistort ${OpenCV_LIBS})

# baseline_eigen
add_executable(baseline_eigen 01_Calibration_and_Stereo/baseline_eigen.cpp)
target_link_libraries(baseline_eigen Eigen3::Eigen)

# stereo_pointcloud
add_executable(stereo_pointcloud 01_Calibration_and_Stereo/stereo_pointcloud.cpp)
target_link_libraries(stereo_pointcloud ${OpenCV_LIBS})
```

</details>

---

## 👤 About

**作者**：[Your Name Here]

光学硕士，现从事机器人数据与算法开发。具备**光学硬件底层**与**算法架构顶层**的跨界能力——既能在频域推演光子传播，也能在 C++ 里手搓齐次变换。坚信：不理解物理的算法工程师，终将被物理淘汰。

> 📧 Email: [your.email@example.com]
> 🏠 GitHub: [https://github.com/your-username]

---

<div align="center">

**从第一性原理出发，让每一行代码都有物理依据。**

</div>
