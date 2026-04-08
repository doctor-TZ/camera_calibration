
#include <iostream>
#include <string>
#include <vector>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>

/**
 * @brief 辅助函数：计算归一化单应矩阵 H
 * 使用各向同性归一化提高 SVD 数值稳定性
 */
Eigen::Matrix3d computeHomography(const std::vector<cv::Point3f>& objPoints, 
                                 const std::vector<cv::Point2f>& imgPoints) {
    //                                
    int n = objPoints.size();
    
    //对像素点的坐标进行归一化处理
    auto get_norm_mat = [&](const std::vector<Eigen::Vector2d>& pts) {
        Eigen::Vector2d mean(0, 0);
        for (const auto& p : pts) mean += p;
        mean /= n;
        double mean_dist = 0;
        for (const auto& p : pts) mean_dist += (p - mean).norm();
        double scale = std::sqrt(2.0) / (mean_dist / n);
        Eigen::Matrix3d T = Eigen::Matrix3d::Identity();
        T(0, 0) = scale; T(1, 1) = scale;
        T(0, 2) = -scale * mean.x(); T(1, 2) = -scale * mean.y();
        return T;
    };

    std::vector<Eigen::Vector2d> pts_obj, pts_img;
    for (int i = 0; i < n; ++i) {
        pts_obj.push_back({objPoints[i].x, objPoints[i].y});
        pts_img.push_back({imgPoints[i].x, imgPoints[i].y});
    }

    Eigen::Matrix3d T_obj = get_norm_mat(pts_obj);
    Eigen::Matrix3d T_img = get_norm_mat(pts_img);

    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(2 * n, 9);
    for (int i = 0; i < n; ++i) {
        Eigen::Vector3d P = T_obj * Eigen::Vector3d(objPoints[i].x, objPoints[i].y, 1.0);
        Eigen::Vector3d p = T_img * Eigen::Vector3d(imgPoints[i].x, imgPoints[i].y, 1.0);
        double X = P.x(), Y = P.y(), u = p.x(), v = p.y();
        A.row(2 * i)     << X, Y, 1, 0, 0, 0, -u * X, -u * Y, -u;
        A.row(2 * i + 1) << 0, 0, 0, X, Y, 1, -v * X, -v * Y, -v;
    }

    Eigen::VectorXd h = Eigen::BDCSVD<Eigen::MatrixXd>(A, Eigen::ComputeThinV).matrixV().col(8);
    Eigen::Matrix3d H_norm;
    H_norm << h(0), h(1), h(2), h(3), h(4), h(5), h(6), h(7), h(8);

    Eigen::Matrix3d H = T_img.inverse() * H_norm * T_obj;
    return H / H(2, 2);
}

/**
 * @brief 构造 V 矩阵中的一行向量 v_ij
 */
Eigen::Matrix<double, 1, 6> get_v(const Eigen::Matrix3d& H, int i, int j) {
    // 这里的 i, j 是 1-based (符合论文公式)，转换到 0-based 索引
    auto h = [&](int row, int col) { return H(row - 1, col - 1); };
    Eigen::Matrix<double, 1, 6> v;
    v << h(1,i)*h(1,j), 
         h(1,i)*h(2,j) + h(2,i)*h(1,j), 
         h(2,i)*h(2,j),
         h(3,i)*h(1,j) + h(1,i)*h(3,j), 
         h(3,i)*h(2,j) + h(2,i)*h(3,j), 
         h(3,i)*h(3,j);
    return v;
}

int main() {
    // 1. 准备世界坐标 (9x6 棋盘格，间距 25mm)
    std::vector<cv::Point3f> worlds;
    for (int i = 0; i < 6; i++) {
        for (int j = 0; j < 9; j++) {
            worlds.push_back(cv::Point3f(j * 25.0f, i * 25.0f, 0.0f));
        }
    }

    // 2. 读取图片并检测角点
    std::string path = "assets/chessboard1/*.jpg";
    std::vector<std::string> filenames;
    cv::glob(path, filenames);

    Eigen::MatrixXd V_matrix(2 * filenames.size(), 6);
    int valid_count = 0;

    for (const auto& file : filenames) {
        cv::Mat frame = cv::imread(file);
        if (frame.empty()) continue;

        std::vector<cv::Point2f> corners;
        cv::Size boardSize(9, 6);
        if (cv::findChessboardCorners(frame, boardSize, corners)) {
            cv::Mat gray;
            cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
            cv::cornerSubPix(gray, corners, cv::Size(11, 11), cv::Size(-1, -1),
                             cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.1));

            // 计算单应性矩阵 H
            Eigen::Matrix3d H = computeHomography(worlds, corners);

            // 验证第一个点
            Eigen::Vector3d p_calc = H * Eigen::Vector3d(0, 0, 1);
            p_calc /= p_calc.z();
            std::cout << "Img: " << file << " | Err: " << std::abs(p_calc.x() - corners[0].x) << " px" << std::endl;

            // 构造 V 矩阵行
            V_matrix.row(2 * valid_count) = get_v(H, 1, 2);
            V_matrix.row(2 * valid_count + 1) = get_v(H, 1, 1) - get_v(H, 2, 2);
            valid_count++;

            cv::drawChessboardCorners(frame, boardSize, corners, true);
            cv::imshow("Calibration", frame);
            cv::waitKey(50);
        }
    }

    // 3. 求解 B 矩阵 (Vb = 0)
    if (valid_count < 3) {
        std::cerr << "需要至少 3 张有效的图片！" << std::endl;
        return -1;
    }

    Eigen::MatrixXd V_final = V_matrix.topRows(2 * valid_count);
    Eigen::VectorXd b = Eigen::BDCSVD<Eigen::MatrixXd>(V_final, Eigen::ComputeFullV).matrixV().col(5);

    // 4. 提取内参
    if (b(0) < 0) b = -b;
    double b11 = b(0), b12 = b(1), b22 = b(2), b13 = b(3), b23 = b(4), b33 = b(5);

    double v0 = (b12 * b13 - b11 * b23) / (b11 * b22 - b12 * b12);
    double lambda = b33 - (b13 * b13 + v0 * (b12 * b13 - b11 * b23)) / b11;
    double fx = std::sqrt(lambda / b11);
    double fy = std::sqrt(lambda * b11 / (b11 * b22 - b12 * b12));
    double u0 = (b12 * v0 / fx) - (b13 * fx * fx / lambda); 

    std::cout << "\n[结果报告]" << std::endl;
    std::cout << "有效图片数: " << valid_count << std::endl;
    std::cout << "内参矩阵 K:" << std::endl;
    std::cout << fx << "\t0\t" << u0 << std::endl;
    std::cout << "0\t" << fy << "\t" << v0 << std::endl;
    std::cout << "0\t0\t1" << std::endl;

    return 0;
}