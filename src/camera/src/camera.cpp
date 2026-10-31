#include "camera.hpp"
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <opencv2/core/persistence.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

using cv::FileStorage;
using cv::Mat;
using cv::Point2d;
using cv::Point2f;
using cv::Scalar;
using cv::Size;
using std::cerr;
using std::cout;
using std::endl;

Camera::Camera() : use_otsu(true) { initialize("/dev/video0"); }
Camera::Camera(uint8_t threshold)
    : threshold_value(threshold), use_otsu(false) {
    initialize("/dev/video0");
}
Camera::Camera(const char *capture_source) : use_otsu(true) {
    initialize(capture_source);
}
Camera::Camera(const char *capture_source, uint8_t threshold)
    : threshold_value(threshold), use_otsu(false) {
    initialize(capture_source);
}

Camera::Camera(CalibrationOnly,int threshold):threshold_value(threshold<0?127:threshold),use_otsu(threshold<0),capture_width(640),capture_height(480){initializeCalibration();}
Camera::Camera(const Mat &h):threshold_value(127),use_otsu(true),capture_width(640),capture_height(480){if(!h.empty())setGroundHomography(h);}
void Camera::setGroundHomography(const Mat &h){h.convertTo(H_img2gnd,CV_64F);H_gnd2img=H_img2gnd.inv();}
void Camera::correct(Mat &frame) const {
    if(frame.size()!=Size(640,480))throw std::invalid_argument("Camera expects native 640x480 frames");
    if(!mapx.empty())remap(frame,frame,mapx,mapy,cv::INTER_LINEAR);
}

void Camera::initialize(const char *capture_source) {
    capture.open(capture_source);
    if (!capture.isOpened()) {
        cerr << "无法打开摄像机源 " << capture_source << endl;
        exit(4);
    }
    capture.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    capture.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    capture_height = capture.get(cv::CAP_PROP_FRAME_HEIGHT);
    capture_width = capture.get(cv::CAP_PROP_FRAME_WIDTH);
    if (!capture_width || !capture_height) {
        cerr << "无法读取摄像机源的宽高" << endl;
        exit(5);
    }
    cout << "打开了摄像机源 " << capture_source << "：\n  宽：" << capture_width
         << "\n  高：" << capture_height << "\n  FPS："
         << capture.get(cv::CAP_PROP_FPS) << endl;

    initializeCalibration();
}

void Camera::initializeCalibration() {
    // 读取xml中的相机标定参数
    auto camera_matrix = Mat(3, 3, CV_32FC1, Scalar::all(0)); // 摄像机内参矩阵
    auto dist_coeffs = Mat(1, 5, CV_32FC1, Scalar::all(0));   // 相机的畸变矩阵
    FileStorage file;
    if (file.open("../res/calibration/calibration.xml",
                  FileStorage::READ)) { // 读取本地保存的标定文件
        file["cameraMatrix"] >> camera_matrix;
        file["distCoeffs"] >> dist_coeffs;
        int referenceWidth=0,referenceHeight=0;
        file["image_width"]>>referenceWidth;file["image_height"]>>referenceHeight;
        if(!referenceWidth)file["imageWidth"]>>referenceWidth;
        if(!referenceHeight)file["imageHeight"]>>referenceHeight;
        camera_matrix.convertTo(camera_matrix,CV_64F);
        // Older calibration.xml has no dimensions. Its principal point
        // distinguishes this project's 320 and 640 reference images.
        if(!referenceWidth)referenceWidth=camera_matrix.at<double>(0,2)<240?320:640;
        if(!referenceHeight)referenceHeight=referenceWidth*3/4;
        if(referenceWidth*3!=referenceHeight*4)throw std::runtime_error("Lens reference must be 4:3");
        cout<<"[Camera] lens reference="<<referenceWidth<<'x'<<referenceHeight<<"; processing=640x480\n";
        const double scale=640.0/referenceWidth;
        camera_matrix.row(0)*=scale;camera_matrix.row(1)*=scale;
        Mat h_img2gnd, h_gnd2img;
        file["ground_H_img2gnd"] >> h_img2gnd;
        file["ground_H_gnd2img"] >> h_gnd2img;
        if (!h_img2gnd.empty()) {
            h_img2gnd.convertTo(H_img2gnd, CV_64F);
            H_img2gnd.col(0)/=scale;H_img2gnd.col(1)/=scale;
            H_gnd2img=H_img2gnd.inv();
        }
        if (H_img2gnd.empty()&&!h_gnd2img.empty()) {
            h_gnd2img.convertTo(H_gnd2img, CV_64F);
            H_gnd2img.row(0)*=scale;H_gnd2img.row(1)*=scale;H_img2gnd=H_gnd2img.inv();
        }
        file.release();
    } else {
        cerr << "打开相机矫正参数失败!!!" << endl;
        throw std::runtime_error("Cannot read lens calibration.xml");
    }
    if (H_img2gnd.empty() || H_gnd2img.empty()) {
        // 地面单应只影响 image_to_ground/ground_to_image，不阻塞取图
        cerr << "[Camera] No ground_H_*; task ground calibration may supply it; image-only tracking remains available" << endl;
    }

    Size camera_size{capture_width, capture_height};
    /* 经过矫正后的X坐标重映射参数 */
    mapx = Mat(camera_size, CV_32FC1);
    /* 经过矫正后的Y坐标重映射参数 */
    mapy = Mat(camera_size, CV_32FC1);
    auto rotMatrix = Mat::eye(3, 3, CV_32F); // 内参矩阵与畸变矩阵之间的旋转矩阵
    // 采用initUndistortRectifyMap+remap进行图像矫正
    initUndistortRectifyMap(camera_matrix, dist_coeffs, rotMatrix,
                            camera_matrix, camera_size, CV_32FC1, mapx, mapy);
}

Camera::~Camera() { capture.release(); }

/**
 * @brief 图像二值化
 *
 * 统一约定：可行驶路面为白，背景为黑。白路面直接阈值；黑路面取反，
 * 并修补被反相成黑的反光/白漆内部空洞（保留连接图像边界的背景排除区）。
 *
 * @param img  输入图像
 * @return cv::Mat 二值化图像
 */
Mat Camera::binarize(const Mat &img) {
    Mat gray_img;
    cvtColor(img, gray_img, cv::COLOR_BGR2GRAY); // RGB转灰度图

    const int threshold_value_arg = use_otsu ? -1 : threshold_value;
    const int polarity = blackRoad ? cv::THRESH_BINARY_INV : cv::THRESH_BINARY;
    Mat bin_img;
    threshold(gray_img, bin_img,
              threshold_value_arg < 0 ? 0 : threshold_value_arg, 255,
              polarity | (threshold_value_arg < 0 ? cv::THRESH_OTSU : 0));
    if (!blackRoad)
        return bin_img;

    // 只修补有界的内部空洞；与下边界连通的背景、与远边界连通的分隔带保留。
    Mat exterior;
    copyMakeBorder(bin_img, exterior, 1, 1, 1, 1, cv::BORDER_CONSTANT,
                   cv::Scalar(0));
    floodFill(exterior, {0, 0}, cv::Scalar(255));
    Mat holes;
    bitwise_not(exterior(cv::Rect(1, 1, bin_img.cols, bin_img.rows)), holes);
    Mat labels, stats, centers;
    const int count =
        connectedComponentsWithStats(holes, labels, stats, centers, 8, CV_32S);
    for (int component = 1; component < count; ++component) {
        if (stats.at<int>(component, cv::CC_STAT_AREA) >
                int(bin_img.total() / 10) ||
            stats.at<int>(component, cv::CC_STAT_WIDTH) >
                bin_img.cols * 45 / 100 ||
            stats.at<int>(component, cv::CC_STAT_HEIGHT) >
                bin_img.rows * 60 / 100)
            continue;
        const cv::Rect box(stats.at<int>(component, cv::CC_STAT_LEFT),
                           stats.at<int>(component, cv::CC_STAT_TOP),
                           stats.at<int>(component, cv::CC_STAT_WIDTH),
                           stats.at<int>(component, cv::CC_STAT_HEIGHT));
        bin_img(box).setTo(255, labels(box) == component);
    }
    return bin_img;
}

bool Camera::read(Mat &frame) {
    if (!capture.read(frame) || frame.empty()) {
        return false;
    }
    remap(frame, frame, mapx, mapy, cv::INTER_LINEAR);
    return true;
}

static Point2d applyHomography(const Mat &H, double x, double y) {
    if (H.empty()) { // 还没标定地面单应
        return Point2d(NAN, NAN);
    }
    const double w=H.at<double>(2,0)*x+H.at<double>(2,1)*y+H.at<double>(2,2);
    if(!std::isfinite(w)||std::abs(w)<1e-10)return {NAN,NAN};
    return {(H.at<double>(0,0)*x+H.at<double>(0,1)*y+H.at<double>(0,2))/w,
            (H.at<double>(1,0)*x+H.at<double>(1,1)*y+H.at<double>(1,2))/w};
}

Point2d Camera::image_to_ground(const Point2d &img_pt) const {
    return applyHomography(H_img2gnd, img_pt.x, img_pt.y);
}

Point2d Camera::ground_to_image(const Point2d &gnd_pt) const {
    return applyHomography(H_gnd2img, gnd_pt.x, gnd_pt.y);
}
