#pragma once

#include "mapping.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>

using namespace cv;

#define COLSCAMERA 320   // 相机的列数
#define ROWSCAMERA 240   // 相机的行数
#define COLSIMAGE 320    // 图像的列数
#define ROWSIMAGE 240    // 图像的行数
#define COLSIMAGEIPM 800 // IPM图像的列数
#define ROWSIMAGEIPM 500 // IPM图像的行数
#define PWMSERVOMAX 1900 // 舵机PWM最大值（左）1840
#define PWMSERVOMID 1500 // 舵机PWM中值 1520
#define PWMSERVOMIN 1100 // 舵机PWM最小值（右）1200

#define LABEL_CONE 0    // AI标签: 锥桶
#define LABEL_PERSON 1  // AI标签: 行人
#define LABEL_BUSY 2    // AI标签: 施工区标
#define LABEL_LIMIT 3   // AI标签: 限速标志
#define LABEL_UNLIMIT 4 // AI标签: 解除限速标志
#define LABEL_STOP 5    // AI标签: 禁止通行标志
#define LABEL_FLOW 6    // AI标签: 畅通标志
#define LABEL_PARK 7    // AI标签: 停车场
#define LABEL_GATE 8    // AI标签: ETC阻拦杆
#define LABEL_CROSS 9   // AI标签: 斑马线
#define LABEL_FORK 10   // AI标签: 岔路标志
#define LABEL_LEFT 11   // AI标签: 左转标志
#define LABEL_CHOICE 12 // AI标签: 双向箭头标志
#define LABEL_YLEFT 13  // AI标签: 左岔路口标志
#define LABEL_YRIGHT 14 // AI标签: 右岔路口标志

extern Mapping ipm;

/**
 * @brief 目标检测结果
 *
 */
struct PredictResult {
    int type;          // ID
    std::string label; // 标签
    float score;       // 置信度
    int x;             // 坐标(左下角点)
    int y;             // 坐标
    int width;         // 尺寸
    int height;        // 尺寸
};

/**
 * @brief 构建二维坐标
 *
 */
struct PointX {
    int x = 0;
    int y = 0;
    float slope = 0.0f;

    PointX() {};
    PointX(int x, int y) : x(x), y(y) {};
    PointX(int x, int y, float z) : x(x), y(y), slope(z) {};
};

/**
 * @brief 存储图像至本地
 *
 * @param image 需要存储的图像
 */
void savePicture(Mat &image);

/**
 * @brief 存储图像至本地
 *
 * @param image 需要存储的图像
 */
void savePicture(std::string path, Mat &image);

/**
 * @brief int集合平均值计算
 *
 * @param arr 输入数据集合
 * @return double
 */
double average(std::vector<int> vec);

/**
 * @brief int集合数据方差计算
 *
 * @param vec Int集合
 * @return double
 */
double sigma(std::vector<int> vec);

/**
 * @brief 赛道点集的方差计算
 *
 * @param vec
 * @return double
 */
double sigma(std::vector<PointX> vec);

/**
 * @brief 贝塞尔曲线
 *
 * @param dt
 * @param input
 * @return vector<PointX>
 */
std::vector<PointX> Bezier(double dt, std::vector<PointX> input);

/**
 * @brief 格式化double类型数据为字符串
 *
 * @param val 输入数据
 * @param fixed 保留小数位数
 * @return auto 输出字符串
 */
std::string double2String(double val, int fixed);
/**
 * @brief 点到直线的距离计算
 *
 * @param a 直线的起点
 * @param b 直线的终点
 * @param p 目标点
 * @return double
 */
double distanceForPoint2Line(PointX a, PointX b, PointX p);

/**
 * @brief 两点之间的距离
 *
 * @param a
 * @param b
 * @return double
 */
inline double distanceForPoints(PointX a, PointX b) {
    return sqrt((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y));
}
inline double distanceForPoints(Point a, Point b) {
    return sqrt((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y));
}

/**
 * @brief 计算两点原是域上距离
 *
 * @param startPoint起点
 * @param endPoint终点
 *        KickEdgePoint 是否剔除边界点（包含边界点的距离默认999）
 * @return double
 */
double distanceForOrigin(PointX startPoint, PointX endPoint);

/**
 * @brief 自定义的比较函数，根据元素的x值进行比较
 * @inform
 * @return
 */
inline bool compareX(const PointX &a, const PointX &b) {
    return a.x > b.x; // 降序排序
}

/**
 * @brief 图像帧数是否在范围内判断
 * @param 输入起始点和终止点
 * @inform 可以用于任何判断是否在范围内
 * @return
 */
bool posInRange(double pos, double a = 0, double b = ROWSIMAGE);
/**
 * @brief 利用贝塞尔曲线的补线函数
 *
 * @param k 纵向倍率，默认为1
 * @param n 贝塞尔参数，越大补线间隔越大，点越稀疏
 * @param 输入起始点和终止点
 *
 * @return
 */
std::vector<PointX> simpleLine(PointX startPoint, PointX endPoint,
                               double n = 0.1, double k = 1);

/**
 * @brief 图像平滑算法
 * @param 输入点集
 *
 * @return
 */
std::vector<PointX> smoothLine(std::vector<PointX> Temp1, double n = 0.1,
                               double k = 1);

/**
 * @brief 计算两点之间斜率
 * @param PointX a,b
 * @return double
 */
double gradientCal(PointX a, PointX b);

/**
 * @brief 判断范围内边界点是否为一条直线
 *
 * @param  PointsEdge边界点集
 *         startLine起点
 *         endLine终点
 * @return bias起点终点拟合直线后偏移较大的点
 */
int linearCheck(std::vector<PointX> PointsEdge, int startLine = 0,
                int endLine = ROWSIMAGE - 1, int judgebias = 3);

/**
 * @brief 矫正域上判断点的位置是否适合
 *
 * @param startPoint起点
 * @param endPoint终点
 * @param Min_Len 矫正域赛道允许最短长度
 * @param Max_Len 矫正域赛道允许最长长度
 * @return bool
 */
bool breakpointCheck(PointX startPoint, PointX endPoint, int Min_Len = 80,
                     int Max_Len = 110);