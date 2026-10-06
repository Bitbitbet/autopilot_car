#pragma once

#include "params.hpp"
#include "tools.hpp"
#include <cstdint>

#define DIS_MOVE                                                               \
    48 // 偏移距离，对应赛道距离的一般，在我的打表软件中默认48像素对应20cm
#define MAX_POINT_NUM 240 // 无需修改
#define DIS_SECTION 75    // 使用centerMove时每一段点集的最大长度

/**
 * @brief 控制中心处理类
 *
 */
class Center {

  public:
    uint16_t validRowsLeft = 0;  // 边缘有效行数（左）
    uint16_t validRowsRight = 0; // 边缘有效行数（右）
    double sigmaCenter = 0;      // 中心点集的方差

    void fitting(std::shared_ptr<Params> &params);
    void drawImage(std::shared_ptr<Params> &params, Mat &img);

  private:
    std::string style = "";
    int countOut = 0;
    int timeout = 0;

    void showMode(Mat &img, FsmMode mode);
    uint16_t searchBreakLeftDown(std::vector<PointX> pointsEdgeLeft);
    uint16_t searchBreakRightDown(std::vector<PointX> pointsEdgeRight);
    std::vector<PointX> centerCompute(std::vector<PointX> pointsEdge, int side);
    void validRowsCal(std::vector<PointX> pointsEdgeLeft,
                      std::vector<PointX> pointsEdgeRight);
    void derailmentCheck(std::vector<PointX> pointsEdgeLeft,
                         std::vector<PointX> pointsEdgeRight);
};