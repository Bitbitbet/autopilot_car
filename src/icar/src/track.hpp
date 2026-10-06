#pragma once

#include "tools.hpp"

using namespace cv;

class Track {
  public:
    std::vector<PointX> pointsEdgeLeft;  // 赛道左边缘点集
    std::vector<PointX> pointsEdgeRight; // 赛道右边缘点集
    std::vector<PointX> widthBlock;      // 色块宽度=终-起（每行）
    std::vector<PointX> spurroad;        // 保存岔路信息
    double stdevLeft;                    // 边缘斜率方差（左）
    double stdevRight;                   // 边缘斜率方差（右）
    int validRowsLeft = 0;               // 边缘有效行数（左）
    int validRowsRight = 0;              // 边缘有效行数（右）
    uint16_t rowCutUp = 1;               // 图像顶部切行
    uint16_t rowCutBottom = 20;          // 图像底部切行

    void handle(Mat img);
    void handle(bool isResearch, uint16_t rowStart);
    void drawImage(Mat &img);
    double stdevEdgeCal(std::vector<PointX> &v_edge, int img_height);

  private:
    Mat imgShare; // 赛道搜索图像
    void slopeCal(std::vector<PointX> &edge, int index);
    void validRowsCal(void);
    int getMiddleValue(std::vector<int> vec);
};
