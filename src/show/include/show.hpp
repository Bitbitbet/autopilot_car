#pragma once

#include <opencv2/core/mat.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;

class Show {
  private:
    bool enable = false;   // 显示窗口使能
    int sizeWindow = 1;    // 窗口数量
    Mat imgShow;           // 窗口图像
    bool realShow = false; // 实时更新画面
  public:
    int index = 0;      // 图像序号
    int indexLast = -1; // 图像序号
    int frameMax = 0;   // 视频总帧数
    bool save = false;  // 图像存储

    /**
     * @brief 显示窗口初始化
     *
     * @param size 窗口数量(1~7)
     */
    Show(const int size);
    ~Show();

    /**
     * @brief 设置新窗口属性
     *
     * @param index 窗口序号
     * @param name 窗口名称
     * @param img 显示图像
     */
    void setNewWindow(int index, std::string name, Mat img);

    /**
     * @brief 融合后的图像显示
     *
     */
    void show(void);
};
