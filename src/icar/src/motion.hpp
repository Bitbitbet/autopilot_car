#pragma once

#include "params.hpp"

class Motion {
  public:
    /**
     * @brief 姿态PD控制器
     *
     * @param center 智能车控制中心
     */
    void poseControl(std::shared_ptr<Params> &params);

    /**
     * @brief 变加速控制
     *
     * @param params
     */
    void speedControl(std::shared_ptr<Params> &params);

    /**
     * @brief 显示赛道线识别结果
     *
     * @param img 需要叠加显示的图像
     */
    void drawImage(std::shared_ptr<Params> &params, Mat &img);

    /**
     * @brief 车辆冲出赛道检测（保护车辆）
     *
     * @param track
     * @return true
     * @return false
     */
    void outlineCheck(std::shared_ptr<Params> &params);

  private:
    int countRes = 0;
    int countOut = 0;
    bool outline = false; // 出线标志
};
