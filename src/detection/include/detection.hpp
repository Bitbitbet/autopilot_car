#pragma once

#include "predictor_api.h"
#include <memory>
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <sys/time.h>
#include <vector>

/**
 * @brief 目标检测结果
 *
 */
struct InferenceResult {
    int type;      // ID
    float score;   // 置信度
    cv::Rect rect; // 框的位置，x y(左上角) width height
};

class Detection {
  public:
    std::vector<InferenceResult> results; // AI推理结果

    /**
     * @brief Construct a new Detection object
     *
     * @param pathModel
     */
    Detection(const std::string pathModel, float minimum_confidence);

    /**
     * @brief AI模型推理
     */
    void inference(const cv::Mat &img);

    void draw_box(cv::Mat &img);

  private:
    // onnx info
    std::vector<std::string> onnx_input_names;
    std::vector<const char *> onnx_input_names_cstr;
    std::vector<std::string> onnx_output_names;
    std::vector<const char *> onnx_output_names_cstr;

    Ort::Env onnx_env;

    // predictor
    PPNCPredictor predictor_nna;
    PPNCPredictor predictor_nms;
    std::unique_ptr<Ort::Session> predictor_onnx;

    // 推理相关 helper
    float minimum_confidence = 0.5;
    static constexpr size_t PREPROCESS_DST_W = 320;
    static constexpr size_t PREPROCESS_DST_H = 320;
    NDTensor image, im_shape, scale_factor;
    void preprocess(const cv::Mat &frame);
    void run();
    void render();

    /**
     * @brief 获取Opencv颜色
     *
     * @param index 序号
     * @return cv::Scalar
     */
    cv::Scalar getCvcolor(int index);
};

// 必须与label_list.txt一致
enum : int {
    LABEL_CONE = 0,    // 锥桶
    LABEL_PERSON = 1,  // 行人
    LABEL_BUSY = 2,    // 施工区标
    LABEL_LIMIT = 3,   // 限速
    LABEL_UNLIMIT = 4, // 解除限速
    LABEL_STOP = 5,    // 禁止通行
    LABEL_FLOW = 6,    // 畅通
    LABEL_PARK = 7,    // 停车场
    LABEL_GATE = 8,    // 道闸
    LABEL_CROSS = 9,   // 斑马线（地面上）
    LABEL_FORK = 10,   // 岔路标志（地面上）
    LABEL_LEFT = 11,   // 左转标志（地面上）
    LABEL_CHOICE = 12, // 双向箭头（地面上）
    LABEL_YLEFT = 13,  // 左岔路口
    LABEL_YRIGHT = 14, // 右岔路口
};

inline const char *get_label_name(int label) {
    constexpr const char *NAMES[] = {
        "cone", "person", "busy", "limit", "unlimit", "stop",  "flow",  "park",
        "gate", "cross",  "fork", "left",  "choice",  "yleft", "yright"};

    return NAMES[label];
}

inline bool is_ground_label(int label) {
    return label >= LABEL_CROSS && label <= LABEL_CHOICE;
}

constexpr int LABEL_COUNT = 15;
