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
#include "predict_result.hpp"

class Detection {
  public:
    std::vector<PredictResult> results; // AI推理结果
    std::vector<int>
        drawSkipLabels; // 调试画面需跳过的标签(type)：normal态滤掉fork不画

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

    void drawBox(cv::Mat &img);
    void drawBox(cv::Mat &img, const std::vector<PredictResult> &snapshot);

  private:
    std::vector<std::string> labels;
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
