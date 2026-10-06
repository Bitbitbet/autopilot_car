#pragma once

#include "predictor_api.h"
#include "tools.hpp"
#include <memory>
#include <onnxruntime_cxx_api.h>
#include <stdlib.h>
#include <string>
#include <sys/time.h>
#include <unordered_map>
#include <vector>

class Detection {
  public:
    std::vector<PredictResult> results; // AI推理结果
    float score = 0.5;                  // AI检测置信度
    std::vector<int>
        drawSkipLabels; // 调试画面需跳过的标签(type)：normal态滤掉fork不画

    /**
     * @brief Construct a new Detection object
     *
     * @param pathModel
     */
    Detection(const std::string pathModel, float score_nms);

    /**
     * @brief AI模型推理
     */
    void inference(cv::Mat img);

    void transposeAndCopyToTensor(const Mat &src, NDTensor &dst);

    std::shared_ptr<std::unordered_map<std::string, NDTensor>>
    preprocess(cv::Mat frame, const std::vector<int64_t> &input_size);

    void run(const std::unordered_map<std::string, NDTensor> &feeds);

    NDTensor get_output(int index);

    void drawBox(Mat &img);

    /**
     * @brief 获取Opencv颜色
     *
     * @param index 序号
     * @return cv::Scalar
     */
    cv::Scalar getCvcolor(int index);

  private:
    std::vector<std::string> labels;
    // onnx info
    std::pair<std::vector<std::string>, std::vector<const char *>>
        onnx_input_names_;
    std::pair<std::vector<std::string>, std::vector<const char *>>
        onnx_out_names_;
    Ort::Env onnx_env_;
    // predictor
    std::shared_ptr<PPNCPredictor> predictor_nna_;
    std::shared_ptr<PPNCPredictor> predictor_nms_;
    std::shared_ptr<Ort::Session> predictor_onnx_;

    void render();
};
