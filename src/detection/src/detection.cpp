#include "detection.hpp"
#include "nlohmann/json.hpp"
#include "predictor_api.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <numeric>
#include <onnxruntime_cxx_api.h>
#include <ppnc/predictor_api.h>
#include <stdexcept>
#include <stdlib.h>
#include <string>
#include <sys/time.h>
#include <unordered_map>
#include <vector>

using cv::Mat;
using cv::Scalar;
using std::accumulate;
using std::cerr;
using std::cout;
using std::endl;
using std::ifstream;
using std::make_unique;
using std::multiplies;
using std::string;
using std::to_string;
using std::vector;

void buildNms(const string &model_dir) {
    string model_file = model_dir + "/nms.tar";
    string untar_cmd = "tar -xf " + model_file + " -C . --no-same-owner";
    string final_file = model_file + ".so";
    string cc_cmd = "g++ -shared -fPIC -o " + final_file + " lib0.o devc.o";
    int sys_status = 0;

    sys_status = system(untar_cmd.c_str());
    if (sys_status) {
        cerr << "Error: cannot untar file " << model_file << endl;
        throw std::runtime_error("Cannot unpack NMS model: " + model_file);
    }

    // create shared
    sys_status = system(cc_cmd.c_str());
    if (sys_status) {
        cerr << "Error: compile for " << model_file << endl;
        throw std::runtime_error("Cannot build NMS model: " + model_file);
    }
    cout << "nms模型编译完成。" << endl;
}

Detection::Detection(const string pathModel, float score_nms)
    : predictor_nna("../res/models/config_ppncnna.json"),
      predictor_nms("../res/models/config_ppncnms.json"),
      image({1, 3, PREPROCESS_DST_H, PREPROCESS_DST_W}), im_shape({1, 2}),
      scale_factor({1, 2}) {
    score = score_nms;

    this->onnx_env =
        Ort::Env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, "test");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(8);
    session_options.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_ALL);
    string onnx_model = pathModel + "/post.onnx";
    this->predictor_onnx = make_unique<Ort::Session>(
        this->onnx_env, onnx_model.c_str(), session_options);

    // ONNX模型加载
    onnx_input_names.push_back("im_shape");
    onnx_input_names.push_back("scale_factor");
    ifstream ifs(pathModel + "/io_paddle.json");
    nlohmann::json j;
    ifs >> j;
    ifs.close();
    for (size_t i = 0; i < j.size(); ++i) {
        if (j[i]["type"] == "OUTPUT") {
            assert(j[i]["shape"].size() == 4);

            onnx_input_names.push_back(j[i]["name"]);
        }
        if (j[i]["type"] == "post_out") {
            onnx_output_names.push_back(j[i]["name"]);
        }
    }
    onnx_input_names.shrink_to_fit();
    onnx_output_names.shrink_to_fit();
    for (const auto &str : onnx_input_names) {
        onnx_input_names_cstr.push_back(str.c_str());
    }
    for (const auto &str : onnx_output_names) {
        onnx_output_names_cstr.push_back(str.c_str());
    }
    onnx_input_names_cstr.shrink_to_fit();
    onnx_output_names_cstr.shrink_to_fit();

    buildNms(pathModel); // 编译生成.so文件

    this->predictor_nna.load();
    this->predictor_nms.load();

    // 模型标签加载
    string pathLabels = pathModel + "/label_list.txt";

    ifstream file(pathLabels);
    if (!file.is_open()) {
        cerr << "Open Lable File failed: " << pathLabels << endl;
        exit(20);
    }
    string line;
    while (getline(file, line)) {
        labels.push_back(line);
    }
}

void Detection::preprocess(const Mat &frame) {

    const int dst_h = PREPROCESS_DST_H;
    const int dst_w = PREPROCESS_DST_W;
    const int src_h = frame.rows;
    const int src_w = frame.cols;

    Mat resized;
    if (src_w != dst_w || src_h != dst_h) {
        cv::resize(frame, resized, cv::Size(dst_w, dst_h), 0, 0,
                   cv::INTER_LINEAR);
    } else {
        resized = frame;
    }

    /* ------ 写 img ------ */
    float *out = image.value();
    const auto plane = (size_t)dst_h * dst_w;

    // (p/255 - mean) * std_inv = p * (std_inv/255) - mean*std_inv
    static constexpr float kMean[3] = {0.485f, 0.456f, 0.406f}; // R,G,B
    static constexpr float kStdInv[3] = {1.f / 0.229f, 1.f / 0.224f,
                                         1.f / 0.225f};
    const float aR = kStdInv[0] / 255.f, bR = -kMean[0] * kStdInv[0];
    const float aG = kStdInv[1] / 255.f, bG = -kMean[1] * kStdInv[1];
    const float aB = kStdInv[2] / 255.f, bB = -kMean[2] * kStdInv[2];

    float *out_r = out;
    float *out_g = out + plane;
    float *out_b = out + 2 * plane;

    if (resized.isContinuous()) {
        const uint8_t *p = resized.ptr<uint8_t>(0);
        for (size_t i = 0; i < plane; ++i) {
            const uint8_t b = p[3 * i];
            const uint8_t g = p[3 * i + 1];
            const uint8_t r = p[3 * i + 2];
            out_r[i] = r * aR + bR;
            out_g[i] = g * aG + bG;
            out_b[i] = b * aB + bB;
        }
    } else {
        for (int y = 0; y < dst_h; ++y) {
            const uint8_t *row = resized.ptr<uint8_t>(y);
            for (int x = 0; x < dst_w; ++x) {
                out_r[y * dst_w + x] = row[3 * x + 2] * aR + bR;
                out_g[y * dst_w + x] = row[3 * x + 1] * aG + bG;
                out_b[y * dst_w + x] = row[3 * x] * aB + bB;
            }
        }
    }

    /* ------ 写 im_shape ------ */
    im_shape.value()[0] = dst_h;
    im_shape.value()[1] = dst_w;

    /* ------ 写 scale_factor ------ */
    scale_factor.value()[0] = (float)dst_h / src_h;
    scale_factor.value()[1] = (float)dst_w / src_w;
}
void Detection::inference(const cv::Mat &img) {
    preprocess(img); // 图像前处理
    run();           // 模型推理
    render();        // 后处理
}

void Detection::run() {
    // ppnc_nna run
    predictor_nna.set_inputs({{"image", image}});
    predictor_nna.run();

    auto memory_info =
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    vector<Ort::Value> input_tensors;

    int numel = accumulate(im_shape.shape.begin(), im_shape.shape.end(), 1,
                           multiplies<>());
    input_tensors.push_back(Ort::Value::CreateTensor<float>(
        memory_info, im_shape.value(), numel, im_shape.shape.data(),
        im_shape.shape.size()));
    numel = accumulate(scale_factor.shape.begin(), scale_factor.shape.end(), 1,
                       multiplies<>());
    input_tensors.push_back(Ort::Value::CreateTensor<float>(
        memory_info, scale_factor.value(), numel, scale_factor.shape.data(),
        scale_factor.shape.size()));

    for (size_t i = 2; i < this->onnx_input_names_cstr.size(); ++i) {
        const NDTensor &t = this->predictor_nna.get_output(i - 2);
        numel = accumulate(t.shape.begin(), t.shape.end(), 1, multiplies<>());
        input_tensors.push_back(Ort::Value::CreateTensor<float>(
            memory_info, t.value(), numel, t.shape.data(), t.shape.size()));
    }

    Ort::RunOptions run_options;
    auto onnx_out = predictor_onnx->Run(
        run_options, onnx_input_names_cstr.data(), input_tensors.data(),
        input_tensors.size(), onnx_output_names_cstr.data(),
        onnx_output_names_cstr.size());

    auto *data0 = onnx_out[0].GetTensorData<float>();
    auto *data1 = onnx_out[1].GetTensorData<float>();
    auto output_shape0 = onnx_out[0].GetTensorTypeAndShapeInfo().GetShape();
    auto output_shape1 = onnx_out[1].GetTensorTypeAndShapeInfo().GetShape();

    int32_t cls_num, num, box_point;
    // bbox
    num = output_shape0[1];
    box_point = output_shape0[2];
    cls_num = output_shape1[1];

    NDTensor bboxes({1, num, box_point});
    NDTensor scores({1, cls_num, num});

    memcpy(bboxes.value(), data0, num * box_point * sizeof(float));
    memcpy(scores.value(), data1, cls_num * num * sizeof(float));

    // ppnc_nms run
    this->predictor_nms.set_inputs({{"bboxes", bboxes}, {"scores", scores}});
    this->predictor_nms.run();
}

void Detection::render() {
    NDTensor res = predictor_nms.get_output(0);
    auto data = res.value();
    auto prod = accumulate(res.shape.begin(), res.shape.end(), 1,
                           multiplies<int64_t>());

    results.clear();
    PredictResult result;
    for (int i = 0; i < prod; i += 6) {
        result.type = data[i];
        result.score = data[i + 1];

        if (result.score < score) // 阈值
            continue;

        // turnning....
        if ((size_t)result.type < labels.size())
            result.label = labels[result.type];
        result.x = data[i + 2];
        result.y = data[i + 3];
        result.width = data[i + 4] - data[i + 2];
        result.height = data[i + 5] - data[i + 3];
        results.push_back(result);
    }
}

void Detection::drawBox(Mat &img) { drawBox(img, results); }

void Detection::drawBox(Mat &img, const vector<PredictResult> &snapshot) {
    for (const auto &item : snapshot) {
        PredictResult result = item;

        if (find(drawSkipLabels.begin(), drawSkipLabels.end(), result.type) !=
            drawSkipLabels.end())
            continue; // 跳过指定标签（normal态滤掉fork，避免框盖住车道线影响画线）

        auto score = to_string(result.score);
        int pointY = result.y - 15;
        if (pointY < 0)
            pointY = 0;
        cv::Rect rectText(result.x, pointY, result.width, 20);
        cv::rectangle(img, rectText, getCvcolor(result.type), -1);
        string label_name =
            result.label + " [" + score.substr(0, score.find(".") + 3) + "]";
        cv::Rect rect(result.x, result.y, result.width, result.height);
        cv::rectangle(img, rect, getCvcolor(result.type), 1);

        if (result.y < 15)
            result.y = 15;
        cv::putText(img, label_name, cv::Point(result.x, result.y),
                    cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 0, 0), 1);
    }
}

Scalar Detection::getCvcolor(int index) {
    switch (index) { // BGR
    case 0:
        return Scalar(204, 0, 255); // 锥桶: 紫色
        break;
    case 1:
        return Scalar(0, 0, 255); // 行人: 紫色
        break;
    case 2:
        return Scalar(0, 255, 255); // 施工区标志: 黄色
        break;
    case 3:
        return Scalar(0, 255, 255); // 限速: 黄色
        break;
    case 4:
        return Scalar(0, 255, 0); // 解除限速: 绿色
        break;
    case 5:
        return Scalar(0, 0, 255); // 禁行标志: 红色
        break;
    case 6:
        return Scalar(0, 255, 0); // 畅行标志: 绿色
        break;
    case 7:
        return Scalar(255, 51, 0); // 停车场标志: 蓝色
        break;
    case 8:
        return Scalar(0, 0, 255); // 阻拦杆: 红色
        break;
    case 9:
        return Scalar(255, 255, 255); // 斑马线: 白色
        break;
    case 10:
        return Scalar(0, 102, 255); // 岔路标志: 橙色
        break;
    case 11:
        return Scalar(0, 102, 255); // 左转标志: 橙色
        break;
    case 12:
        return Scalar(0, 102, 255); // 双向标志: 橙色
        break;
    case 13:
        return Scalar(255, 51, 0); // 左岔路标志: 蓝色
        break;
    case 14:
        return Scalar(255, 51, 0); // 右岔路标志: 蓝色
        break;
    case 15:
        return Scalar(255, 255, 255);
        break;
    case 16:
        return Scalar(237, 226, 19);
        break;
    case 17:
        return Scalar(132, 255, 10);
        break;
    default:
        return Scalar(255, 0, 0);
        break;
    }
}
