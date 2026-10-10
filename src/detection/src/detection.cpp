#include "detection.hpp"
#include "nlohmann/json.hpp"
#include "predictor_api.h"
#include "tools.hpp"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <onnxruntime_cxx_api.h>
#include <ppnc/predictor_api.h>
#include <stdlib.h>
#include <string>
#include <sys/time.h>
#include <unordered_map>
#include <vector>

using std::accumulate;
using std::cerr;
using std::cout;
using std::endl;
using std::ifstream;
using std::make_shared;
using std::multiplies;
using std::shared_ptr;
using std::string;
using std::to_string;
using std::unordered_map;
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
    cout << "compile done." << endl;
}

Detection::Detection(const string pathModel, float score_nms) {
    score = score_nms;
    // 模型初始化
    this->predictor_nna_ =
        make_shared<PPNCPredictor>("../res/models/config_ppncnna.json");
    this->predictor_nms_ =
        make_shared<PPNCPredictor>("../res/models/config_ppncnms.json");
    this->onnx_env_ =
        Ort::Env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, "test");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(8);
    session_options.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_ALL);
    string onnx_model = pathModel + "/post.onnx";
    this->predictor_onnx_ = make_shared<Ort::Session>(
        this->onnx_env_, onnx_model.c_str(), session_options);

    // ONNX模型加载
    this->onnx_input_names_.first.push_back("im_shape");
    this->onnx_input_names_.first.push_back("scale_factor");
    ifstream ifs(pathModel + "/io_paddle.json");
    nlohmann::json j;
    ifs >> j;
    ifs.close();
    for (size_t i = 0; i < j.size(); ++i) {
        if (j[i]["type"] == "OUTPUT") {
            assert(j[i]["shape"].size() == 4);
            this->onnx_input_names_.first.push_back(j[i]["name"]);
        }
        if (j[i]["type"] == "post_out") {
            this->onnx_out_names_.first.push_back(j[i]["name"]);
        }
    }

    for (auto &s : this->onnx_input_names_.first) {
        this->onnx_input_names_.second.push_back(s.c_str());
    }

    for (auto &s : this->onnx_out_names_.first) {
        this->onnx_out_names_.second.push_back(s.c_str());
    }
    buildNms(pathModel); // 编译生成.so文件

    this->predictor_nna_->load();
    this->predictor_nms_->load();

    // 模型标签加载
    string pathLabels = pathModel + "/label_list.txt";
    labels.clear();
    ifstream file(pathLabels);
    if (file.is_open()) {
        string line;
        while (getline(file, line)) {
            labels.push_back(line);
        }
        file.close();
    } else {
        cerr << "Open Lable File failed: " << pathLabels << endl;
    }
};

void Detection::inference(cv::Mat img) {
    auto feeds = preprocess(img, {320, 320}); // 图像前处理
    run(*feeds);                              // 模型推理
    render();                                 // 后处理
}

void Detection::transposeAndCopyToTensor(const Mat &src, NDTensor &dst) {
    Mat channels[3];
    split(src, channels);
    int offset = src.rows * src.cols;
    auto start = dst.value();
    for (int i = 0; i < 3; ++i) {
        memcpy(start + i * offset, channels[i].data, offset * sizeof(float));
    }
}

shared_ptr<unordered_map<string, NDTensor>>
Detection::preprocess(cv::Mat frame, const vector<int64_t> &input_size) {
    cv::Mat x;
    NDTensor scale_factor({1, 2}), img({1, 3, input_size[0], input_size[1]});
    scale_factor.value()[0] = static_cast<float>(input_size[0]) / frame.size[0];
    scale_factor.value()[1] = static_cast<float>(input_size[1]) / frame.size[1];
    NDTensor im_shape({1, 2});
    im_shape.value()[0] = input_size[0];
    im_shape.value()[1] = input_size[1];
    cv::cvtColor(frame, x, cv::COLOR_BGR2RGB);
    cv::resize(x, x, cv::Size(input_size[0], input_size[1]), 0, 0, 2);
    x.convertTo(x, CV_32FC3);
    x *= 1 / 255.0;
    cv::subtract(x, cv::Scalar(0.485, 0.456, 0.406), x);
    cv::multiply(x, cv::Scalar(1 / 0.229, 1 / 0.224, 1 / 0.225), x);
    transposeAndCopyToTensor(x, img);

    unordered_map<string, NDTensor> ret = {
        {"image", img}, {"im_shape", im_shape}, {"scale_factor", scale_factor}};

    return make_shared<unordered_map<string, NDTensor>>(ret);
}

void Detection::run(const unordered_map<string, NDTensor> &feeds) {
    auto &image = feeds.at("image");
    this->predictor_nna_->set_inputs({{"image", image}});

    // ppnc_nna run
    this->predictor_nna_->run();

    auto memory_info =
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    vector<Ort::Value> input_tensors;
    NDTensor im_shape = feeds.at("im_shape");
    NDTensor scale_factor = feeds.at("scale_factor");

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

    for (size_t i = 2; i < this->onnx_input_names_.second.size(); ++i) {
        const NDTensor &t = this->predictor_nna_->get_output(i - 2);
        numel = accumulate(t.shape.begin(), t.shape.end(), 1, multiplies<>());
        input_tensors.push_back(Ort::Value::CreateTensor<float>(
            memory_info, t.value(), numel, t.shape.data(), t.shape.size()));
    }

    Ort::RunOptions run_options;
    auto onnx_out = this->predictor_onnx_->Run(
        run_options, this->onnx_input_names_.second.data(),
        input_tensors.data(), input_tensors.size(),
        this->onnx_out_names_.second.data(),
        this->onnx_out_names_.second.size());

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
    this->predictor_nms_->set_inputs({{"bboxes", bboxes}, {"scores", scores}});
    this->predictor_nms_->run();
}

NDTensor Detection::get_output(int index) {
    return this->predictor_nms_->get_output(index);
}

void Detection::render() {
    NDTensor res = get_output(0);
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
        if (result.type < labels.size())
            result.label = labels[result.type];
        result.x = data[i + 2];
        result.y = data[i + 3];
        result.width = data[i + 4] - data[i + 2];
        result.height = data[i + 5] - data[i + 3];
        results.push_back(result);
    }
}

void Detection::drawBox(Mat &img) {
    drawBox(img, results);
}

void Detection::drawBox(Mat &img, const std::vector<PredictResult> &snapshot) {
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
        cv::putText(img, label_name, Point(result.x, result.y),
                    cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 0, 0), 1);
    }
}

cv::Scalar Detection::getCvcolor(int index) {
    switch (index) // BGR
    {
    case 0:
        return cv::Scalar(204, 0, 255); // 锥桶: 紫色
        break;
    case 1:
        return cv::Scalar(0, 0, 255); // 行人: 紫色
        break;
    case 2:
        return cv::Scalar(0, 255, 255); // 施工区标志: 黄色
        break;
    case 3:
        return cv::Scalar(0, 255, 255); // 限速: 黄色
        break;
    case 4:
        return cv::Scalar(0, 255, 0); // 解除限速: 绿色
        break;
    case 5:
        return cv::Scalar(0, 0, 255); // 禁行标志: 红色
        break;
    case 6:
        return cv::Scalar(0, 255, 0); // 畅行标志: 绿色
        break;
    case 7:
        return cv::Scalar(255, 51, 0); // 停车场标志: 蓝色
        break;
    case 8:
        return cv::Scalar(0, 0, 255); // 阻拦杆: 红色
        break;
    case 9:
        return cv::Scalar(255, 255, 255); // 斑马线: 白色
        break;
    case 10:
        return cv::Scalar(0, 102, 255); // 岔路标志: 橙色
        break;
    case 11:
        return cv::Scalar(0, 102, 255); // 左转标志: 橙色
        break;
    case 12:
        return cv::Scalar(0, 102, 255); // 双向标志: 橙色
        break;
    case 13:
        return cv::Scalar(255, 51, 0); // 左岔路标志: 蓝色
        break;
    case 14:
        return cv::Scalar(255, 51, 0); // 右岔路标志: 蓝色
        break;
    case 15:
        return cv::Scalar(255, 255, 255);
        break;
    case 16:
        return cv::Scalar(237, 226, 19);
        break;
    case 17:
        return cv::Scalar(132, 255, 10);
        break;
    default:
        return cv::Scalar(255, 0, 0);
        break;
    }
}
