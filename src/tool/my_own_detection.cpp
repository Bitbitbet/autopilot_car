#include "camera.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

using std::ifstream;
using std::string;
using std::vector;
using std::chrono::microseconds;

vector<string> labels;

struct PredictResult {
    int type;    // 类别 ID
    float score; // 置信度
    int x;       // 坐标(左下角点)
    int y;       // 坐标
    int width;   // 尺寸
    int height;  // 尺寸
};
cv::Scalar getCvcolor(int index) {
    switch (index) { // BGR
    case 0:
        return cv::Scalar(204, 0, 255);
    case 1:
        return cv::Scalar(0, 0, 255);
    case 2:
        return cv::Scalar(0, 255, 255);
    case 3:
        return cv::Scalar(0, 255, 255);
    case 4:
        return cv::Scalar(0, 255, 0);
    case 5:
        return cv::Scalar(0, 0, 255);
    case 6:
        return cv::Scalar(0, 255, 0);
    case 7:
        return cv::Scalar(255, 51, 0);
    case 8:
        return cv::Scalar(0, 0, 255);
    case 9:
        return cv::Scalar(255, 255, 255);
    case 10:
        return cv::Scalar(0, 102, 255);
    case 11:
        return cv::Scalar(0, 102, 255);
    case 12:
        return cv::Scalar(0, 102, 255);
    case 13:
        return cv::Scalar(255, 51, 0);
    case 14:
        return cv::Scalar(255, 51, 0);
    case 15:
        return cv::Scalar(255, 255, 255);
    case 16:
        return cv::Scalar(237, 226, 19);
    case 17:
        return cv::Scalar(132, 255, 10);
    default:
        return cv::Scalar(255, 0, 0);
    }
}
void drawBox(cv::Mat &img, const std::vector<PredictResult> &snapshot) {
    for (const auto &item : snapshot) {
        PredictResult result = item;
        auto score = std::to_string(result.score);
        int pointY = result.y - 15;
        if (pointY < 0)
            pointY = 0;
        cv::Rect rectText(result.x, pointY, result.width, 20);
        cv::rectangle(img, rectText, getCvcolor(result.type), -1);
        std::string label_name = labels[result.type] + " [" +
                                 score.substr(0, score.find(".") + 3) + "]";
        cv::Rect rect(result.x, result.y, result.width, result.height);
        cv::rectangle(img, rect, getCvcolor(result.type), 1);
        if (result.y < 15)
            result.y = 15;
        cv::putText(img, label_name, cv::Point(result.x, result.y),
                    cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 0, 0), 1);
    }
}
struct LetterboxInfo {
    float scale; // 缩放比例
    int padTop;  // 顶部填充像素
    int padLeft; // 左侧填充像素（本场景为 0）
};

// 输入: 原始 640x480 BGR 图像
// 输出: 320x320 RGB float32 NCHW 张量 (已归一化到 0~1)
LetterboxInfo letterbox(const cv::Mat &src, std::vector<float> &tensor) {
    const int targetW = 320, targetH = 320;
    const cv::Scalar fillColor(114, 114, 114); // BGR 灰色

    LetterboxInfo info;
    info.scale = std::min((float)targetW / src.cols, (float)targetH / src.rows);

    int newW = std::round(src.cols * info.scale);
    int newH = std::round(src.rows * info.scale);

    cv::Mat resized;
    cv::resize(src, resized, cv::Size(newW, newH));

    info.padLeft = (targetW - newW) / 2;
    info.padTop = (targetH - newH) / 2;

    cv::Mat padded(targetH, targetW, CV_8UC3, fillColor);
    resized.copyTo(padded(cv::Rect(info.padLeft, info.padTop, newW, newH)));

    // BGR -> RGB, uint8 -> float32 / 255
    cv::Mat rgb;
    cv::cvtColor(padded, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(rgb, CV_32F, 1.0 / 255.0);

    // HWC -> CHW
    tensor.resize(3 * targetH * targetW);
    std::vector<cv::Mat> channels(3);
    for (int c = 0; c < 3; ++c) {
        channels[c] = cv::Mat(targetH, targetW, CV_32F,
                              tensor.data() + c * targetH * targetW);
    }
    cv::split(rgb, channels);

    return info;
}
struct Detection {
    cv::Rect box; // 原图坐标系下的框
    float score;
    int classId;
};

// 将 letterbox 坐标系下的框映射回原图
cv::Rect scaleBox(float cx, float cy, float w, float h,
                  const LetterboxInfo &info, int imgW, int imgH) {
    float x1 = (cx - w / 2.0f - info.padLeft) / info.scale;
    float y1 = (cy - h / 2.0f - info.padTop) / info.scale;
    float x2 = (cx + w / 2.0f - info.padLeft) / info.scale;
    float y2 = (cy + h / 2.0f - info.padTop) / info.scale;

    x1 = std::max(0.0f, std::min(x1, (float)imgW - 1));
    y1 = std::max(0.0f, std::min(y1, (float)imgH - 1));
    x2 = std::max(0.0f, std::min(x2, (float)imgW - 1));
    y2 = std::max(0.0f, std::min(y2, (float)imgH - 1));

    return cv::Rect(cv::Point((int)x1, (int)y1), cv::Point((int)x2, (int)y2));
}

// NMS（按类别分别执行）
void nms(std::vector<Detection> &dets, float iouThreshold = 0.45f) {
    std::sort(dets.begin(), dets.end(),
              [](const Detection &a, const Detection &b) {
                  return a.score > b.score;
              });

    std::vector<Detection> result;
    std::vector<bool> suppressed(dets.size(), false);

    for (size_t i = 0; i < dets.size(); ++i) {
        if (suppressed[i])
            continue;
        result.push_back(dets[i]);
        for (size_t j = i + 1; j < dets.size(); ++j) {
            if (suppressed[j])
                continue;
            if (dets[i].classId != dets[j].classId)
                continue;

            float inter = (dets[i].box & dets[j].box).area();
            float uni = dets[i].box.area() + dets[j].box.area() - inter;
            float iou = inter / uni;
            if (iou > iouThreshold)
                suppressed[j] = true;
        }
    }
    dets = std::move(result);
}

// 解码 YOLOv8 输出
std::vector<Detection> decodeYOLOv8(const float *output, int numChannels,
                                    int numCandidates,
                                    const LetterboxInfo &info, int imgW,
                                    int imgH, float confThreshold = 0.25f,
                                    int numClasses = 18) {

    std::vector<Detection> dets;
    for (int i = 0; i < numCandidates; ++i) {
        // 找最大类别分数
        float maxScore = 0.0f;
        int maxClass = -1;
        for (int c = 0; c < numClasses; ++c) {
            float s = output[(4 + c) * numCandidates + i];
            if (s > maxScore) {
                maxScore = s;
                maxClass = c;
            }
        }
        if (maxScore < confThreshold)
            continue;

        float cx = output[0 * numCandidates + i];
        float cy = output[1 * numCandidates + i];
        float w = output[2 * numCandidates + i];
        float h = output[3 * numCandidates + i];

        cv::Rect box = scaleBox(cx, cy, w, h, info, imgW, imgH);
        if (box.area() <= 0)
            continue;

        dets.push_back({box, maxScore, maxClass});
    }

    nms(dets);
    return dets;
}
using sclock = std::chrono::steady_clock;

void load_labels(const char *path) {
    ifstream f(path);
    string line;
    while (getline(f, line)) {
        labels.push_back(line);
    }
}
int main() {
    load_labels("../res/models/yolov3_mobilenet_v1/label_list.txt");
    // ---------- 1. 初始化 ONNX Runtime ----------
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "YOLOv8s");
    Ort::SessionOptions sessionOptions;

    // Cortex-A55 是纯 CPU，建议 intra_op 从 4 开始 benchmark，不要直接拉满 8
    sessionOptions.SetIntraOpNumThreads(8);
    sessionOptions.SetInterOpNumThreads(1);
    sessionOptions.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_ALL);

    // 如果 ONNX Runtime 编译时启用了 XNNPACK，可以进一步启用
    // sessionOptions.AppendExecutionProvider("XNNPACK");

    const char *modelPath = "../yolov8s.onnx";
    Ort::Session session(env, modelPath, sessionOptions);

    // 动态获取输入输出名称
    Ort::AllocatorWithDefaultOptions allocator;
    auto inputName = session.GetInputNameAllocated(0, allocator);
    auto outputName = session.GetOutputNameAllocated(0, allocator);
    const char *inputNames[] = {inputName.get()};
    const char *outputNames[] = {outputName.get()};

    // 输入形状固定为 [1, 3, 320, 320]
    std::array<int64_t, 4> inputShape{1, 3, 320, 320};

    // ---------- 2. 摄像头循环 ----------
    Camera camera;
    cv::Mat img;

    while (true) {
        if (!camera.read(img) || img.empty()) {
            std::cerr << "Camera read failed" << std::endl;
            break;
        }

        // ---------- 2.1 预处理 ----------
        std::vector<float> inputTensorValues;
        LetterboxInfo lbInfo = letterbox(img, inputTensorValues);

        // 创建输入 Tensor（CPU 内存，直接引用 vector 数据，零拷贝）
        auto memoryInfo =
            Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
            memoryInfo, inputTensorValues.data(), inputTensorValues.size(),
            inputShape.data(), inputShape.size());

        // ---------- 2.2 推理 ----------
        auto timepoint_bf = sclock::now();

        auto outputs = session.Run(Ort::RunOptions{nullptr}, inputNames,
                                   &inputTensor, 1, outputNames, 1);

        // ---------- 2.3 后处理 ----------
        auto &outputTensor = outputs[0];
        auto outputShape = outputTensor.GetTensorTypeAndShapeInfo().GetShape();
        // YOLOv8 输出形状: [1, 4+numClasses, numCandidates]
        int numChannels = (int)outputShape[1];   // 例如 22
        int numCandidates = (int)outputShape[2]; // 例如 8400
        const float *outputData = outputTensor.GetTensorData<float>();

        int numClasses = numChannels - 4; // 你的场景为 18
        std::vector<Detection> detections =
            decodeYOLOv8(outputData, numChannels, numCandidates, lbInfo,
                         img.cols, img.rows, 0.25f, numClasses);

        auto timepoint_aft = sclock::now();
        auto microsec =
            duration_cast<microseconds>(timepoint_aft - timepoint_bf).count();
        printf("推理用时: %.2lfms，即%.2lf ps\n", microsec / 1000.0,
               1000000.0 / microsec);

        // ---------- 2.4 转换为 PredictResult 并绘制 ----------
        std::vector<PredictResult> results;
        for (const auto &det : detections) {
            PredictResult r;
            r.type = det.classId;
            r.score = det.score;
            r.x = det.box.x;
            r.y = det.box.y; // 左上角 y
            r.width = det.box.width;
            r.height = det.box.height;
            results.push_back(r);
        }

        drawBox(img, results);

        // 显示结果（如果开发板有显示输出；无显示时可注释）
        cv::imshow("YOLOv8s", img);
        if (cv::waitKey(1) == 27)
            break; // ESC 退出
    }

    return 0;
}
