#include "track.hpp"
#include <algorithm>
#include <cmath>
#include <utility>

using namespace cv;
using std::abs;
using std::make_pair;
using std::pair;
using std::to_string;
using std::vector;

using Blocks = vector<pair<int, int>>;

// 边缘识别内部辅助函数
void find_white_blocks(const Mat &binary, int row, Blocks &blocks);
void find_connected_blocks(const Blocks blocks, int last_left, int last_right,
                           Blocks &connected_blocks);
pair<int, int> selectClosestBlock(const Blocks &connected_blocks, int last_left,
                                  int last_right);

/**
 * @brief 赛道线识别
 *
 * 二值图自下而上逐行搜索，行号 0 在图像顶部（远处），越大越靠近车头。
 * 搜索范围、色块宽度、缺口大小都用 camera 的单应换算成地面坐标（米）判断，
 * 所以换分辨率/换相机只需要重标 calibration.xml，不用改这里的数字。
 *
 * @param camera 相机（提供地面单应，见 calibration.xml 的 ground_H_img2gnd）
 * @param img    赛道识别基准图像（CV_8UC1 二值图）
 */
void Track::handle(Camera &camera, Mat img) {
    // 图像坐标(列,行) -> 地面坐标(米)，X 右正左负、Y 前正后负
    const auto groundX = [&](int row, int col) {
        return float(camera.image_to_ground({float(col), float(row)}).x);
    };
    // 行中心处的纵向距离：同一行上各列的距离差可以忽略
    const auto distanceAt = [&](int row) {
        return float(
            camera.image_to_ground({float(img.cols) / 2, float(row)}).y);
    };
    // 同一行上左右边界的实际横向间距，就是色块在路面上的宽度
    const auto widthAt = [&](int row, int left, int right) {
        return std::fabs(groundX(row, right) - groundX(row, left));
    };

    pointsEdgeLeft.clear();  // 初始化边缘结果
    pointsEdgeRight.clear(); // 初始化边缘结果
    widthBlock.clear();      // 初始化色块数据
    spurroad.clear();        // 岔路信息

    bool isSearchingFirstRow = true; // 搜索到色块起始行的标志（行）
    bool spurroadEnabled = false;
    float lastDistance = 0;

    auto appendEdge = [&](int row, int left, int right) {
        pointsEdgeLeft.emplace_back(row, left);
        pointsEdgeRight.emplace_back(row, right);
        widthBlock.emplace_back(row, right - left + 1);
    };

    Blocks blocks, connected_blocks;
    for (int r = img.rows - 1; r > 0; r -= 2) {
        const float distance = distanceAt(r);
        // 单应缺失、或者已经越过最远搜索距离，就没有可用的地面信息了
        if (!std::isfinite(distance) || distance > maxDistance)
            break;
        if (distance < minDistance)
            continue;

        //  开始识别赛道左右边缘
        find_white_blocks(img, r, blocks);
        if (blocks.empty()) {
            // 只桥接窄的缺失带，缺口超过 maxGapDistance 说明赛道已经结束
            if (!isSearchingFirstRow &&
                distance - lastDistance > maxGapDistance)
                break;
            continue;
        }

        if (isSearchingFirstRow) {
            const auto widest = max_element(
                blocks.begin(), blocks.end(), [](const auto &a, const auto &b) {
                    return (a.second - a.first) < (b.second - b.first);
                });
            if (widthAt(r, widest->first, widest->second) < minRoadWidth)
                continue; // 还不够宽，继续往上找起始行
            appendEdge(r, widest->first, widest->second);
            isSearchingFirstRow = false;
            spurroadEnabled = false;
            lastDistance = distance;
            continue;
        }
        // 其它行色块坐标处理
        const auto last_left = pointsEdgeLeft.back().y;
        const auto last_right = pointsEdgeRight.back().y;
        find_connected_blocks(blocks, last_left, last_right, connected_blocks);

        if (connected_blocks.empty())
            break; // 如果没有发现联通色块，则图像搜索完成，结束任务

        if (connected_blocks.size() == 1) {
            // 只存在单个色块，正常情况，提取边缘信息
            const auto [left, right] = connected_blocks.front();

            if (widthAt(r, left, right) < minRoadWidth) {
                break; // 路面已经收窄到不可信，结束搜索
            }

            appendEdge(r, left, right);
            spurroadEnabled = false;
            lastDistance = distance;
            continue;
        }

        // 存在多个色块，则需要择优处理：选取与上一行最近的色块
        const auto [left, right] =
            selectClosestBlock(connected_blocks, last_left, last_right);
        if (widthAt(r, left, right) < minRoadWidth)
            continue;
        appendEdge(r, left, right);
        lastDistance = distance;

        //-------------------------------<岔路信息提取>----------------------------------------
        if (!spurroadEnabled) {
            spurroad.emplace_back(r, connected_blocks[0].second);
            spurroadEnabled = true;
        }
        //------------------------------------------------------------------------------------
    }

    stdevLeft = stdevEdgeCal(pointsEdgeLeft); // 计算边缘方差
    stdevRight = stdevEdgeCal(pointsEdgeRight);
}

void find_white_blocks(const Mat &binary, int row, Blocks &blocks) {
    blocks.resize(0);
    bool found_white = false;
    int white_col_start = 0;
    for (int c = 0; c < binary.cols; ++c) {
        if (binary.at<uint8_t>(row, c) > 127) {
            if (!found_white) {
                found_white = true;
                white_col_start = c;
            }
        } else {
            if (found_white) {
                found_white = false;
                blocks.push_back(make_pair(white_col_start, c - 1));
            }
        }
    }
    if (found_white) {
        blocks.push_back(make_pair(white_col_start, binary.cols - 1));
    }
}

void find_connected_blocks(const Blocks blocks, const int last_left,
                           const int last_right, Blocks &connected_blocks) {
    connected_blocks.resize(0);
    for (const auto &block : blocks) { // 上下行色块的连通性判断
        int g_cover = std::min(block.second, last_right) -
                      std::max(block.first, last_left);
        if (g_cover >= 0) {
            connected_blocks.push_back(block);
        }
    }
}

pair<int, int> selectClosestBlock(const Blocks &connected_blocks, int last_left,
                                  int last_right) {
    const int prevCenter = last_left + last_right;
    int bestIdx = 0;
    int bestDist = abs(connected_blocks[bestIdx].first +
                       connected_blocks[bestIdx].second - prevCenter);

    for (size_t i = 1; i < connected_blocks.size(); ++i) {
        const auto dist = abs(connected_blocks[i].first +
                              connected_blocks[i].second - prevCenter);

        if (dist < bestDist) {
            bestDist = dist;
            bestIdx = i;
        }
    }
    return connected_blocks[bestIdx];
}

/**
 * @brief 显示赛道线识别结果
 *
 * @param img 需要叠加显示的图像
 */
void Track::drawImage(Mat &img) {
    for (size_t i = 0; i < pointsEdgeLeft.size(); i++) {
        circle(img, Point(pointsEdgeLeft[i].y, pointsEdgeLeft[i].x), 2,
               Scalar(0, 255, 0), -1); // 绿色点
    }
    for (size_t i = 0; i < pointsEdgeRight.size(); i++) {
        circle(img, Point(pointsEdgeRight[i].y, pointsEdgeRight[i].x), 2,
               Scalar(0, 255, 255), -1); // 黄色点
    }

    for (size_t i = 0; i < spurroad.size(); i++) {
        circle(img, Point(spurroad[i].y, spurroad[i].x), 3, Scalar(0, 0, 255),
               -1); // 红色点
    }

    putText(img, to_string(stdevRight), Point(img.cols - 100, img.rows - 50),
            FONT_HERSHEY_TRIPLEX, 0.3, Scalar(0, 0, 255), 1, LINE_AA);
    putText(img, to_string(stdevLeft), Point(20, img.rows - 50),
            FONT_HERSHEY_TRIPLEX, 0.3, Scalar(0, 0, 255), 1, LINE_AA);
}

/**
 * @brief 边缘斜率方差计算
 *
 * @param v_edge 边缘点集（x=行, y=列）
 * @return double 点太少时返回 1000 表示未知
 */
double stdevEdgeCal(const vector<Point2i> &v_edge) {
    const size_t n = v_edge.size();
    if (n < 10) {
        return 1000;
    }
    double mean = 0.0, m2 = 0.0;
    const auto STEP = std::max<size_t>(v_edge.size() / 5, 1);
    size_t k = 0;
    for (size_t i = STEP; i < n; i += STEP) {
        const int dx = v_edge[i].x - v_edge[i - STEP].x;
        if (dx == 0)
            continue;
        const double s =
            static_cast<double>(v_edge[i].y - v_edge[i - STEP].y) * 100.0 / dx;
        ++k;
        const double delta = s - mean;
        mean += delta / k;
        m2 += delta * (s - mean);
    }
    if (k < 2)
        return 0.0;
    return std::sqrt(m2 / (k - 1));
}
