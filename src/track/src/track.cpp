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
bool qualify_as_first_row(const Mat &binary, int r, const Blocks &blocks,
                          pair<int, int> &widest_block_out);
pair<int, int> selectClosestBlock(const Blocks &connected_blocks, int last_left,
                                  int last_right);

/**
 * @brief 赛道线识别
 *
 * 二值图自下而上逐行搜索，行号 0 在图像顶部（远处），越大越靠近车头。
 *
 * @param camera 相机（暂未参与识别）
 * @param img    赛道识别基准图像（CV_8UC1 二值图）
 */
void Track::handle([[maybe_unused]] Camera &camera, Mat img) {
    rowCutUp = std::min(img.rows / 4, rowCutUp);
    rowCutBottom = std::min(img.rows / 4, rowCutBottom);

    pointsEdgeLeft.clear();  // 初始化边缘结果
    pointsEdgeRight.clear(); // 初始化边缘结果
    widthBlock.clear();      // 初始化色块数据
    spurroad.clear();        // 岔路信息
    validRowsLeft = 0;       // 边缘有效行数（左）
    validRowsRight = 0;      // 边缘有效行数（右）

    bool isSearchingFirstRow = true; // 搜索到色块起始行的标志（行）
    bool spurroadEnabled = false;
    int missingRows = 0;

    auto appendEdge = [&](int row, int left, int right) {
        pointsEdgeLeft.emplace_back(row, left);
        pointsEdgeRight.emplace_back(row, right);
        widthBlock.emplace_back(row, right - left + 1);
    };

    Blocks blocks, connected_blocks;
    for (int r = img.rows - 1 - rowCutBottom; r > rowCutUp; r -= 2) {
        //  开始识别赛道左右边缘
        find_white_blocks(img, r, blocks);
        if (blocks.empty()) {
            // 只桥接窄的缺失带，大段断开说明赛道已经结束
            if (!isSearchingFirstRow && ++missingRows > 3)
                break;
            continue;
        }
        missingRows = 0;

        if (isSearchingFirstRow) {
            if (r < img.rows / 3)
                break;

            pair<int, int> widest_block;
            if (qualify_as_first_row(img, r, blocks,
                                     widest_block)) { // 满足首行宽度要求
                appendEdge(r, widest_block.first, widest_block.second);
                isSearchingFirstRow = false;
                spurroadEnabled = false;
            }
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

            if (right - left < img.cols / 20) {
                break;
            }

            appendEdge(r, left, right);
            spurroadEnabled = false;
            continue;
        }

        // 存在多个色块，则需要择优处理：选取与上一行最近的色块
        const auto [left, right] =
            selectClosestBlock(connected_blocks, last_left, last_right);
        if (right - left < img.cols / 10)
            continue;
        appendEdge(r, left, right);

        //-------------------------------<岔路信息提取>----------------------------------------
        if (!spurroadEnabled) {
            spurroad.emplace_back(r, connected_blocks[0].second);
            spurroadEnabled = true;
        }
        //------------------------------------------------------------------------------------
    }

    stdevLeft = stdevEdgeCal(pointsEdgeLeft, img.rows); // 计算边缘方差
    stdevRight = stdevEdgeCal(pointsEdgeRight, img.rows);
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

bool qualify_as_first_row(const Mat &binary, int r, const Blocks &blocks,
                          pair<int, int> &widest_block_out) {
    auto widest_block = std::max_element(
        blocks.begin(), blocks.end(), [](const auto &a, const auto &b) {
            return (a.second - a.first) < (b.second - b.first);
        });
    widest_block_out = *widest_block;
    const int block_width_requirement =
        (r < binary.rows * 3 / 4)
            ? (binary.cols / 2)
            : ((binary.cols - (binary.rows - r)) * 13 /
               20); // 首行色块宽度限制（不能太小）
    return (widest_block->second - widest_block->first + 1) >=
           block_width_requirement;
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

    putText(img, to_string(validRowsRight) + " " + to_string(stdevRight),
            Point(img.cols - 100, img.rows - 50), FONT_HERSHEY_TRIPLEX, 0.3,
            Scalar(0, 0, 255), 1, LINE_AA);
    putText(img, to_string(validRowsLeft) + " " + to_string(stdevLeft),
            Point(20, img.rows - 50), FONT_HERSHEY_TRIPLEX, 0.3,
            Scalar(0, 0, 255), 1, LINE_AA);
}

/**
 * @brief 边缘斜率方差计算
 *
 * @param v_edge
 * @param img_height
 * @return double
 */
double stdevEdgeCal(const vector<Point2i> &v_edge, int img_height) {
    const size_t n = v_edge.size();
    if (v_edge.size() < size_t(img_height) / 4) {
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
