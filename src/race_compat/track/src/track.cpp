#include "track.hpp"
#include <algorithm>
#include <utility>
#include <stdexcept>

using namespace cv;
using std::abs;
using std::make_pair;
using std::pair;
using std::to_string;
using std::vector;

/**
 * @brief 赛道线识别
 *
 * @param imageBinary 赛道识别基准图像
 */
void Track::handle(Mat img) {
    track_img = img;
    handle(false, 0);
}

/**
 * @brief 边缘斜率计算
 *
 * @param edge
 * @param index
 */
void slopeCal(vector<PointX> &edge, int index) {
    if (index < 4) {
        return;
    }
    float slop1 = 0.0, slop2 = 0.0;
    bool has_slop1 = false, has_slop2 = false;

    auto dx1 = edge[index].x - edge[index - 2].x;
    auto dy1 = edge[index].y - edge[index - 2].y;
    if (dx1 != 0) {
        slop1 = (float)dy1 / dx1;
        has_slop1 = true;
    }

    auto dx2 = edge[index].x - edge[index - 4].x;
    auto dy2 = edge[index].y - edge[index - 4].y;
    if (dx2 != 0) {
        slop2 = (float)dy2 / dx2;
        has_slop2 = true;
    }

    if (has_slop1 && has_slop2) {
        edge[index].slope = (slop1 + slop2) / 2;
    } else if (has_slop1) {
        edge[index].slope = slop1;
    } else if (has_slop2) {
        edge[index].slope = slop2;
    } else {
        edge[index].slope = dy1 > 0 ? 255 : -255;
    }
}

void find_white_blocks(const Mat &track_img, int row,
                       vector<pair<int, int>> &blocks) {
    blocks.resize(0);
    bool found_white = false;
    int white_col_start;
    for (int c = 0; c < track_img.cols; ++c) {
        if (track_img.at<uint8_t>(row, c) > 127) {
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
        blocks.push_back(make_pair(white_col_start, track_img.cols - 1));
    }
}

void find_connected_blocks(const vector<pair<int, int>> blocks,
                           const int last_left, const int last_right,
                           vector<pair<int, int>> &connected_blocks) {
    connected_blocks.resize(0);
    for (const auto &block : blocks) { // 上下行色块的连通性判断
        int g_cover = std::min(block.second, last_right) -
                      std::max(block.first, last_left);
        if (g_cover >= 0) {
            connected_blocks.push_back(block);
        }
    }
}

bool qualify_as_first_row(const Mat &track_img, int r,
                          const vector<pair<int, int>> &blocks,
                          pair<int, int> &widest_block_out) {
    auto widest_block = std::max_element(
        blocks.begin(), blocks.end(), [](const auto &a, const auto &b) {
            return (a.second - a.first) < (b.second - b.first);
        });
    widest_block_out = *widest_block;
    const int block_width_requirement =
        (r < track_img.rows * 3 / 4)
            ? (track_img.cols / 2)
            : ((track_img.cols - (track_img.rows - r)) * 13 /
               20); // 首行色块宽度限制（不能太小）
    return (widest_block->second - widest_block->first + 1) >=
           block_width_requirement;
}
pair<int, int>
selectClosestBlock(const vector<pair<int, int>> &connected_blocks,
                   int last_left, int last_right) {
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
 * @brief 赛道线识别
 *
 * @param isResearch 是否重复搜索
 * @param rowStart 边缘搜索起始行
 */
void Track::handle(bool isResearch, uint16_t rowStart) {
    bool is_searching_first_row = true; // 搜索到色块起始行的标志（行）
    PointX pointSpurroad;               // 岔路坐标
    bool spurroadEnabled = false;

    if (track_img.empty() || track_img.type() != CV_8UC1 ||
        track_img.rows != IMAGE_ROW || track_img.cols != IMAGE_COL) {
        pointsEdgeLeft.clear(); pointsEdgeRight.clear();
        widthBlock.clear(); spurroad.clear();
        stdevLeft = stdevRight = 1000;
        throw std::invalid_argument("Track requires a 320x240 CV_8UC1 image");
    }
    rowCutUp = std::clamp(rowCutUp, 0, track_img.rows / 4);
    rowCutBottom = std::clamp(rowCutBottom, 0, track_img.rows / 4);

    if (!isResearch) {
        pointsEdgeLeft.clear();        // 初始化边缘结果
        pointsEdgeRight.clear();       // 初始化边缘结果
        widthBlock.clear();            // 初始化色块数据
        spurroad.clear();              // 岔路信息
        validRowsLeft = 0;             // 边缘有效行数（左）
        validRowsRight = 0;            // 边缘有效行数（右）
        is_searching_first_row = true; // 搜索到色块起始行的标志（行）
        rowStart = track_img.rows - 1 - rowCutBottom;
    } else {
        // 此重载参数表示保留的边界点数量，不是图像行号。
        const size_t keep = std::min({size_t(rowStart), pointsEdgeLeft.size(),
                                     pointsEdgeRight.size(), widthBlock.size()});
        if (keep == 0) { handle(false, 0); return; }
        pointsEdgeLeft.resize(keep);
        pointsEdgeRight.resize(keep);
        widthBlock.resize(keep);
        const int next = widthBlock.back().x - 2;
        if (next <= rowCutUp) return;
        rowStart = std::clamp(next, rowCutUp + 1, track_img.rows - 1);

        is_searching_first_row = false; // 搜索到色块起始行的标志（行）
    }

    auto append_edge = [&](int row, int left, int right, bool compute_slope) {
        pointsEdgeLeft.emplace_back(row, left);
        pointsEdgeRight.emplace_back(row, right);
        widthBlock.emplace_back(row, right - left + 1);
        if (compute_slope) {
            slopeCal(pointsEdgeLeft, pointsEdgeLeft.size() - 1);
            slopeCal(pointsEdgeRight, pointsEdgeRight.size() - 1);
        }
    };

    vector<pair<int, int>> blocks, connected_blocks;
    int missingRows = 0;
    for (int r = rowStart; r > rowCutUp; r -= 2) { // 有效行：10~220
        //  开始识别赛道左右边缘
        find_white_blocks(track_img, r, blocks);
        if (blocks.empty()) {
            // 只允许跨过少量连续缺失的扫描行，不跨越大范围断开的路面。
            if (!is_searching_first_row && ++missingRows > 3) break;
            continue;
        }
        missingRows = 0;

        if (is_searching_first_row) {
            if (r < track_img.rows / 3)
                break;

            pair<int, int> widest_block;
            if (qualify_as_first_row(track_img, r, blocks,
                                     widest_block)) { // 满足首行宽度要求

                append_edge(r, widest_block.first, widest_block.second, false);
                is_searching_first_row = false;
                spurroadEnabled = false;
            }
            continue;
        }
        // 其它行色块坐标处理
        const auto last_left = pointsEdgeLeft.back().y;
        const auto last_right = pointsEdgeRight.back().y;
        find_connected_blocks(blocks, last_left, last_right, connected_blocks);

        if (connected_blocks.size() == 0)
            break; // 如果没有发现联通色块，则图像搜索完成，结束任务

        if (connected_blocks.size() == 1) {
            // 只存在单个色块，正常情况，提取边缘信息
            const auto [left, right] = connected_blocks.front();

            if (right - left < track_img.cols / 20) {
                break;
            }

            append_edge(r, left, right, true);
            spurroadEnabled = false;
            continue;
        }

        if (connected_blocks.size() >= 2) {
            // 存在多个色块，则需要择优处理：选取与上一行最近的色块
            const auto [left, right] =
                selectClosestBlock(connected_blocks, last_left, last_right);
            if (right - left < track_img.cols / 10)
                continue;
            append_edge(r, left, right, true);

            //-------------------------------<岔路信息提取>----------------------------------------
            if (!spurroadEnabled) {
                spurroad.emplace_back(r, connected_blocks[0].second);
                spurroadEnabled = true;
            }
            //------------------------------------------------------------------------------------
        }
    }

    stdevLeft = stdevEdgeCal(pointsEdgeLeft, IMAGE_ROW); // 计算边缘方差
    stdevRight = stdevEdgeCal(pointsEdgeRight, IMAGE_ROW);
}

/**
 * @brief 显示赛道线识别结果
 *
 * @param img 需要叠加显示的图像
 */
void Track::drawImage(Mat &img) {
    for (int i = 0; i < pointsEdgeLeft.size(); i++) {
        circle(img, Point(pointsEdgeLeft[i].y, pointsEdgeLeft[i].x), 2,
               Scalar(0, 255, 0), -1); // 绿色点
    }
    for (int i = 0; i < pointsEdgeRight.size(); i++) {
        circle(img, Point(pointsEdgeRight[i].y, pointsEdgeRight[i].x), 2,
               Scalar(0, 255, 255), -1); // 黄色点
    }

    for (int i = 0; i < spurroad.size(); i++) {
        circle(img, Point(spurroad[i].y, spurroad[i].x), 3, Scalar(0, 0, 255),
               -1); // 红色点
    }

    putText(img, to_string(validRowsRight) + " " + to_string(stdevRight),
            Point(IMAGE_COL - 100, IMAGE_ROW - 50), FONT_HERSHEY_TRIPLEX, 0.3,
            Scalar(0, 0, 255), 1, LINE_AA);
    putText(img, to_string(validRowsLeft) + " " + to_string(stdevLeft),
            Point(20, IMAGE_ROW - 50), FONT_HERSHEY_TRIPLEX, 0.3,
            Scalar(0, 0, 255), 1, LINE_AA);
}

/**
 * @brief 边缘斜率计算
 *
 * @param v_edge
 * @param img_height
 * @return double
 */
double stdevEdgeCal(const vector<PointX> &v_edge, int img_height) {
    const size_t n = v_edge.size();
    if (v_edge.size() < img_height / 4) {
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
