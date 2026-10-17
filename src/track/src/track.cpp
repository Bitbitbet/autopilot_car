#include "track.hpp"
#include <algorithm>
#include <cmath>
#include <utility>

using namespace cv;
using std::pair;
using std::vector;

constexpr double TRACK_MIN_DISTANCE = 0;
constexpr double TRACK_MAX_DISTANCE = 1.2;
constexpr double TRACK_MIN_WIDTH = 0.2;
constexpr double TRACK_MAX_GAP_DISTANCE = 0.05;

namespace {

using Blocks = vector<pair<int, int>>; // 一行里的白色色块（起止列）

/**
 * @brief 找出第 row 行的所有白色色块
 */
void find_white_blocks(const Mat &binary, int row, Blocks &blocks) {
    blocks.clear();
    bool found_white = false;
    int white_col_start = 0;
    for (int c = 0; c < binary.cols; ++c) {
        if (binary.at<uint8_t>(row, c) > 127) {
            if (!found_white) {
                found_white = true;
                white_col_start = c;
            }
        } else if (found_white) {
            found_white = false;
            blocks.emplace_back(white_col_start, c - 1);
        }
    }
    if (found_white)
        blocks.emplace_back(white_col_start, binary.cols - 1);
}

/**
 * @brief 和上一行的边缘区间有重叠的色块，才算同一块路面
 */
void find_connected_blocks(const Blocks &blocks, int last_left, int last_right,
                           Blocks &connected_blocks) {
    connected_blocks.clear();
    for (const auto &block : blocks) {
        const int cover = std::min(block.second, last_right) -
                          std::max(block.first, last_left);
        if (cover >= 0)
            connected_blocks.push_back(block);
    }
}

/**
 * @brief 多个色块时，取中心与上一行最近的那个
 */
pair<int, int> selectClosestBlock(const Blocks &connected_blocks, int last_left,
                                  int last_right) {
    const int prevCenter = last_left + last_right;
    size_t bestIdx = 0;
    int bestDist = std::abs(connected_blocks[0].first +
                            connected_blocks[0].second - prevCenter);
    for (size_t i = 1; i < connected_blocks.size(); ++i) {
        const int dist = std::abs(connected_blocks[i].first +
                                  connected_blocks[i].second - prevCenter);
        if (dist < bestDist) {
            bestDist = dist;
            bestIdx = i;
        }
    }
    return connected_blocks[bestIdx];
}

} // namespace

void track_identify_from_bin(const Mat &bin_img, Camera &camera,
                             vector<Point2i> &left_edge_points,
                             vector<Point2i> &right_edge_points) {
    left_edge_points.clear();
    right_edge_points.clear();

    // 图像坐标(列,行) -> 地面坐标(米)，X 右正左负、Y 前正后负
    const auto ground = [&](int row, int col) {
        return camera.image_to_ground({float(col), float(row)});
    };
    // 行中心处的纵向距离：同一行上各列的距离差可以忽略
    const auto distanceAt = [&](int row) {
        return float(ground(row, bin_img.cols / 2).y);
    };
    // 同一行上左右边界的实际横向间距，就是色块在路面上的宽度
    const auto widthAt = [&](int row, int left, int right) {
        return float(std::fabs(ground(row, right).x - ground(row, left).x));
    };

    bool isSearchingFirstRow = true; // 还没找到最下面那行有效路面
    float lastDistance = 0;

    auto appendEdge = [&](int row, int left, int right) {
        left_edge_points.emplace_back(row, left);
        right_edge_points.emplace_back(row, right);
    };

    Blocks blocks, connected_blocks;
    for (int r = bin_img.rows - 1; r > 0; r -= 2) {
        const float distance = distanceAt(r);
        // 单应缺失、或者已经越过最远搜索距离，就没有可用的地面信息了
        if (!std::isfinite(distance) || distance > TRACK_MAX_DISTANCE)
            break;
        if (distance < TRACK_MIN_DISTANCE)
            continue;

        //  开始识别赛道左右边缘
        find_white_blocks(bin_img, r, blocks);
        if (blocks.empty()) {
            // 只桥接窄的缺失带，缺口超过 TRACK_MAX_GAP_DISTANCE 说明赛道结束了
            if (!isSearchingFirstRow &&
                distance - lastDistance > TRACK_MAX_GAP_DISTANCE)
                break;
            continue;
        }

        if (isSearchingFirstRow) {
            const auto widest = std::max_element(
                blocks.begin(), blocks.end(), [](const auto &a, const auto &b) {
                    return (a.second - a.first) < (b.second - b.first);
                });
            if (widthAt(r, widest->first, widest->second) < TRACK_MIN_WIDTH)
                continue; // 还不够宽，继续往上找起始行
            appendEdge(r, widest->first, widest->second);
            isSearchingFirstRow = false;
            lastDistance = distance;
            continue;
        }

        // 其它行色块坐标处理
        const int last_left = left_edge_points.back().y;
        const int last_right = right_edge_points.back().y;
        find_connected_blocks(blocks, last_left, last_right, connected_blocks);
        if (connected_blocks.empty())
            break; // 没有联通色块，图像搜索完成

        if (connected_blocks.size() == 1) {
            // 只存在单个色块，正常情况，提取边缘信息
            const auto [left, right] = connected_blocks.front();
            if (widthAt(r, left, right) < TRACK_MIN_WIDTH)
                break; // 路面已经收窄到不可信，结束搜索
            appendEdge(r, left, right);
            lastDistance = distance;
            continue;
        }

        // 存在多个色块，则需要择优处理：选取与上一行最近的色块
        const auto [left, right] =
            selectClosestBlock(connected_blocks, last_left, last_right);
        if (widthAt(r, left, right) < TRACK_MIN_WIDTH)
            continue;
        appendEdge(r, left, right);
        lastDistance = distance;
    }
}

void track_draw_result(const vector<Point2i> &left_edge_points,
                       const vector<Point2i> &right_edge_points, Mat &result) {
    for (const auto &p : left_edge_points)
        circle(result, Point(p.y, p.x), 2, Scalar(0, 255, 0), -1); // 绿色点
    for (const auto &p : right_edge_points)
        circle(result, Point(p.y, p.x), 2, Scalar(0, 255, 255), -1); // 黄色点
}

/**
 * Hint: DEAD CODE
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
