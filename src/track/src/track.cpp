#include "track.hpp"
#include <algorithm>
#include <cmath>
#include <utility>

using namespace cv;
using std::greater;
using std::map;
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

} // namespace

void track_identify_from_bin(const Mat &bin_img, Camera &camera,
                             map<int, pair<int, int>, greater<int>> &rows) {
    track_identify_from_bin(bin_img,[&camera](Point2d p){return camera.image_to_ground(p);},rows);
}

void track_identify_from_bin(const Mat &bin_img,const std::function<Point2d(Point2d)> &project,
                             NativeTrackRows &rows) {
    rows.clear();

    // 行中心处的纵向距离：同一行上各列的距离差可以忽略
    const auto distanceAt = [&](int row) {
        return float(
            project({bin_img.cols / 2.0, double(row)}).y);
    };
    // 同一行上左右边界的实际横向间距，就是色块在路面上的宽度
    const auto widthAt = [&](int row, int left, int right) {
        return float(
            fabs(project({double(right), double(row)}).x -
                 project({double(left), double(row)}).x));
    };

    bool isSearchingFirstRow = true; // 还没找到最下面那行有效路面
    float lastDistance = 0;

    auto appendEdge = [&](int row, int left, int right) {
        rows.insert({row, {left, right}});
    };

    const bool metric=std::isfinite(distanceAt(bin_img.rows-1));
    Blocks blocks, connected_blocks;
    for (auto r = bin_img.rows - 1; r > 0; r -= 2) {
        const float distance = distanceAt(r);
        // 单应缺失、或者已经越过最远搜索距离，就没有可用的地面信息了
        if (metric&&(!std::isfinite(distance) || distance > TRACK_MAX_DISTANCE))
            break;
        if (metric&&distance < TRACK_MIN_DISTANCE)
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
            // 白色标记会把路面切成几块，取并集；还不够宽就继续往上找起始行
            const int left = blocks.front().first;
            const int right = blocks.back().second;
            if (metric&&widthAt(r, left, right) < TRACK_MIN_WIDTH)
                continue;
            appendEdge(r, left, right);
            isSearchingFirstRow = false;
            lastDistance = distance;
            continue;
        }

        // 其它行色块坐标处理
        auto last_row = *rows.rbegin();
        const int last_left = last_row.second.first;
        const int last_right = last_row.second.second;
        find_connected_blocks(blocks, last_left, last_right, connected_blocks);
        if (connected_blocks.empty())
            break; // 没有联通色块，图像搜索完成

        // 白色标记会把路面切成几块：取并集，别把标记边缘当路面边缘
        const int left = connected_blocks.front().first;
        const int right = connected_blocks.back().second;
        if (metric&&widthAt(r, left, right) < TRACK_MIN_WIDTH)
            break; // 路面已经收窄到不可信，结束搜索
        appendEdge(r, left, right);
        lastDistance = distance;
    }
}

void track_draw_result(const map<int, pair<int, int>, greater<int>> &rows,
                       Mat &result) {
    for (auto [row, col] : rows) {
        auto [left, right] = col;
        circle(result, Point(left, row), 2, Scalar(0, 255, 0), -1);    // 绿色点
        circle(result, Point(right, row), 2, Scalar(0, 255, 255), -1); // 黄色点
    }
}
