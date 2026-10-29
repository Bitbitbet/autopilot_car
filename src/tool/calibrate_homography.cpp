/**
 * @file calibrate_homography.cpp
 * @brief 地面单应（俯视标定）交互式工具
 *
 * 用途：在地面上铺一张 3x3 的靶纸（横向 ±0.1m，纵向 0.14 / 0.08 / 0 m），
 *       依次点出 9 个点，求出「图像 <-> 地面(米)」的单应矩阵 H 并存盘。
 *
 * 世界坐标系（与项目其它部分一致，固定 640x480 采集）：
 *   X：垂直于前进方向，左负右正（单位：米）
 *   Y：前进方向，前正后负（单位：米）
 *
 * 关于原点（重要）：
 *   靶纸上那 9 个世界坐标只是**相对网格**，用来提供尺度与朝向；靶纸上标着
 *   (0,0) 的那个点只是靶纸自身的参考点，它**不必**、通常也**不会**落在锚点
 *   像素上。世界原点由锚点像素独立决定：本工具把图像 (320,440) 定为世界
 *   (0,0)。实现方式是先用 9 个点稳健拟合相对网格的单应，再在世界系里平移，
 *   把原点搬到"锚点像素正下方那个地面点"，因此最终结果精确满足
 *   图像(320,440) <-> 世界(0,0)。靶纸随便放，但放在 (320,440) 视线附近
 *   可以减小外推误差。
 *
 * 操作：
 *   鼠标左键  依次点出 9 个点（屏幕上有提示当前该点第几个、相对坐标是多少）
 *   r         重新点（清空）
 *   c         重新拟合（点数 >= 4 时可用；点满 9 个自动拟合）
 *   a         开关"把世界原点锚定到 (320,440)"（默认开）
 *   p         打印 行 -> 前方距离 对照表
 *   s         保存：../res/calibration/ground_H.yml + 写进 calibration.xml
 *   q / ESC   退出
 */

#include <opencv2/calib3d.hpp>
#include <opencv2/core/persistence.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>

using cv::FileStorage;
using cv::Mat;
using cv::Point2d;
using cv::Point2f;
using cv::Rect;
using cv::Size;
using cv::VideoCapture;
using std::cerr;
using std::cout;
using std::endl;
using std::max;
using std::min;
using std::string;
using std::vector;

constexpr const char *CALIBRATION_FILENAME =
    "../res/calibration/calibration.xml";
constexpr const char *OUTPUT_FILENAME = "../res/calibration/ground_H.yml";

// 采集分辨率：锚点像素语义与它绑定，必须显式设置
constexpr int FRAME_WIDTH = 640;
constexpr int FRAME_HEIGHT = 480;
// 世界原点对应的图像像素
constexpr float ANCHOR_X = 320.f;
constexpr float ANCHOR_Y = 440.f;

// RANSAC 阈值：目标域是像素，这个单位才有意义
constexpr double RANSAC_THRESH_PX = 3.0;

// 度量俯视图预览（1 格 = GRID_STEP_M 米）
constexpr double PREVIEW_SCALE = 400.0; // px / m
constexpr double PREVIEW_XMIN = -0.6;
constexpr double PREVIEW_XMAX = 0.6;
constexpr double PREVIEW_YMIN = -0.1;
constexpr double PREVIEW_YMAX = 1.3;
constexpr double GRID_STEP_M = 0.1;

bool grab_image(VideoCapture &camera, const Mat &mapx, const Mat &mapy,
                Mat &frame) {
    if (!camera.read(frame)) {
        return false;
    }
    if (frame.empty()) {
        return false;
    }

    remap(frame, frame, mapx, mapy, cv::INTER_LINEAR);

    return true;
}

/* ==========================================================================
 * 单应应用：齐次坐标必须除以第三分量；H 必须是 CV_64F
 * ========================================================================== */
static bool apply_h(const Mat &H, double x, double y, Point2d &out) {
    if (H.empty()) {
        return false;
    }
    const Mat v = (cv::Mat_<double>(3, 1) << x, y, 1.0);
    const Mat r = H * v;
    const double w = r.at<double>(2);
    if (std::fabs(w) < 1e-12) {
        return false; // 地平线附近，映射到无穷远
    }
    out = Point2d(r.at<double>(0) / w, r.at<double>(1) / w);
    return std::isfinite(out.x) && std::isfinite(out.y);
}

/* ==========================================================================
 * 交互状态
 * ========================================================================== */
struct Calib {
    const vector<Point2f> *world = nullptr; // 世界点（米），点击顺序即此顺序
    vector<Point2f> image;                  // 已点击的图像点（同顺序）
    Mat H_g2i;                              // 地面(米) -> 图像(px)
    Mat H_i2g;                              // 图像(px) -> 地面(米)
    bool has_H = false;
    bool enforce_anchor = true;   // 把世界原点锚定到图像(320,440)
    double rmse_px = 0.0;         // 重投影残差（像素）
    int inliers = 0;              // RANSAC 内点数
    double anchor_delta_px = 0.0; // 靶纸参考点与锚点的像素距离
    double anchor_shift_m = 0.0;  // 原点从靶纸参考点搬到锚点的距离(米)
    // 靶纸参考点(标着 0,0 那个)在 world 中的下标，仅用于高亮
    int anchor_index = 8;
    cv::Size frame_size{FRAME_WIDTH, FRAME_HEIGHT}; // 图像尺寸（点击钳位用）
};

static void on_mouse(int event, int x, int y, int, void *userdata) {
    if (event != cv::EVENT_LBUTTONDOWN) {
        return;
    }
    Calib &st = *static_cast<Calib *>(userdata);
    if (st.image.size() >= st.world->size()) {
        cout << "已经点满 " << st.world->size()
             << " 个点，按 r 重来，按 c 重新拟合" << endl;
        return;
    }

    // 窗口里显示的就是矫正后的原图，点击坐标直接可用，只做边界钳位
    const double u = max(0.0, min((double)x, st.frame_size.width - 1.0));
    const double v = max(0.0, min((double)y, st.frame_size.height - 1.0));

    st.image.emplace_back(static_cast<float>(u), static_cast<float>(v));
    const Point2f &w = (*st.world)[st.image.size() - 1];
    printf("[%zu/%zu] 图像(%.1f,%.1f)  <->  世界 X=%.3f Y=%.3f m\n",
           st.image.size(), st.world->size(), u, v, w.x, w.y);
}

/* ==========================================================================
 * 拟合：world -> image（目标域=像素，RANSAC 阈值单位才正确），再取逆
 * ========================================================================== */
static bool fit(Calib &st) {
    const vector<Point2f> &wp = *st.world;
    if (st.image.size() < 4) {
        cout << "点太少（至少 4 个，且不能共线）" << endl;
        return false;
    }

    vector<Point2f> src(wp.begin(), wp.begin() + st.image.size());
    vector<Point2f> dst(st.image.begin(), st.image.end());

    /* ------------------------------------------------------------------
     * 1) 稳健拟合：只用点击点。
     *    不要把锚点"替换"成点击点再送进 RANSAC —— 实测那会因为锚点与其它
     *    点不一致而被判成外点剔除，锚点约束静默失效。锚点改用第 3 步的
     *    世界系平移来精确实现。
     * ------------------------------------------------------------------ */
    st.inliers = 0;
    if (dst.size() == 4) {
        st.H_g2i = cv::findHomography(src, dst, 0); // 4 点：精确解
        st.inliers = 4;
    } else {
        vector<uchar> mask;
        st.H_g2i =
            cv::findHomography(src, dst, cv::RANSAC, RANSAC_THRESH_PX, mask);
        st.inliers = static_cast<int>(
            std::count(mask.begin(), mask.end(), static_cast<uchar>(1)));
        if (!st.H_g2i.empty() && st.inliers < static_cast<int>(dst.size())) {
            printf("RANSAC 内点 %d/%zu，外点(建议重点这些): ", st.inliers,
                   dst.size());
            for (size_t k = 0; k < mask.size(); ++k) {
                if (!mask[k]) {
                    printf("%zu ", k + 1);
                }
            }
            printf("\n");
        }
    }
    if (st.H_g2i.empty()) {
        st.has_H = false;
        cout << "拟合失败：点共线或过于集中，请把点铺开" << endl;
        return false;
    }

    // 2) 重投影 RMSE（衡量点击质量，在锚点平移之前算）
    {
        vector<Point2f> back;
        cv::perspectiveTransform(src, back, st.H_g2i);
        double sse = 0.0;
        for (size_t k = 0; k < back.size(); ++k) {
            sse += cv::norm(back[k] - dst[k]);
        }
        st.rmse_px = std::sqrt(sse / back.size());
    }

    // 3) 锚点约束：把世界原点平移到"锚点像素正下方那个地面点"，
    //    使 世界(0,0) <-> 图像(320,440) 精确成立
    Point2d origin_px(-1, -1);
    apply_h(st.H_g2i, 0.0, 0.0, origin_px);
    st.anchor_delta_px = cv::norm(origin_px - Point2d(ANCHOR_X, ANCHOR_Y));
    st.anchor_shift_m = 0.0;
    Point2d g_anchor(0.0, 0.0);
    if (st.enforce_anchor) {
        const Mat H_click_i2g = st.H_g2i.inv();
        if (apply_h(H_click_i2g, ANCHOR_X, ANCHOR_Y, g_anchor)) {
            // 新世界系: X' = X - g.x, Y' = Y - g.y
            const Mat T = (cv::Mat_<double>(3, 3) << 1, 0, g_anchor.x, 0, 1,
                           g_anchor.y, 0, 0, 1);
            st.H_g2i = st.H_g2i * T;
            st.anchor_shift_m = cv::norm(g_anchor);
        }
    }
    st.H_i2g = st.H_g2i.inv();
    st.has_H = true;

    // 4) 报告
    printf("\n=== 拟合完成 ===\n");
    printf("使用点数=%zu, RANSAC 内点=%d, 重投影 RMSE=%.2f px\n", dst.size(),
           st.inliers, st.rmse_px);
    printf("原点锚定 %s\n", st.enforce_anchor ? "ON" : "OFF");
    if (st.enforce_anchor) {
        printf("  靶纸参考点在新世界系里: X=%+.1f mm(左负右正) "
               "Y=%+.1f mm(前正后负)  <- 它只是相对网格原点，不必与锚点重合\n",
               g_anchor.x * 1000.0, g_anchor.y * 1000.0);
        printf("  世界(0,0) <-> 图像(%.0f,%.0f) 精确成立（原点已按像素锚定）\n",
               ANCHOR_X, ANCHOR_Y);
        // 锚点离靶纸越远，越依赖外推，精度越差
        double dmin = 1e9;
        for (const auto &p : st.image) {
            Point2d gi;
            if (apply_h(st.H_i2g, p.x, p.y, gi)) {
                dmin = std::min(dmin, cv::norm(gi));
            }
        }
        printf("  锚点到最近靶纸点的地面距离 %.0f mm", dmin * 1000.0);
        if (dmin > 0.30) {
            printf("  <- 偏大：外推误差会被放大，建议把靶纸挪到 (320,440) "
                   "视线附近\n");
        } else {
            printf("\n");
        }
    } else {
        printf("  未锚定：世界原点=靶纸参考点，落在图像(%.1f,%.1f)，"
               "与锚点(%.0f,%.0f) 差 %.1f px\n",
               origin_px.x, origin_px.y, ANCHOR_X, ANCHOR_Y,
               st.anchor_delta_px);
        if (origin_px.x < 0 || origin_px.x >= FRAME_WIDTH || origin_px.y < 0 ||
            origin_px.y >= FRAME_HEIGHT) {
            cout << "警告：靶纸参考点落在图像外，请检查点选" << endl;
        }
    }
    // 靶纸越小，点击误差在远场被放大得越厉害
    if (!st.image.empty()) {
        const cv::Rect bb = cv::boundingRect(st.image);
        const double cover =
            std::max(bb.width, bb.height) / static_cast<double>(FRAME_WIDTH);
        printf("靶纸包围盒 %dx%d px (占画面宽 %.0f%%)", bb.width, bb.height,
               cover * 100.0);
        if (cover < 0.25) {
            printf("  <- 偏小：点击误差会被放大数倍\n");
        } else {
            printf("\n");
        }
    }
    return true;
}

/* ==========================================================================
 * 行 -> 前方距离 对照表（就是可以直接拿去替换 IMAGE_ROW*0.35 的那张表）
 * ========================================================================== */
static void print_row_table(const Calib &st) {
    if (!st.has_H) {
        cout << "还没有拟合结果" << endl;
        return;
    }
    printf("\n--- 图像行 -> 前方距离（u=%d 图像中线上）---\n", FRAME_WIDTH / 2);
    printf("%6s %14s %14s\n", "行 v", "前方 Y(m)", "横向 X(m)");
    for (int v = FRAME_HEIGHT - 1; v >= 0; v -= 20) {
        Point2d g;
        if (!apply_h(st.H_i2g, FRAME_WIDTH / 2.0, v, g)) {
            break;
        }
        if (g.y < -1.0 || g.y > 50.0) {
            printf("%6d %14s\n", v, "(地平线以上/无效)");
            break;
        }
        printf("%6d %14.3f %14.3f\n", v, g.y, g.x);
    }
}

/* ==========================================================================
 * 绘制：把地面网格投影回图像（用卷尺校验用）
 * ========================================================================== */
static void project_and_draw_grid(Mat &img, const Calib &st) {
    if (!st.has_H) {
        return;
    }
    const cv::Scalar grid(0, 200, 0);
    const cv::Scalar axis(0, 200, 255);

    auto draw_line = [&](double x0, double y0, double x1, double y1,
                         const cv::Scalar &color, int thick) {
        Point2d a, b;
        if (!apply_h(st.H_g2i, x0, y0, a) || !apply_h(st.H_g2i, x1, y1, b)) {
            return;
        }
        const double lim = 1e5;
        if (std::fabs(a.x) > lim || std::fabs(a.y) > lim ||
            std::fabs(b.x) > lim || std::fabs(b.y) > lim) {
            return;
        }
        cv::line(img, cv::Point(cvRound(a.x), cvRound(a.y)),
                 cv::Point(cvRound(b.x), cvRound(b.y)), color, thick,
                 cv::LINE_AA);
    };

    for (double x = PREVIEW_XMIN; x <= PREVIEW_XMAX + 1e-9; x += GRID_STEP_M) {
        const bool is_axis = std::fabs(x) < 1e-9;
        draw_line(x, PREVIEW_YMIN, x, PREVIEW_YMAX, is_axis ? axis : grid,
                  is_axis ? 2 : 1);
    }
    for (double y = 0.0; y <= PREVIEW_YMAX + 1e-9; y += GRID_STEP_M) {
        const bool is_axis = std::fabs(y) < 1e-9;
        draw_line(PREVIEW_XMIN, y, PREVIEW_XMAX, y, is_axis ? axis : grid,
                  is_axis ? 2 : 1);
    }

    // 每 0.5m 标注一次距离
    for (double y = 0.5; y <= PREVIEW_YMAX; y += 0.5) {
        Point2d p;
        if (apply_h(st.H_g2i, 0.0, y, p) && std::fabs(p.x) < 1e5 &&
            std::fabs(p.y) < 1e5) {
            char buf[32];
            snprintf(buf, sizeof(buf), "%.1fm", y);
            cv::putText(img, buf, cv::Point(cvRound(p.x) + 4, cvRound(p.y)),
                        cv::FONT_HERSHEY_PLAIN, 0.8, axis, 1);
        }
    }
}

/* ==========================================================================
 * 度量俯视图预览：米 -> 画布像素，直观看出 1 米 / 车道宽是多少
 * ========================================================================== */
static void build_preview(const Mat &frame, const Calib &st, Mat &preview) {
    const int cw =
        static_cast<int>((PREVIEW_XMAX - PREVIEW_XMIN) * PREVIEW_SCALE);
    const int ch =
        static_cast<int>((PREVIEW_YMAX - PREVIEW_YMIN) * PREVIEW_SCALE);
    if (!st.has_H) {
        preview = Mat::zeros(ch, cw, CV_8UC3);
        cv::putText(preview, "no H yet", cv::Point(10, 20),
                    cv::FONT_HERSHEY_PLAIN, 1.0, cv::Scalar(0, 0, 255), 1);
        return;
    }

    // 米 -> 画布像素：Y 越大越靠上，所以纵轴取负
    const double ox = -PREVIEW_XMIN * PREVIEW_SCALE;
    const double oy = (PREVIEW_YMAX)*PREVIEW_SCALE;
    Mat S = (cv::Mat_<double>(3, 3) << PREVIEW_SCALE, 0, ox, 0, -PREVIEW_SCALE,
             oy, 0, 0, 1);
    Mat M = S * st.H_i2g; // 图像 -> 画布
    cv::warpPerspective(frame, preview, M, Size(cw, ch), cv::INTER_LINEAR,
                        cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

    // 画网格 + 原点
    for (double x = PREVIEW_XMIN; x <= PREVIEW_XMAX + 1e-9; x += GRID_STEP_M) {
        const int col = cvRound(x * PREVIEW_SCALE + ox);
        cv::line(preview, cv::Point(col, 0), cv::Point(col, ch),
                 cv::Scalar(60, 60, 60), 1);
    }
    for (double y = PREVIEW_YMIN; y <= PREVIEW_YMAX + 1e-9; y += GRID_STEP_M) {
        const int row = cvRound(oy - y * PREVIEW_SCALE);
        cv::line(preview, cv::Point(0, row), cv::Point(cw, row),
                 cv::Scalar(60, 60, 60), 1);
    }
    for (double y = 0.0; y <= PREVIEW_YMAX; y += 0.5) {
        char buf[32];
        snprintf(buf, sizeof(buf), "%.1fm", y);
        const int row = cvRound(oy - y * PREVIEW_SCALE);
        cv::putText(preview, buf, cv::Point(cvRound(ox) + 4, row - 3),
                    cv::FONT_HERSHEY_PLAIN, 0.8, cv::Scalar(0, 200, 255), 1);
    }
    cv::circle(preview, cv::Point(cvRound(ox), cvRound(oy)), 4,
               cv::Scalar(0, 0, 255), -1);
    cv::putText(preview, "origin(0,0)",
                cv::Point(cvRound(ox) + 6, cvRound(oy) + 14),
                cv::FONT_HERSHEY_PLAIN, 0.8, cv::Scalar(0, 0, 255), 1);
}

/* ==========================================================================
 * 主画面叠加：已点/待点提示、锚点、网格
 * ========================================================================== */
static void draw_overlay(Mat &img, const Calib &st) {
    project_and_draw_grid(img, st);

    // 锚点
    cv::drawMarker(img, cv::Point(cvRound(ANCHOR_X), cvRound(ANCHOR_Y)),
                   cv::Scalar(255, 0, 255), cv::MARKER_CROSS, 24, 2);

    const vector<Point2f> &wp = *st.world;
    for (size_t k = 0; k < wp.size(); ++k) {
        if (k < st.image.size()) {
            const cv::Point p(cvRound(st.image[k].x), cvRound(st.image[k].y));
            // 世界原点(最后一个点)用洋红色，方便和锚点十字对照
            const bool is_origin = (static_cast<int>(k) == st.anchor_index);
            cv::circle(img, p, is_origin ? 6 : 4,
                       is_origin ? cv::Scalar(255, 0, 255)
                                 : cv::Scalar(0, 0, 255),
                       -1);
            char buf[48];
            snprintf(buf, sizeof(buf), "%zu", k + 1);
            cv::putText(img, buf, p + cv::Point(5, -5), cv::FONT_HERSHEY_PLAIN,
                        1.0, cv::Scalar(0, 0, 255), 1);
        }
    }

    int y = 18;
    auto line = [&](const string &s, cv::Scalar c) {
        cv::putText(img, s, cv::Point(8, y), cv::FONT_HERSHEY_PLAIN, 1.0, c, 1);
        y += 16;
    };
    line("click points in order 1.." + std::to_string(wp.size()),
         cv::Scalar(255, 255, 0));
    if (st.image.size() < wp.size()) {
        const Point2f &n = wp[st.image.size()];
        char buf[96];
        snprintf(buf, sizeof(buf), "next #%zu : X=%+.3f Y=%.3f m",
                 st.image.size() + 1, n.x, n.y);
        line(buf, cv::Scalar(0, 255, 255));
    } else {
        line("all points done", cv::Scalar(0, 255, 0));
    }
    if (st.has_H) {
        char buf[112];
        snprintf(buf, sizeof(buf), "RMSE=%.2fpx inliers=%d anchor=%s d=%.1fpx",
                 st.rmse_px, st.inliers, st.enforce_anchor ? "ON" : "OFF",
                 st.anchor_delta_px);
        line(buf, st.rmse_px < 2.0 ? cv::Scalar(0, 255, 0)
                                   : cv::Scalar(0, 128, 255));
        if (st.enforce_anchor) {
            snprintf(buf, sizeof(buf), "origin shifted %.1f mm to anchor",
                     st.anchor_shift_m * 1000.0);
            line(buf, cv::Scalar(255, 0, 255));
        }
    }
    line("r:reset  c:fit  a:anchor  p:table  s:save",
         cv::Scalar(200, 200, 200));
}

/* ==========================================================================
 * 存盘
 * ========================================================================== */
static bool save_result(const Calib &st) {
    if (!st.has_H) {
        cout << "还没有拟合结果，不能保存" << endl;
        return false;
    } // 先把已有内参读出来，绝不能把棋盘格标定结果冲掉
    Mat camera_matrix, dist_coeffs;
    {
        FileStorage in(CALIBRATION_FILENAME, FileStorage::READ);
        if (!in.isOpened()) {
            cerr << "无法读取 " << CALIBRATION_FILENAME << "，跳过写入" << endl;
            return false;
        }
        in["cameraMatrix"] >> camera_matrix;
        in["distCoeffs"] >> dist_coeffs;
    }
    if (camera_matrix.empty() || dist_coeffs.empty()) {
        cerr << CALIBRATION_FILENAME
             << " 缺少 cameraMatrix/distCoeffs，为安全起见不重写该文件" << endl;
        return false;
    }

    FileStorage out(CALIBRATION_FILENAME, FileStorage::WRITE);
    if (!out.isOpened()) {
        cerr << "无法写入 " << CALIBRATION_FILENAME << endl;
        return false;
    }
    out << "cameraMatrix" << camera_matrix;
    out << "distCoeffs" << dist_coeffs;
    out.writeComment("ground homography: X right / Y forward, unit meter, "
                     "image 640x480, image(320,440) <-> world(0,0)");
    out << "ground_H_gnd2img" << st.H_g2i;
    out << "ground_H_img2gnd" << st.H_i2g;
    out << "ground_anchor_px" << vector<Point2f>{Point2f(ANCHOR_X, ANCHOR_Y)};
    out << "ground_rmse_px" << st.rmse_px;
    out << "ground_anchor_delta_px" << st.anchor_delta_px;
    out << "ground_anchor_shift_m" << st.anchor_shift_m;
    out.release();
    cout << "已把地面单应写入 " << CALIBRATION_FILENAME
         << "注：重新进行棋盘格标定calibration会覆盖掉这次的结果" << endl;
    return true;
}

int main() {
    FileStorage calib_file(CALIBRATION_FILENAME, FileStorage::READ);
    if (!calib_file.isOpened()) {
        cerr << "无法读取 " << CALIBRATION_FILENAME << endl;
        return 1;
    }
    Mat camera_matrix, dist_coeffs;
    calib_file["cameraMatrix"] >> camera_matrix;
    calib_file["distCoeffs"] >> dist_coeffs;
    calib_file.release();

    VideoCapture camera("/dev/video0");
    if (!camera.isOpened()) {
        cerr << "无法打开摄像机" << endl;
        return 2;
    }

    int capture_height =
        static_cast<int>(camera.get(cv::CAP_PROP_FRAME_HEIGHT));
    int capture_width = static_cast<int>(camera.get(cv::CAP_PROP_FRAME_WIDTH));
    cout << "摄像机参数 width: " << capture_width
         << ", height: " << capture_height
         << ", fps: " << camera.get(cv::CAP_PROP_FPS) << endl;

    Size camera_size{capture_width, capture_height};
    Mat mapx = Mat(camera_size, CV_32FC1), mapy = Mat(camera_size, CV_32FC1);
    auto R = Mat::eye(3, 3, CV_32F);
    initUndistortRectifyMap(camera_matrix, dist_coeffs, R, camera_matrix,
                            camera_size, CV_32FC1, mapx, mapy);

    vector<Point2f> world_points{{-0.1, 0.14}, {0, 0.14}, {0.1, 0.14},
                                 {-0.1, 0.08}, {0, 0.08}, {0.1, 0.08},
                                 {-0.1, 0},    {0, 0},    {0.1, 0}};

    Calib st;
    st.world = &world_points;
    // 靶纸参考点(标着 0,0 那个)的下标，仅用于高亮显示
    st.anchor_index = -1;
    for (size_t k = 0; k < world_points.size(); ++k) {
        if (std::fabs(world_points[k].x) < 1e-6f &&
            std::fabs(world_points[k].y) < 1e-6f) {
            st.anchor_index = static_cast<int>(k);
        }
    }

    const string win = "Calibrate Homography";
    cv::namedWindow(win, cv::WINDOW_AUTOSIZE);
    cv::setMouseCallback(win, on_mouse, &st);

    cout << "按顺序点 " << world_points.size()
         << " 个点：先最远一排(Y=0.14)，从左到右，再 Y=0.08，最后 Y=0。"
         << endl;
    cout << "点满后自动拟合；r 重点，a 开关原点锚定，p 打印行距表，s 保存"
         << endl;

    while (true) {
        Mat frame;
        if (!grab_image(camera, mapx, mapy, frame)) {
            cerr << "无法读取摄像机源" << endl;
            return 1;
        }

        Mat canvas = frame.clone();
        draw_overlay(canvas, st);

        Mat preview;
        build_preview(frame, st, preview);

        // 直接显示矫正后的原图（1:1），点击坐标无需换算
        st.frame_size = canvas.size();

        if (!preview.empty()) {
            cv::imshow("ground preview (metric)", preview);
        }
        cv::imshow(win, canvas);

        const int key = cv::waitKey(10) & 0xFF;
        if (key == 27 || key == 'q') {
            break;
        } else if (key == 'r') {
            st.image.clear();
            st.has_H = false;
            st.rmse_px = 0;
            st.inliers = 0;
            cout << "已清空，重新点" << endl;
        } else if (key == 'c') {
            fit(st);
        } else if (key == 'a') {
            st.enforce_anchor = !st.enforce_anchor;
            cout << "锚点约束 " << (st.enforce_anchor ? "ON" : "OFF") << endl;
            if (st.image.size() >= 4) {
                fit(st);
            }
        } else if (key == 'p') {
            print_row_table(st);
        } else if (key == 's') {
            save_result(st);
        }

        // 点满自动拟合
        if (!st.has_H && st.image.size() == world_points.size()) {
            fit(st);
        }
    }

    return 0;
}
