#include "tools.hpp"
#include "mapping.hpp"
#include <cstdint>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>

using namespace cv;

using std::string;
using std::to_string;
using std::vector;

Mapping ipm = Mapping(Size(COLSIMAGE, ROWSIMAGE),
                      Size(COLSIMAGEIPM, ROWSIMAGEIPM)); // 逆透视变换类

void savePicture(Mat &image) {
    // 存图
    string name = ".jpg";
    static int counter = 0;
    counter++;
    string img_path = "../res/samples/train/";
    name = img_path + to_string(counter) + ".jpg";
    imwrite(name, image);
}

void savePicture(string path, Mat &image) {
    // 存图
    string name = ".jpg";
    static int counterSave = 0;
    counterSave++;
    name = path + to_string(counterSave) + ".jpg";
    imwrite(name, image);
}

double average(vector<int> vec) {
    if (vec.size() < 1)
        return -1;

    double sum = 0;
    for (int i = 0; i < vec.size(); i++) {
        sum += vec[i];
    }

    return (double)sum / vec.size();
}

double sigma(vector<int> vec) {
    if (vec.size() < 1)
        return 0;

    double aver = average(vec); // 集合平均值
    double sigma = 0;
    for (int i = 0; i < vec.size(); i++) {
        sigma += (vec[i] - aver) * (vec[i] - aver);
    }
    sigma /= (double)vec.size();
    return sigma;
}

double sigma(vector<PointX> vec) {
    if (vec.size() < 1)
        return 0;

    double sum = 0;
    for (int i = 0; i < vec.size(); i++) {
        sum += vec[i].y;
    }
    double aver = (double)sum / vec.size(); // 集合平均值

    double sigma = 0;
    for (int i = 0; i < vec.size(); i++) {
        sigma += (vec[i].y - aver) * (vec[i].y - aver);
    }
    sigma /= (double)vec.size();
    return sigma;
}

static uint64_t comb(uint64_t n, uint64_t k) {
    if (k > n)
        return 0;
    if (k > n - k)
        k = n - k;

    uint64_t res = 1;
    for (uint64_t j = 1; j <= k; ++j) {
        // res = res * (n - k + j) / j
        __uint128_t tmp = (__uint128_t)res * (n - k + j);
        tmp /= j;

        if (tmp > UINT64_MAX) {
            return 0;
        }
        res = (uint64_t)tmp;
    }
    return res;
}

vector<PointX> Bezier(double dt, vector<PointX> input) {
    vector<PointX> output;

    double t = 0;
    const auto n = input.size() - 1;
    while (t <= 1) {
        PointX p;
        double x_sum = 0.0;
        double y_sum = 0.0;
        for (int i = 0; i <= n; ++i) {
            double k = comb(n, i) * pow(t, i) * pow(1 - t, n - i);
            x_sum += k * input[i].x;
            y_sum += k * input[i].y;
        }
        p.x = x_sum;
        p.y = y_sum;
        output.push_back(p);
        t += dt;
    }
    return output;
}

string double2String(double val, int fixed) {
    auto str = to_string(val);
    return str.substr(0, str.find(".") + fixed + 1);
}

double distanceForPoint2Line(PointX a, PointX b, PointX p) {
    int d = 0; // 距离

    double ab_distance =
        sqrt((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y));
    double ap_distance =
        sqrt((a.x - p.x) * (a.x - p.x) + (a.y - p.y) * (a.y - p.y));
    double bp_distance =
        sqrt((p.x - b.x) * (p.x - b.x) + (p.y - b.y) * (p.y - b.y));

    double half = (ab_distance + ap_distance + bp_distance) / 2;
    double area = sqrt(half * (half - ab_distance) * (half - ap_distance) *
                       (half - bp_distance));

    return (2 * area / ab_distance);
}

double distanceForOrigin(PointX startPoint, PointX endPoint) {
    Point2d a, b;
    a.x = startPoint.y;
    a.y = startPoint.x;
    b.x = endPoint.y;
    b.y = endPoint.x;
    Point2d c = ipm.homography(a);
    Point2d d = ipm.homography(b);
    double e = distanceForPoints(c, d);
    return e;
}

bool posInRange(double pos, double a, double b) { return pos < a || pos > b; }

vector<PointX> simpleLine(PointX startPoint, PointX endPoint, double n,
                          double k) {
    PointX midPoint1((startPoint.x + endPoint.x) * k / 2,
                     (startPoint.y + endPoint.y) / 2);
    vector<PointX> input = {startPoint, midPoint1, endPoint};
    vector<PointX> b_modify = Bezier(n, input); // 贝塞尔曲线方法
    return b_modify;
}

vector<PointX> smoothLine(vector<PointX> Temp1, double n, double k) {
    vector<PointX> AimPoints;
    if (Temp1.size()) {
        AimPoints.push_back(Temp1[0]);
        for (int i = 1; i < Temp1.size(); i++) {
            if (abs(Temp1[i].y - AimPoints[AimPoints.size() - 1].y) < 5) {
                AimPoints.push_back(Temp1[i]);
            } else {
                vector<PointX> tp =
                    simpleLine(AimPoints[AimPoints.size() - 1], Temp1[i], n, k);
                for (int i = 1; i < tp.size() - 1; i++) {
                    if (abs(tp[i].y - AimPoints[AimPoints.size() - 1].y) > 3)
                        AimPoints.push_back(tp[i]);
                }
            }
        }
    }
    return AimPoints;
}

double gradientCal(PointX a, PointX b) {
    double k = (double)(a.y - b.y) / (a.x - b.x); // 斜率
    return k;
}

int linearCheck(vector<PointX> PointsEdge, int startLine, int endLine,
                int judgebias) {
    int bias = 0;
    endLine = min(endLine, (int)PointsEdge.size());
    double k = gradientCal(PointsEdge[startLine], PointsEdge[endLine]); // 斜率
    double b = PointsEdge[startLine].y - k * PointsEdge[startLine].x;
    for (
        int j = startLine; j < endLine;
        j++) { // 从找到的圆环拐点开始遍历，直到上拐点对应的右侧直道斜率都差不多
        if (abs(k * PointsEdge[j].x + b - PointsEdge[j].y) >
            judgebias) // 斜率差值过大
            bias++;    // 环外部分
    }
    return bias;
}

bool breakpointCheck(PointX startPoint, PointX endPoint, int Min_Len,
                     int Max_Len) {
    Point2f a, b;
    a.x = startPoint.y;
    a.y = startPoint.x;
    b.x = endPoint.y;
    b.y = endPoint.x;
    Point2f c = ipm.homography(a), d = ipm.homography(b);
    double e = distanceForPoints(c, d);

    return e >= Min_Len && e <= Max_Len;
}
