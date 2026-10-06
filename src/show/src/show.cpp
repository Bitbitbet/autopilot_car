#include "show.hpp"

using std::string;
using std::to_string;

using namespace cv;

Show::Show(const int size) {
    if (size <= 0 || size > 7)
        return;

    namedWindow("ICAR", WINDOW_NORMAL);     // 图像名称
    resizeWindow("ICAR", 480 * 2, 320 * 2); // 分辨率

    imgShow = Mat::zeros(ROWSIMAGE * 2, COLSIMAGE * 2, CV_8UC3);
    enable = true;
    sizeWindow = size;
};

Show::~Show() {};

void Show::setNewWindow(int index, string name, Mat img) {
    // 数据溢出保护
    if (!enable || index <= 0 || index > sizeWindow)
        return;

    if (img.cols <= 0 || img.rows <= 0)
        return;

    Mat imgDraw = img.clone();

    if (imgDraw.type() == CV_8UC1) // 非RGB类型的图像
        cvtColor(imgDraw, imgDraw, cv::COLOR_GRAY2BGR);

    // 图像缩放
    if (imgDraw.cols != COLSIMAGE || imgDraw.rows != ROWSIMAGE) {
        float fx = (float)COLSIMAGE / imgDraw.cols;
        float fy = (float)ROWSIMAGE / imgDraw.rows;
        if (fx <= fy)
            resize(imgDraw, imgDraw, Size(COLSIMAGE, ROWSIMAGE), fx, fx);
        else
            resize(imgDraw, imgDraw, Size(COLSIMAGE, ROWSIMAGE), fy, fy);
    }

    // 限制图片标题长度
    string text = "[" + to_string(index) + "] ";
    if (name.length() > 15)
        text = text + name.substr(0, 15);
    else
        text = text + name;

    putText(imgDraw, text, Point(10, 20), cv::FONT_HERSHEY_TRIPLEX, 0.5,
            cv::Scalar(255, 0, 0), 1);

    if (index <= 2) {
        Rect placeImg = Rect(COLSIMAGE * (index - 1), 0, COLSIMAGE, ROWSIMAGE);
        imgDraw.copyTo(imgShow(placeImg));
    }

    else {
        Rect placeImg =
            Rect(COLSIMAGE * (index - 3), ROWSIMAGE, COLSIMAGE, ROWSIMAGE);
        imgDraw.copyTo(imgShow(placeImg));
    }

    if (save)
        savePicture(img); // 保存图像
}

void Show::show() {
    if (enable) {
        putText(imgShow, "Frame:" + to_string(index),
                Point(COLSIMAGE / 2 - 50, ROWSIMAGE * 2 - 20),
                cv::FONT_HERSHEY_TRIPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
        imshow("ICAR", imgShow);

        auto key = waitKey(1);
        if (key != -1) {
            if (key == 32) // 空格
                realShow = !realShow;
        }
        if (realShow) {
            index++;
            if (index < 0)
                index = 0;
            if (index > frameMax)
                index = frameMax;
        }
    }
}