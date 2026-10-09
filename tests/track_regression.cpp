#include "track.hpp"
#include "loop.hpp"
#include <atomic>
#include <cassert>
#include <iostream>

int main() {
    Track track;
    // ROI has no extra trailing row: checked access must reject the old row 240.
    cv::Mat white(ROWSIMAGE, COLSIMAGE, CV_8UC1, cv::Scalar(255));
    track.handle(white);
    assert(!track.pointsEdgeLeft.empty());
    for (const auto &p : track.pointsEdgeLeft)
        assert(p.x >= 0 && p.x < ROWSIMAGE);
    track.handle(true, 65535); // research start must remain within the image
    track.handle(cv::Mat::zeros(ROWSIMAGE, COLSIMAGE, CV_8UC1));
    assert(track.pointsEdgeLeft.empty() && track.pointsEdgeRight.empty());
    bool rejected = false;
    try { track.handle(cv::Mat::zeros(1, 1, CV_8UC1)); }
    catch (const std::invalid_argument &) { rejected = true; }
    assert(rejected);
    std::atomic_int calls{0};
    Loops loop("Regression", 0.005, [&] {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        ++calls;
    });
    loop.start();
    std::this_thread::sleep_for(std::chrono::milliseconds(30));
    loop.shutdown();
    int last = calls;
    std::this_thread::sleep_for(std::chrono::milliseconds(30));
    assert(last > 0 && calls == last);
    std::cout << "PASS: checked track bounds, black image, invalid image, joined worker\n";
}
