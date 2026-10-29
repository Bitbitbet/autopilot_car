#pragma once
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
inline void raceConfigureRuntime() {
    int threads=1;
    if(const char *value=std::getenv("CAR_OPENCV_THREADS")) {
        char *end=nullptr;const long n=std::strtol(value,&end,10);
        if(!end||*end||n<1||n>4)throw std::invalid_argument("CAR_OPENCV_THREADS must be 1..4");
        threads=int(n);
    }
    cv::setNumThreads(threads);
    std::cout<<"[Runtime] OpenCV workers="<<threads<<"; GUI publishes latest frame at <=10 FPS\n";
}
inline void raceNormalizeFrame(cv::Mat &image) {
    if(image.empty()||image.type()!=CV_8UC3||image.cols*3!=image.rows*4)
        throw std::invalid_argument("Expected 4:3 BGR capture; processing calibration uses 320x240");
    if(image.size()!=cv::Size(320,240))cv::resize(image,image,{320,240},0,0,cv::INTER_AREA);
}
