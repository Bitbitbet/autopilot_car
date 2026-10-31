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
inline void raceNormalizeFrame(cv::Mat &image,bool archived=false) {
    if(image.empty()||image.type()!=CV_8UC3||image.cols*3!=image.rows*4)
        throw std::invalid_argument("Expected native 640x480 BGR capture");
    if(archived&&image.size()==cv::Size(320,240))cv::resize(image,image,{640,480},0,0,cv::INTER_LINEAR);
    if(image.size()!=cv::Size(640,480))throw std::invalid_argument("Restore camera capture to 640x480; live frames are never downsampled");
}
