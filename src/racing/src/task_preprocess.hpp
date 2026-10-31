#pragma once
#include "camera.hpp"
#include "task_track.hpp"
#include "race_ground.hpp"
// File/video/preview modes use master's calibration and binarization without
// opening a second camera. The caller owns the one VideoCapture stream.
class Predeal:public Camera {
public:
    Predeal():Camera(CalibrationOnly{},-1){Track::camera=this;}
    explicit Predeal(int threshold):Camera(CalibrationOnly{},threshold){Track::camera=this;}
    ~Predeal(){if(Track::camera==this)Track::camera=nullptr;}
    void setGroundCalibration(const RaceGroundCalibration &cfg){
        std::array<cv::Point2f,4> image,ground{{{-cfg.roadWidth/2,cfg.nearDistance},{cfg.roadWidth/2,cfg.nearDistance},{-cfg.roadWidth/2,cfg.farDistance},{cfg.roadWidth/2,cfg.farDistance}}};
        for(size_t i=0;i<4;++i)image[i]={cfg.imagePoints[i][0],cfg.imagePoints[i][1]};
        setGroundHomography(cv::getPerspectiveTransform(image.data(),ground.data()));
    }
};
