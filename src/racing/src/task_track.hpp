#pragma once
#include "track.hpp"
#include "task_types.hpp"
// This is an interchange view of master TrackRows, never a resized image or
// a second edge-search algorithm. The master Camera owns the ground mapping.
class Track {
public:
    inline static Camera *camera=nullptr;
    std::vector<PointX> pointsEdgeLeft,pointsEdgeRight,widthBlock,spurroad;
    int rowCutUp=20,rowCutBottom=40,validRowsLeft=0,validRowsRight=0;
    double stdevLeft=0,stdevRight=0;
    void handle(const cv::Mat &image){
        if(image.size()!=cv::Size(640,480)||image.type()!=CV_8UC1)throw std::invalid_argument("Track expects 640x480 binary");
        NativeTrackRows rows;
        track_identify_from_bin(image,[](cv::Point2d p){return camera?camera->image_to_ground(p):cv::Point2d(NAN,NAN);},rows);
        pointsEdgeLeft.clear();pointsEdgeRight.clear();widthBlock.clear();
        for(auto [row,cols]:rows)if(row>=rowCutUp&&row<480-rowCutBottom){pointsEdgeLeft.emplace_back(row,cols.first);pointsEdgeRight.emplace_back(row,cols.second);widthBlock.emplace_back(row,cols.second-cols.first);}
        validRowsLeft=pointsEdgeLeft.size();validRowsRight=pointsEdgeRight.size();
    }
    void drawImage(cv::Mat &image){for(auto p:pointsEdgeLeft)cv::circle(image,{p.y,p.x},2,{0,255,0},-1);for(auto p:pointsEdgeRight)cv::circle(image,{p.y,p.x},2,{0,255,255},-1);}
};
