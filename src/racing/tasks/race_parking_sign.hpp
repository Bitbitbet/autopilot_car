#pragma once
#include <opencv2/opencv.hpp>

// Supplement for the blue circular P/car/text plate. This is a shape test,
// not a neural-model score; a blue disk or arrow alone is insufficient.
inline cv::Rect raceParkingPShape(const cv::Mat &image) {
    if(image.type()!=CV_8UC3||image.size()!=cv::Size(640,480))return {};
    cv::Mat hsv,blue,white;cv::cvtColor(image,hsv,cv::COLOR_BGR2HSV);
    cv::inRange(hsv,cv::Scalar(90,65,25),cv::Scalar(145,255,255),blue);
    cv::inRange(hsv,cv::Scalar(0,0,105),cv::Scalar(179,105,255),white);
    std::vector<std::vector<cv::Point>> plates;cv::findContours(blue,plates,cv::RETR_EXTERNAL,cv::CHAIN_APPROX_SIMPLE);
    for(const auto &plate:plates) {
        auto b=cv::boundingRect(plate);const float aspect=float(b.width)/b.height;
        if(b.width<28||b.height<28||b.width>420||b.height>420||aspect<.55f||aspect>1.65f||
           cv::contourArea(plate)<b.area()*.5)continue;
        std::vector<std::vector<cv::Point>> letters;std::vector<cv::Vec4i> hierarchy;
        cv::findContours(white(b).clone(),letters,hierarchy,cv::RETR_CCOMP,cv::CHAIN_APPROX_SIMPLE);
        for(size_t i=0;i<letters.size();++i) {
            if(hierarchy[i][3]>=0||hierarchy[i][2]<0)continue;
            const auto p=cv::boundingRect(letters[i]);
            if(p.height<16||p.width<b.width*.20||p.width>b.width*.75||p.height<b.height*.34||p.height>b.height*.82||
               p.x<b.width*.06||p.x+p.width>b.width*.90||p.y>b.height*.35)continue;
            bool upperHole=false;
            for(int child=hierarchy[i][2];child>=0;child=hierarchy[child][0]) {
                auto h=cv::boundingRect(letters[child]);
                upperHole|=h.x+h.width*.5>p.x+p.width*.35&&h.y+h.height*.5<p.y+p.height*.50&&
                    h.width>p.width*.16&&h.height>p.height*.10&&cv::contourArea(letters[child])>p.area()*.055;
            }
            if(!upperHole)continue;
            // Use this connected letter only. The small car/text can overlap
            // P's bounding rectangle without belonging to the P component.
            cv::Mat component=cv::Mat::zeros(b.size(),CV_8UC1),glyph;
            cv::drawContours(component,letters,int(i),{255},-1);
            for(int child=hierarchy[i][2];child>=0;child=hierarchy[child][0])cv::drawContours(component,letters,child,{0},-1);
            cv::resize(component(p),glyph,{64,64},0,0,cv::INTER_NEAREST);
            const auto fill=[&](cv::Rect r){return float(cv::countNonZero(glyph(r)))/r.area();};
            // Tall left stem, top bowl, empty lower right: rejects O/B/R and arrows.
            if(fill({5,5,14,54})>.55&&fill({15,5,30,12})>.28&&fill({32,44,28,16})<.18)return b;
        }
    }
    return {};
}
