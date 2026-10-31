#pragma once
#include "tools.hpp"
#include <algorithm>
#include <cmath>

struct ForkDirectionCheck {
    int rawType=-1,finalType=-1;
    float modelScore=0;
    cv::Rect box;
    const char *reason="UNVERIFIED";
};

// 只复核蓝底白色斜向箭头，不把标签整体交换，也不改变其他交通目标。
// 两个凹角是箭头头部和杆部的连接点；用杆部到头部的轴向寻找箭尖，
// 允许印刷圆角、透视和轻微裁切产生额外的轮廓顶点。
inline int forkArrowDirection(const cv::Mat &image,const PredictResult &d,const char *&reason) {
    reason="ARROW_NOT_CLEAR";
    if(image.empty()||image.type()!=CV_8UC3||d.x<0||d.y<0||d.width<12||d.height<12||
       d.x>=image.cols||d.y>=image.rows||d.width>image.cols-d.x||d.height>image.rows-d.y)
        return 0;
    const int pad=std::max(4,std::min(d.width,d.height)/6);
    const cv::Rect roi=cv::Rect(d.x-pad,d.y-pad,d.width+2*pad,d.height+2*pad)&
        cv::Rect(0,0,image.cols,image.rows);
    cv::Mat hsv,blue;
    cv::cvtColor(image(roi),hsv,cv::COLOR_BGR2HSV);
    cv::inRange(hsv,cv::Scalar(90,65,35),cv::Scalar(140,255,255),blue);
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(blue,contours,cv::RETR_EXTERNAL,cv::CHAIN_APPROX_SIMPLE);
    if(contours.empty()) { reason="NO_BLUE_SIGN"; return 0; }
    const auto largest=std::max_element(contours.begin(),contours.end(),[](const auto &a,const auto &b) {
        return cv::contourArea(a)<cv::contourArea(b);
    });
    const double blueArea=cv::contourArea(*largest);
    if(blueArea<150) { reason="BLUE_SIGN_TOO_SMALL"; return 0; }
    cv::Mat inside=cv::Mat::zeros(blue.size(),CV_8UC1),white;
    cv::drawContours(inside,std::vector<std::vector<cv::Point>>{*largest},0,cv::Scalar(255),-1);
    cv::erode(inside,inside,cv::getStructuringElement(cv::MORPH_ELLIPSE,{3,3}));
    cv::inRange(hsv,cv::Scalar(0,0,135),cv::Scalar(179,110,255),white);
    cv::bitwise_and(white,inside,white);
    cv::findContours(white,contours,cv::RETR_EXTERNAL,cv::CHAIN_APPROX_SIMPLE);
    if(contours.empty()) return 0;
    const auto arrow=std::max_element(contours.begin(),contours.end(),[](const auto &a,const auto &b) {
        return cv::contourArea(a)<cv::contourArea(b);
    });
    const double area=cv::contourArea(*arrow);
    if(area<45||area<blueArea*.06||area>blueArea*.65) return 0;
    const auto bounds=cv::boundingRect(*arrow);
    int leftVotes=0,rightVotes=0;
    for(double tolerance:{.012,.016,.020,.025,.030,.040}) {
        std::vector<cv::Point> polygon;
        cv::approxPolyDP(*arrow,polygon,cv::arcLength(*arrow,true)*tolerance,true);
        const int n=int(polygon.size());
        if(n<7||n>12) continue;
        const double orientation=cv::contourArea(polygon,true);
        std::vector<int> notches;
        for(int i=0;i<n;++i) {
            const cv::Point2f a=polygon[i]-polygon[(i+n-1)%n];
            const cv::Point2f b=polygon[(i+1)%n]-polygon[i];
            const double cross=a.x*b.y-a.y*b.x;
            if(cross*orientation<0&&std::fabs(cross)>4) notches.push_back(i);
        }
        if(notches.size()!=2) continue;
        int a=notches[0],b=notches[1];
        auto between=[&](int from,int to) {
            std::vector<int> ids;
            for(int i=(from+1)%n;i!=to;i=(i+1)%n) ids.push_back(i);
            return ids;
        };
        const cv::Point2f neck=(cv::Point2f(polygon[a])+cv::Point2f(polygon[b]))*.5f;
        const double neckWidth=cv::norm(polygon[a]-polygon[b]);
        for(int side=0;side<2;++side) {
            const auto head=side?between(b,a):between(a,b);
            const auto tail=side?between(a,b):between(b,a);
            if(head.size()<3||tail.size()<2) continue;
            const double headWidth=cv::norm(polygon[head.front()]-polygon[head.back()]);
            if(headWidth<neckWidth*1.25) continue;
            cv::Point2f tailCenter;
            for(int id:tail) tailCenter+=cv::Point2f(polygon[id]);
            tailCenter*=1.f/tail.size();
            const cv::Point2f axis=neck-tailCenter;
            if(cv::norm(axis)<4) continue;
            const int tipId=*std::max_element(head.begin(),head.end(),[&](int x,int y) {
                return (cv::Point2f(polygon[x])-neck).dot(axis)<(cv::Point2f(polygon[y])-neck).dot(axis);
            });
            const cv::Point2f tip=polygon[tipId];
            if((tip-neck).dot(axis)<=0||
               std::fabs(tip.x-neck.x)<bounds.width*.15) continue;
            if(tip.x<neck.x) ++leftVotes; else ++rightVotes;
        }
    }
    if(leftVotes>=2&&rightVotes==0) { reason="ARROW_LEFT"; return -1; }
    if(rightVotes>=2&&leftVotes==0) { reason="ARROW_RIGHT"; return 1; }
    return 0;
}

inline std::vector<ForkDirectionCheck> verifyForkDirections(const cv::Mat &image,
                                                           std::vector<PredictResult> &detections) {
    std::vector<ForkDirectionCheck> checks;
    detections.erase(std::remove_if(detections.begin(),detections.end(),[&](PredictResult &d) {
        if(d.type!=LABEL_YLEFT&&d.type!=LABEL_YRIGHT&&d.type!=LABEL_LEFT) return false;
        ForkDirectionCheck check; check.rawType=d.type; check.modelScore=d.score; check.box={d.x,d.y,d.width,d.height};
        const int direction=forkArrowDirection(image,d,check.reason);
        check.finalType=direction<0?LABEL_YLEFT:direction>0?LABEL_YRIGHT:-1;
        checks.push_back(check);
        if(!direction) return true;
        d.type=check.finalType; d.label=direction<0?"yleft":"yright";
        return false;
    }),detections.end());
    return checks;
}
