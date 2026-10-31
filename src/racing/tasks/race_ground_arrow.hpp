#pragma once
#include "race_manager.hpp"
enum RaceRoadDirection { RoadLeft=1,RoadStraight=2,RoadRight=4 };
struct RaceGroundArrow {int allowed=0;float confidence=0;cv::Rect box;const char *reason="NO_GROUND_ARROW";};
inline RaceGroundArrow raceGroundArrow(const cv::Mat &image,const Track &track,bool blackRoad) {
    RaceGroundArrow result;if(image.size()!=cv::Size(640,480)||image.type()!=CV_8UC3)return result;
    cv::Mat hsv,paint,mask=cv::Mat::zeros(480,640,CV_8UC1);cv::cvtColor(image,hsv,cv::COLOR_BGR2HSV);
    if(blackRoad)cv::inRange(hsv,cv::Scalar(0,0,155),cv::Scalar(179,90,255),paint);
    else cv::inRange(hsv,cv::Scalar(0,0,0),cv::Scalar(179,255,75),paint);
    std::map<int,int> right;for(auto p:track.pointsEdgeRight)right[p.x]=p.y;
    for(auto l:track.pointsEdgeLeft)if(l.x>=70&&l.x<=370&&right.count(l.x)) {
        const int a=std::clamp(l.y+24,0,639),b=std::clamp(right[l.x]-24,0,639);
        if(b>a)cv::line(mask,{a,l.x},{b,l.x},{255},6);
    }
    cv::bitwise_and(paint,mask,paint);std::vector<std::vector<cv::Point>> contours;
    cv::findContours(paint,contours,cv::RETR_EXTERNAL,cv::CHAIN_APPROX_SIMPLE);
    for(const auto &c:contours) {
        const float area=cv::contourArea(c);const auto box=cv::boundingRect(c);
        if(area<260||area>60000||box.width<26||box.height<26||box.width>480||box.height>400)continue;
        std::vector<cv::Point> poly,hull;cv::approxPolyDP(c,poly,cv::arcLength(c,true)*.025,true);cv::convexHull(c,hull);
        const float solidity=area/std::max(1.,cv::contourArea(hull));
        if(poly.size()<7||poly.size()>14||solidity>.94||solidity<.35)continue;
        const double orientation=cv::contourArea(poly,true);std::vector<int> notches;
        for(size_t i=0;i<poly.size();++i) {
            auto a=poly[i]-poly[(i+poly.size()-1)%poly.size()],b=poly[(i+1)%poly.size()]-poly[i];
            if((a.x*b.y-a.y*b.x)*orientation<0)notches.push_back(int(i));
        }
        int allowed=0;
        // Bent exit arrows have a vertical tail and more than nine vertices.
        // Identify the horizontal arrowhead at a pair of concave neck corners;
        // tail orientation must not turn the left head into a straight arrow.
        for(size_t i=0;i<notches.size();++i)for(size_t j=i+1;j<notches.size();++j){
            const auto a=poly[notches[i]],b=poly[notches[j]];
            if(std::abs(a.x-b.x)>box.width*.16||std::abs(a.y-b.y)<box.height*.10)continue;
            const float neck=(a.x+b.x)*.5f,cy=(a.y+b.y)*.5f;
            const auto left=*std::min_element(poly.begin(),poly.end(),[](auto p,auto q){return p.x<q.x;});
            const auto rightmost=*std::max_element(poly.begin(),poly.end(),[](auto p,auto q){return p.x<q.x;});
            const float yTolerance=std::abs(a.y-b.y)*.55f;
            const auto tip=[&](cv::Point p,int side){
                const auto at=std::find(poly.begin(),poly.end(),p)-poly.begin();
                const auto before=poly[(at+poly.size()-1)%poly.size()],after=poly[(at+1)%poly.size()];
                return side*(p.x-before.x)>box.width*.10&&side*(p.x-after.x)>box.width*.10&&
                    (before.y-p.y)*(after.y-p.y)<0;
            };
            if(tip(left,-1)&&neck-left.x>box.width*.22&&std::abs(left.y-cy)<=yTolerance&&rightmost.x-neck>box.width*.20)allowed|=RoadLeft;
            if(tip(rightmost,1)&&rightmost.x-neck>box.width*.22&&std::abs(rightmost.y-cy)<=yTolerance&&neck-left.x>box.width*.20)allowed|=RoadRight;
        }
        if(!allowed&&notches.size()==2&&poly.size()<=9) {
            auto between=[&](int a,int b){std::vector<cv::Point> v;for(int i=(a+1)%int(poly.size());i!=b;i=(i+1)%int(poly.size()))v.push_back(poly[i]);return v;};
            for(int side=0;side<2;++side) {
                const int a=notches[side],b=notches[1-side];const auto head=between(a,b),tail=between(b,a);
                if(head.size()<3||tail.size()<2||cv::norm(head.front()-head.back())<cv::norm(poly[a]-poly[b])*1.2)continue;
                cv::Point2f center;for(auto p:tail)center+=cv::Point2f(p);center*=1.f/tail.size();
                const auto neck=(cv::Point2f(poly[a])+cv::Point2f(poly[b]))*.5f,axis=neck-center;
                const auto tip=*std::max_element(head.begin(),head.end(),[&](auto p,auto q){return (cv::Point2f(p)-neck).dot(axis)<(cv::Point2f(q)-neck).dot(axis);});
                const auto direction=cv::Point2f(tip)-center;
                if(direction.y>box.height*.25)continue;
                allowed=std::abs(direction.x)>std::abs(direction.y)*.55?(direction.x<0?RoadLeft:RoadRight):RoadStraight;
            }
        } else if(!allowed&&notches.size()>=3) {
            const auto left=*std::min_element(poly.begin(),poly.end(),[](auto a,auto b){return a.x<b.x;});
            const auto rightmost=*std::max_element(poly.begin(),poly.end(),[](auto a,auto b){return a.x<b.x;});
            if(left.y<box.y+box.height*.55&&rightmost.y<box.y+box.height*.55)allowed=RoadLeft|RoadRight;
        }
        if(allowed&&area>result.confidence)result={allowed,area,box,"GROUND_ARROW_SHAPE"};
    }
    if(result.allowed)result.confidence=std::min(.95f,.5f+result.confidence/8000);
    return result;
}
