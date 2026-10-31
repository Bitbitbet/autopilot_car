#pragma once
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

struct CrosswalkGeometry {
    bool accepted=false;
    cv::Rect box;
    int stripes=0;
    float contrast=0;
    std::vector<cv::Rect> bars;
    std::string reason="NO_STRIPE_BAND";
};

// 地面斑马线是横跨道路的单排深色短条；不把几何证据伪装成模型分数。
// 使用推理对应的原图，检查短条数量、排列、间隔及条间的浅色地面。
inline CrosswalkGeometry detectCrosswalkGeometry(const cv::Mat &image) {
    CrosswalkGeometry best;
    if(image.type()!=CV_8UC3||image.size()!=cv::Size(640,480)) {
        best.reason="INVALID_MATCHED_IMAGE";return best;
    }
    cv::Mat gray,otsu,dark=cv::Mat::zeros(image.size(),CV_8UC1);
    cv::cvtColor(image,gray,cv::COLOR_BGR2GRAY);
    const double threshold=std::clamp(cv::threshold(gray,otsu,0,255,cv::THRESH_BINARY|cv::THRESH_OTSU),35.,125.);
    for(int y=0;y<480;++y) for(int x=0;x<640;++x) {
        const auto c=image.at<cv::Vec3b>(y,x);
        const int high=std::max({int(c[0]),int(c[1]),int(c[2])});
        const int low=std::min({int(c[0]),int(c[1]),int(c[2])});
        if(gray.at<uint8_t>(y,x)<=threshold&&high-low<=std::max(25,high*35/100)) dark.at<uint8_t>(y,x)=255;
    }
    cv::Mat labels,stats,centers;
    const int count=cv::connectedComponentsWithStats(dark,labels,stats,centers,8,CV_32S);
    struct Part { cv::Rect box; int label; };
    std::vector<Part> parts;
    for(int i=1;i<count;++i) {
        const cv::Rect r(stats.at<int>(i,cv::CC_STAT_LEFT),stats.at<int>(i,cv::CC_STAT_TOP),
                         stats.at<int>(i,cv::CC_STAT_WIDTH),stats.at<int>(i,cv::CC_STAT_HEIGHT));
        const int area=stats.at<int>(i,cv::CC_STAT_AREA);
        if(r.x<6||r.y<6||r.br().x>634||r.br().y>474||r.width<6||r.width>120||r.height<10||r.height>180||
           r.height<r.width*.65||r.height>r.width*7.||area<r.area()*.20) continue;
        parts.push_back({r,i});
    }
    if(parts.size()<6||parts.size()>128) return best;
    for(const auto &seedPart:parts) {
        const auto &seed=seedPart.box;
        std::vector<Part> band;
        for(const auto &part:parts) {
            const auto &r=part.box;
            const int overlap=std::min(seed.br().y,r.br().y)-std::max(seed.y,r.y);
            if(overlap>=std::min(seed.height,r.height)*.65&&
               std::abs(seed.y-r.y)<=std::max(8.,std::max(seed.height,r.height)*.30)&&
               std::abs(seed.br().y-r.br().y)<=std::max(8.,std::max(seed.height,r.height)*.30)) band.push_back(part);
        }
        if(band.size()<6||band.size()>20) continue;
        std::sort(band.begin(),band.end(),[](const Part &a,const Part &b){return a.box.x<b.box.x;});
        // 同一高度的孤立黑点不能并入整排条纹；只检查间距连续的一组。
        std::vector<int> centerSteps;
        for(size_t i=1;i<band.size();++i)
            centerSteps.push_back(band[i].box.x+band[i].box.width/2-band[i-1].box.x-band[i-1].box.width/2);
        auto sortedSteps=centerSteps;std::sort(sortedSteps.begin(),sortedSteps.end());
        const int typicalStep=sortedSteps[sortedSteps.size()/2];
        size_t first=0,bestFirst=0,bestEnd=0;
        for(size_t end=1;end<=band.size();++end) {
            if(end==band.size()||centerSteps[end-1]>typicalStep*1.75) {
                if(end-first>bestEnd-bestFirst) { bestFirst=first;bestEnd=end; }
                first=end;
            }
        }
        if(bestEnd-bestFirst<6)continue;
        band=std::vector<Part>(band.begin()+bestFirst,band.begin()+bestEnd);
        std::vector<int> widths,heights,pitches;
        cv::Rect box=band.front().box;
        int commonTop=0,commonBottom=480;
        for(const auto &part:band) {
            box|=part.box;commonTop=std::max(commonTop,part.box.y);commonBottom=std::min(commonBottom,part.box.br().y);
        }
        if(commonBottom-commonTop<6)continue;
        const int sampleRow=(commonTop+commonBottom)/2;
        std::vector<cv::Rect> sections,bars;
        for(const auto &part:band) {
            const auto &r=part.box;int first=640,last=-1;
            for(int x=r.x;x<r.br().x;++x)if(labels.at<int>(sampleRow,x)==part.label) { first=std::min(first,x);last=x; }
            if(last<first)break;
            sections.emplace_back(first,sampleRow,last-first+1,1);bars.push_back(r);
            widths.push_back(last-first+1);heights.push_back(r.height);
        }
        if(sections.size()!=band.size())continue;
        for(size_t i=1;i<sections.size();++i)
            pitches.push_back(sections[i].x+sections[i].width/2-sections[i-1].x-sections[i-1].width/2);
        auto median=[](std::vector<int> v) { std::sort(v.begin(),v.end());return v[v.size()/2]; };
        const int width=median(widths),height=median(heights),pitch=median(pitches);
        if(box.width<160||box.width<box.height*2.5||pitch<width*1.2||pitch>width*3.5) continue;
        bool regular=true;double darkSum=0,lightSum=0;int gaps=0;
        for(size_t i=0;i<band.size();++i) {
            const auto &r=sections[i];
            if(r.width<width*.55||r.width>width*1.8||heights[i]<height*.60||heights[i]>height*1.6) regular=false;
            const cv::Rect core(r.x+r.width/4,sampleRow,std::max(1,r.width/2),1);
            darkSum+=cv::mean(gray(core))[0];
            if(!i)continue;
            if(std::abs(pitches[i-1]-pitch)>std::max(6.,pitch*.35))regular=false;
            const auto &previous=sections[i-1];const int start=previous.br().x,end=r.x;
            if(end-start<2) { regular=false;continue; }
            lightSum+=cv::mean(gray(cv::Rect(start,sampleRow,end-start,1)))[0];++gaps;
        }
        if(!regular||gaps!=int(band.size())-1)continue;
        const float contrast=lightSum/gaps-darkSum/band.size();
        if(contrast<35)continue;
        const int padding=std::max(4,height/5);
        if(box.y<padding+1||box.br().y+padding>=480)continue;
        auto lightFraction=[&](const cv::Rect &region) {
            int light=0;
            for(int y=region.y;y<region.br().y;++y)for(int x=region.x;x<region.br().x;++x) {
                const auto c=image.at<cv::Vec3b>(y,x);
                const int high=std::max({int(c[0]),int(c[1]),int(c[2])});
                const int low=std::min({int(c[0]),int(c[1]),int(c[2])});
                if(gray.at<uint8_t>(y,x)>threshold+25&&high-low<=std::max(45,high*30/100))++light;
            }
            return float(light)/region.area();
        };
        const float above=lightFraction({box.x,box.y-padding,box.width,padding});
        const float below=lightFraction({box.x,box.br().y,box.width,padding});
        if(std::max(above,below)<.85f||std::min(above,below)<.45f)continue;
        if(!best.accepted||band.size()>size_t(best.stripes))
            best={true,box,int(band.size()),contrast,bars,"CROSS_STRIPE_GEOMETRY"};
    }
    return best;
}

inline bool crosswalkBoxMatches(const CrosswalkGeometry &geometry,const cv::Rect &box) {
    if(!geometry.accepted||box.area()<=0)return false;
    const auto overlap=geometry.box&box;
    return overlap.area()>=std::min(geometry.box.area(),box.area())*.65;
}
