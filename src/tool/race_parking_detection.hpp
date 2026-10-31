#pragma once
#include "race_parking_geometry.hpp"
#include "race_ground.hpp"
#include <opencv2/imgproc.hpp>

struct RaceParkingMouthCandidate {
    cv::Point2f nearImage,farImage,nearGround,farGround;
    float support=0;
};
// Rejected bays remain rejected in the current camera ground frame. Applied
// command integration only prevents retry loops; it is not measured odometry.
class RaceParkingBaySearch {
    std::vector<RaceParkingMouthCandidate> rejected;
public:
    void reset(){rejected.clear();}
    bool hasRejected()const{return !rejected.empty();}
    void advance(float distance,float yaw) {
        if(!std::isfinite(distance)||!std::isfinite(yaw))return;
        const cv::Point2f translation{-std::sin(yaw*.5f)*distance,std::cos(yaw*.5f)*distance};
        auto transform=[&](cv::Point2f q){q-=translation;return cv::Point2f{
            std::cos(yaw)*q.x+std::sin(yaw)*q.y,-std::sin(yaw)*q.x+std::cos(yaw)*q.y};};
        for(auto &m:rejected){m.nearGround=transform(m.nearGround);m.farGround=transform(m.farGround);}
        rejected.erase(std::remove_if(rejected.begin(),rejected.end(),[](const auto &m){return m.farGround.y<-.1f;}),rejected.end());
    }
    bool accepts(const RaceParkingMouthCandidate &m)const {
        for(const auto &old:rejected)if(cv::norm(m.nearGround-old.nearGround)<.12f&&
            cv::norm(m.farGround-old.farGround)<.12f)return false;
        return true;
    }
    void reject(cv::Point2f near,cv::Point2f far) {
        RaceParkingMouthCandidate m{};m.nearGround=near;m.farGround=far;
        if(accepts(m))rejected.push_back(m);
    }
    void filter(std::vector<RaceParkingMouthCandidate> &mouths)const {
        mouths.erase(std::remove_if(mouths.begin(),mouths.end(),[&](const auto &m){return !accepts(m);}),mouths.end());
    }
};
// Geometric paint detector, not an AI class or an occupancy detector. Only uses
// the live corrected image and this car's ground map. Never reuses old clicks.
inline std::vector<RaceParkingMouthCandidate> raceDetectParkingMouths(const cv::Mat &image,const RaceGroundMap &map,
                                                                   const RaceParkingDimensions &dims={},std::vector<cv::Point2f> *junctionDebug=nullptr) {
    if(image.type()!=CV_8UC3||image.size()!=cv::Size(320,240))return {};
    constexpr float resolution=.005f,minX=-.62f,maxZ=1.3f;
    constexpr int width=124,height=260;
    cv::Mat hsv,white;cv::cvtColor(image,hsv,cv::COLOR_BGR2HSV);
    cv::inRange(hsv,cv::Scalar(0,0,155),cv::Scalar(180,85,255),white);
    cv::Mat bird=cv::Mat::zeros(height,width,CV_8UC1),visible=bird.clone();
    for(int row=0;row<height;++row)for(int col=0;col<width;++col) {
        const auto p=map.toImage({minX+col*resolution,row*resolution});
        if(!std::isfinite(p.x)||!std::isfinite(p.y)||p.x<3||p.x>=317||p.y<8||p.y>=220)continue;
        visible.at<uchar>(row,col)=255;bird.at<uchar>(row,col)=white.at<uchar>(int(std::lround(p.y)),int(std::lround(p.x)));
    }
    auto sample=[&](float x,float z,int radius=1) {
        const int col=int(std::lround((x-minX)/resolution)),row=int(std::lround(z/resolution));
        if(col<radius||col>=width-radius||row<radius||row>=height-radius||!visible.at<uchar>(row,col))return -1;
        for(int dy=-radius;dy<=radius;++dy)for(int dx=-radius;dx<=radius;++dx)
            if(bird.at<uchar>(row+dy,col+dx))return 1;
        return 0;
    };
    struct Junction {cv::Point2f point;float score;};std::vector<Junction> junctions;
    const float slope=dims.stagger()/dims.depth;
    for(float z=.015f;z<maxZ-.04f;z+=resolution)for(float x=-.38f;x<=-.08f;x+=resolution) {
        if(sample(x,z,0)!=1)continue;
        int stem=0,stemN=0,arm=0,armN=0,background=0,backgroundN=0;
        for(int i=-7;i<=7;++i) {int v=sample(x,z+i*.005f);if(v>=0){stem+=v;++stemN;}}
        for(int i=2;i<=12;++i) {int v=sample(x-i*.01f,z+i*.01f*slope);if(v>=0){arm+=v;++armN;}}
        for(float dz:{-.04f,0.f,.04f})for(float dx:{.035f,.055f}) {
            int v=sample(x+dx,z+dz,0);if(v>=0){background+=1-v;++backgroundN;}
        }
        if(stemN<9||backgroundN<4)continue;
        // A clipped white stem alone is not a located mouth corner. It could be
        // the outside paper or another part of the marking; skip that bay.
        if(armN<3)continue;
        const float stemRatio=float(stem)/stemN,armRatio=float(arm)/armN,darkRatio=float(background)/backgroundN;
        if(stemRatio<.8f||armRatio<.8f||darkRatio<.8f)continue;
        float low=z,high=z;
        for(int i=1;i<55&&sample(x,z-i*resolution,0)==1;++i)low=z-i*resolution;
        for(int i=1;i<55&&sample(x,z+i*resolution,0)==1;++i)high=z+i*resolution;
        if(high-low<.07f||high-low>.24f)continue;
        // Penalize paint below the outgoing diagonal; broad white areas and digits
        // must not become bay mouths just because they contain a vertical stroke.
        int off=0,offN=0;
        for(float dx:{-.04f,-.07f,-.10f}) {int v=sample(x+dx,z+dx*slope,0);if(v>=0){off+=1-v;++offN;}}
        if(offN<1||float(off)/offN<.66f)continue;
        const float centered=std::max(0.f,1-std::abs(z-(low+high)/2)/.04f);
        junctions.push_back({{x,z},stemRatio+armRatio+darkRatio+float(off)/offN+.5f*centered});
    }
    std::sort(junctions.begin(),junctions.end(),[](auto a,auto b){return a.score>b.score;});
    std::vector<Junction> selected;
    for(auto j:junctions) {
        if(std::any_of(selected.begin(),selected.end(),[&](auto old){return std::abs(old.point.x-j.point.x)<.045f&&std::abs(old.point.y-j.point.y)<.075f;}))continue;
        selected.push_back(j);if(selected.size()>=12)break;
    }
    for(auto &j:selected) {
        // Far stems can be cropped at the top of the image. Locate the actual
        // branch/stem junction from the sloping arm, not the cropped stem's center.
        std::vector<float> intercepts;
        for(float dx:{-.04f,-.06f,-.08f}) {
            const float expected=j.point.y-dx*slope;float sum=0;int count=0;
            for(int k=-5;k<=5;++k) {
                const float z=expected+k*resolution;
                if(sample(j.point.x+dx,z,0)==1){sum+=z;++count;}
            }
            if(count>=2&&count<=8)intercepts.push_back(sum/count+dx*slope);
        }
        if(intercepts.size()>=2) {
            std::sort(intercepts.begin(),intercepts.end());j.point.y=intercepts[intercepts.size()/2];
        }
    }
    std::sort(selected.begin(),selected.end(),[](auto a,auto b){return a.point.y<b.point.y;});
    if(junctionDebug)for(auto j:selected)junctionDebug->push_back(j.point);
    std::vector<RaceParkingMouthCandidate> mouths;
    for(size_t i=0;i<selected.size();++i)for(size_t j=i+1;j<selected.size();++j) {
        const auto a=selected[i].point,b=selected[j].point;
        if(std::abs(a.x-b.x)>.045f||!raceCheckParkingMouth(a,b,dims).valid)continue;
        // Require adjacent junctions on this side; do not skip a visible divider.
        if(std::any_of(selected.begin(),selected.end(),[&](auto k){return k.point.y>a.y+.08f&&k.point.y<b.y-.08f&&std::abs(k.point.x-(a.x+b.x)/2)<.05f;}))continue;
        mouths.push_back({map.toImage(a),map.toImage(b),a,b,(selected[i].score+selected[j].score)/9});
    }
    return mouths;
}
