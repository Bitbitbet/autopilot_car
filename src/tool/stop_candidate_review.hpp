#pragma once
#include <opencv2/imgproc.hpp>
#include <string>

// 诊断专用：仅检查低分 stop 框内的红底白横杠；不输出驾驶/放行决策。
struct StopCandidateReview {
    bool matches=false;
    double redFraction=0;
    std::string reason="INVALID_BOX";
};
inline StopCandidateReview reviewStopCandidate(const cv::Mat &image,const cv::Rect &box) {
    StopCandidateReview result;
    if(image.type()!=CV_8UC3||box.width<22||box.height<22||
       (box&cv::Rect(0,0,image.cols,image.rows))!=box) return result;
    const double aspect=double(box.width)/box.height;
    if(aspect<.55||aspect>1.8) { result.reason="SHAPE_ASPECT";return result; }
    const auto roi=image(box);
    cv::Mat red=cv::Mat::zeros(box.size(),CV_8UC1),white=red.clone();
    for(int y=0;y<roi.rows;++y) for(int x=0;x<roi.cols;++x) {
        const auto c=roi.at<cv::Vec3b>(y,x);
        const int b=c[0],g=c[1],r=c[2],hi=std::max({b,g,r}),lo=std::min({b,g,r});
        if(r>=85&&r>=1.25*g&&r>=1.25*b&&r-std::max(g,b)>=25) red.at<unsigned char>(y,x)=255;
        if(lo>=120&&hi-lo<=70) white.at<unsigned char>(y,x)=255;
    }
    result.redFraction=double(cv::countNonZero(red))/box.area();
    if(result.redFraction<.22) { result.reason="NO_RED_FACE";return result; }
    // 中部连续多行必须有足够宽的白横杠，且上方、下方都能看到红色区域。
    int begin=-1;
    for(int y=box.height/5;y<box.height*4/5;++y) {
        int run=0,longest=0;
        for(int x=0;x<box.width;++x) {
            run=white.at<unsigned char>(y,x)?run+1:0;longest=std::max(run,longest);
        }
        if(longest>=box.width*.48) { if(begin<0) begin=y; }
        else if(begin>=0) {
            const int height=y-begin;const double center=(begin+y)*.5/box.height;
            if(height>=std::max(3,int(box.height*.06))&&height<=box.height*.32&&center>=.28&&center<=.72) {
                const int above=cv::countNonZero(red.rowRange(0,begin));
                const int below=cv::countNonZero(red.rowRange(y,box.height));
                if(above>=box.area()*.065&&below>=box.area()*.065) {
                    result.matches=true;result.reason="RED_WHITE_BAR_MATCH";return result;
                }
            }
            begin=-1;
        }
    }
    result.reason="NO_CENTER_WHITE_BAR";return result;
}
