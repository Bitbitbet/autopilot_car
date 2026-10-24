#pragma once
#include <opencv2/opencv.hpp>

// 单帧二值图无法区分画面顶边截断的红色圆弧与黑色道路分隔带。
// 仅检查同一帧彩色图中明确偏红、扁平、触顶且独立的暗色连通域。
// 只还原该连通域的像素，不扩张矩形、不沿用旧帧，也不修改巡线二值图。
// 黑色分隔带、与近处障碍连通的暗区、无颜色证据的情况仍保留原判断。
inline cv::Mat raceAppearanceGeometry(const cv::Mat &binary,const cv::Mat &image,int &removed) {
    removed=0;
    if(binary.type()!=CV_8UC1||binary.size()!=cv::Size(320,240)||
       image.type()!=CV_8UC3||image.size()!=binary.size()) return binary;
    const auto red=[](const cv::Vec3b &p) {
        return p[2]>=70&&int(p[2])-p[1]>=25&&int(p[2])-p[0]>=25&&
               int(p[2])*10>=int(p[1])*14&&int(p[2])*10>=int(p[0])*14;
    };
    int redAtTop=0;
    for(int col=0;col<binary.cols;++col)
        if(binary.at<uint8_t>(0,col)<=127&&red(image.at<cv::Vec3b>(0,col))) ++redAtTop;
    if(redAtTop<30) return binary;
    cv::Mat dark,labels,stats,centroids;
    cv::compare(binary,127,dark,cv::CMP_LE);
    const int components=cv::connectedComponentsWithStats(dark,labels,stats,centroids,8,CV_32S);
    cv::Mat geometry;
    for(int label=1;label<components;++label) {
        const int x=stats.at<int>(label,cv::CC_STAT_LEFT),y=stats.at<int>(label,cv::CC_STAT_TOP);
        const int width=stats.at<int>(label,cv::CC_STAT_WIDTH),height=stats.at<int>(label,cv::CC_STAT_HEIGHT);
        const int area=stats.at<int>(label,cv::CC_STAT_AREA);
        if(y!=0||x<8||x+width>312||width<40||width>180||height<6||height>40||
           width<height*3||area<100) continue;
        int colored=0;
        for(int row=y;row<y+height;++row) for(int col=x;col<x+width;++col)
            if(labels.at<int>(row,col)==label&&red(image.at<cv::Vec3b>(row,col))) ++colored;
        // 少量抗锯齿像素可以没有颜色，主体必须由本帧的红色像素证明。
        if(colored*100<area*90) continue;
        if(geometry.empty()) geometry=binary.clone();
        for(int row=y;row<y+height;++row) for(int col=x;col<x+width;++col)
            if(labels.at<int>(row,col)==label) geometry.at<uint8_t>(row,col)=255;
        ++removed;
    }
    return geometry.empty()?binary:geometry;
}
