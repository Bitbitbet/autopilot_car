#pragma once
#include "race_preview.hpp"

// 部分 X11 后端返回带修饰位的键值，普通按键统一取低八位；无事件仍保留 -1。
inline int raceUiKey(int raw) {
    if(raw<0) return -1;
    // Qt 的 Escape 键不是 ASCII 27；直接取低八位会变成 0。
    if(raw==0x01000000) return 27;
    return raw&255;
}
inline bool raceUiQuit(int key) { return key==27||key=='q'||key=='Q'; }

// 一个窗口显示当前巡线与匹配的 AI 图像，减少远程窗口数量和传输像素。
// 右侧始终取推理原帧，不能用当前图像冒充旧结果；二值图通过 v 切换。
inline cv::Mat raceCompactView(const cv::Mat &live,const cv::Mat &binary,
                              const RaceSnapshot &snapshot,double time,const RaceConfig &cfg,bool showBinary) {
    cv::Mat canvas=cv::Mat::zeros(300,640,CV_8UC3);
    if(live.type()==CV_8UC3&&live.size()==cv::Size(320,240)) live.copyTo(canvas(cv::Rect(0,0,320,240)));
    if(showBinary) {
        if(binary.type()==CV_8UC1&&binary.size()==cv::Size(320,240)) {
            cv::Mat rgb; cv::cvtColor(binary,rgb,cv::COLOR_GRAY2BGR);
            rgb.copyTo(canvas(cv::Rect(320,0,320,240)));
        }
    } else {
        const auto preview=racePreviewImage(snapshot,time,cfg);
        preview(cv::Rect(0,0,320,240)).copyTo(canvas(cv::Rect(320,0,320,240)));
    }
    const double age=snapshot.sequence?time-snapshot.captured:-1;
    const bool fresh=!snapshot.failed&&snapshot.sequence&&age>=0&&age<=cfg.maxDetectionAge;
    const auto color=fresh?cv::Scalar(0,255,0):cv::Scalar(0,160,255);
    std::ostringstream status;
    status<<"LEFT: live lane | RIGHT: "<<(showBinary?"live binary":"matched AI")
          <<" | AI f="<<snapshot.sequence;
    cv::putText(canvas,status.str(),{5,254},cv::FONT_HERSHEY_PLAIN,.8,color,1);
    status.str(""); status.clear();
    status<<(snapshot.failed?"AI FAILED":fresh?"FRESH":"WAIT / STALE")<<" age="<<std::fixed
          <<std::setprecision(2)<<age<<"s";
    for(const auto &d:snapshot.detections) if(raceBoxValid(d)&&d.score>=cfg.confidence)
        status<<" | "<<raceLabel(d.type)<<' '<<std::setprecision(2)<<d.score;
    cv::putText(canvas,status.str(),{5,270},cv::FONT_HERSHEY_PLAIN,.8,color,1);
    cv::putText(canvas,"ESC/q:quit SPACE:stop v:view s:sample | terminal Ctrl+C",{5,289},
                cv::FONT_HERSHEY_PLAIN,.8,{0,255,255},1);
    return canvas;
}
