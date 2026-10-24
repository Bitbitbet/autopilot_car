#pragma once
#include "race_preview.hpp"

// 先识别后端特殊键，再提取普通字符，避免 X11 左方向键 0xff51 被当成 Q。
inline int raceUiKey(int raw) {
    if(raw<0) return -1;
    if((raw&0x01ffffff)==0x01000000||(raw&0xffff)==0xff1b) return 27;
    if((raw&0xff00)==0xff00||(raw&0x01000000)!=0) return -1;
    const int character=raw&255;
    return character==27||(character>=32&&character<=126)?character:-1;
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
    for(const auto &d:snapshot.detections) {
        const auto review=reviewRaceDetection(d,snapshot.image,cfg);
        if(review.accepted) status<<" | "<<raceLabel(d.type)<<' '<<std::setprecision(2)<<d.score
                                  <<(review.checkedStop?" CHECKED":"");
    }
    cv::putText(canvas,status.str(),{5,270},cv::FONT_HERSHEY_PLAIN,.8,color,1);
    cv::putText(canvas,"ESC/q:quit SPACE:stop v:view s:sample | terminal Ctrl+C",{5,289},
                cv::FONT_HERSHEY_PLAIN,.8,{0,255,255},1);
    return canvas;
}
