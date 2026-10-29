#pragma once
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

// 地面原点是车头最前端在地面的投影；x 向右，z 向前，单位米。
// 必须使用当前车去畸变后的 320x240 图像。不能复制另一辆车的四点。
struct RaceGroundCalibration {
    int version=1;
    float roadWidth=.45f,vehicleWidth=.18f,vehicleLength=.31f;
    float nearDistance=.25f,farDistance=.65f,clearance=.015f;
    std::array<std::array<float,2>,4> imagePoints{}; // 近左、近右、远左、远右
    std::string lensFingerprint;
    NLOHMANN_DEFINE_TYPE_INTRUSIVE(RaceGroundCalibration,version,roadWidth,vehicleWidth,vehicleLength,
        nearDistance,farDistance,clearance,imagePoints,lensFingerprint)
    static std::string fingerprint(const std::string &path="../res/calibration/calibration.xml") {
        std::ifstream in(path,std::ios::binary);if(!in)throw std::runtime_error("Cannot read lens calibration");
        uint64_t hash=14695981039346656037ull;char c;
        while(in.get(c)){hash^=static_cast<unsigned char>(c);hash*=1099511628211ull;}
        std::ostringstream out;out<<std::hex<<hash;return out.str();
    }
    void validate() const {
        auto range=[](float x,float lo,float hi){return std::isfinite(x)&&x>=lo&&x<=hi;};
        if(version!=1||!range(roadWidth,.3f,.8f)||!range(vehicleWidth,.1f,.3f)||
           !range(vehicleLength,.2f,.6f)||!range(nearDistance,.15f,.4f)||
           !range(farDistance,.5f,1.f)||farDistance-nearDistance<.25f||
           !range(clearance,.015f,.04f)||vehicleWidth+2*clearance>=roadWidth||lensFingerprint.empty())
            throw std::invalid_argument("Invalid ground dimensions/calibration");
        for(const auto &p:imagePoints)if(!range(p[0],4,315)||!range(p[1],10,220))
            throw std::invalid_argument("Ground points must be visible inside corrected image");
        const auto &nl=imagePoints[0],&nr=imagePoints[1],&fl=imagePoints[2],&fr=imagePoints[3];
        if(nr[0]-nl[0]<60||fr[0]-fl[0]<25||nl[1]-fl[1]<20||nr[1]-fr[1]<20||
           std::abs(nl[1]-nr[1])>35||std::abs(fl[1]-fr[1])>25||
           nr[0]-nl[0]<=fr[0]-fl[0]+5)
            throw std::invalid_argument("Invalid point order/perspective: near-left, near-right, far-left, far-right");
        std::vector<cv::Point2f> polygon{{nl[0],nl[1]},{nr[0],nr[1]},{fr[0],fr[1]},{fl[0],fl[1]}};
        if(!cv::isContourConvex(polygon)||cv::contourArea(polygon)<1500)
            throw std::invalid_argument("Degenerate ground quadrilateral");
    }
    void load(const std::string &path,bool checkLens=true) {
        std::ifstream in(path);if(!in)throw std::runtime_error("Ground calibration missing: run calibrate first");
        nlohmann::json j;in>>j;*this=j.get<RaceGroundCalibration>();validate();
        if(checkLens&&fingerprint()!=lensFingerprint)throw std::runtime_error("Lens calibration changed; redo ground calibration");
    }
};

class RaceGroundMap {
    cv::Mat imageToGround,groundToImage;
    // 四点标定只确定映射形状；车相对路面的高度/姿态一旦变化，整个米制尺度会
    // 同比例偏大或偏小（x 与 z 一起）。1 表示完全使用标定值。
    float metricScale=1.f;
    static cv::Point2f project(const cv::Mat &h,cv::Point2f p) {
        const double w=h.at<double>(2,0)*p.x+h.at<double>(2,1)*p.y+h.at<double>(2,2);
        if(!std::isfinite(w)||std::abs(w)<1e-8)return {NAN,NAN};
        return {float((h.at<double>(0,0)*p.x+h.at<double>(0,1)*p.y+h.at<double>(0,2))/w),
                float((h.at<double>(1,0)*p.x+h.at<double>(1,1)*p.y+h.at<double>(1,2))/w)};
    }
public:
    explicit RaceGroundMap(const RaceGroundCalibration &c) {
        c.validate();std::array<cv::Point2f,4> image;
        for(size_t i=0;i<4;++i)image[i]={c.imagePoints[i][0],c.imagePoints[i][1]};
        const std::array<cv::Point2f,4> ground{{{-c.roadWidth/2,c.nearDistance},{c.roadWidth/2,c.nearDistance},
            {-c.roadWidth/2,c.farDistance},{c.roadWidth/2,c.farDistance}}};
        imageToGround=cv::getPerspectiveTransform(image.data(),ground.data());groundToImage=imageToGround.inv();
        if(!cv::checkRange(imageToGround)||!cv::checkRange(groundToImage))throw std::invalid_argument("Singular ground mapping");
        // 图像向下必须接近车头；禁止翻转/跨越投影无穷点。
        const auto a=toGround({160,220}),b=toGround({160,190});
        if(!std::isfinite(a.y)||!std::isfinite(b.y)||a.y>=b.y||a.y<-.2f||b.y>c.nearDistance+.1f)
            throw std::invalid_argument("Ground map has implausible near-field projection");
    }
    // 尺度修正只允许有限范围，避免把错误的路边提取当成整幅米制缩放。
    void setMetricScale(float s) {
        if(!std::isfinite(s)||s<.5f||s>2.f)throw std::invalid_argument("Invalid ground metric scale");
        metricScale=s;
    }
    float scale() const {return metricScale;}
    cv::Point2f toGround(cv::Point2f p) const {
        const auto q=project(imageToGround,p);return {q.x*metricScale,q.y*metricScale};
    }
    cv::Point2f toImage(cv::Point2f p) const {
        return project(groundToImage,{p.x/metricScale,p.y/metricScale});
    }
};

// 从已验证属于当前镜头的四点映射生成逐行宽度参考，供v6单边重建冷启动。
// 以两条地面边线与图像行求交，兼容轻微横滚；它仍是标定参考，不冒充现场学习结果。
inline std::vector<std::pair<int,float>> raceGroundLaneWidthProfile(const RaceGroundCalibration &cfg) {
    RaceGroundMap map(cfg);std::vector<std::pair<int,float>> out;
    const auto a=map.toImage({-cfg.roadWidth/2,cfg.nearDistance}),b=map.toImage({-cfg.roadWidth/2,cfg.farDistance});
    const auto c=map.toImage({cfg.roadWidth/2,cfg.nearDistance}),d=map.toImage({cfg.roadWidth/2,cfg.farDistance});
    for(int row=35;row<=219;++row) {
        const auto ground=map.toGround({160.f,float(row)});
        if(!std::isfinite(ground.y)||ground.y<.12f||ground.y>1.f)continue;
        const float left=a.x+(b.x-a.x)*(row-a.y)/(b.y-a.y),right=c.x+(d.x-c.x)*(row-c.y)/(d.y-c.y);
        if(std::isfinite(left)&&std::isfinite(right)&&right>left)out.emplace_back(row,right-left);
    }
    return out;
}

// 独立标定入口：只打开摄像头和图像窗口，不创建串口、手柄或 NPU 后端。
template<class Preprocess,class StopFlag>
int raceCalibrateGround(cv::VideoCapture &capture,Preprocess &predeal,const std::string &path,StopFlag &stop) {
    RaceGroundCalibration cfg;cfg.lensFingerprint=RaceGroundCalibration::fingerprint();
    std::cout<<"地面标定，不连接电机。车头对正直道中线。\n"
        <<"以车头最前端的地面投影为0，在前方25cm和65cm两处分别标记左右道路边缘（宽45cm）。\n"
        <<"车宽18cm、车长约31cm。相机与车体固定后，按f冻结画面，再依次点击近左、近右、远左、远右。\n"
        <<"r重来，s保存，q/Esc取消。点位应在画面内，标记不要遮盖道路边缘。\n";
    struct Clicks {bool frozen=false;std::vector<cv::Point2f> points;};Clicks state;
    const char *window="ground_calibration - NO MOTOR";cv::namedWindow(window,cv::WINDOW_AUTOSIZE);
    cv::setMouseCallback(window,[](int event,int x,int y,int,void *context){
        auto &s=*static_cast<Clicks*>(context);
        if(event==cv::EVENT_LBUTTONDOWN&&s.frozen&&s.points.size()<4)s.points.emplace_back(x/2.f,y/2.f);
    },&state);
    cv::Mat image;
    while(!stop.load()) {
        if(!state.frozen) {
            if(!capture.read(image)||image.empty())throw std::runtime_error("Calibration camera unavailable");
            if(image.size()!=cv::Size(320,240))cv::resize(image,image,{320,240});
            predeal.correct(image);
        }
        cv::Mat shown=image.clone();
        for(size_t i=0;i<state.points.size();++i) {
            cv::circle(shown,state.points[i],3,{0,0,255},-1);
            cv::putText(shown,std::to_string(i+1),state.points[i]+cv::Point2f(4,-4),cv::FONT_HERSHEY_PLAIN,1,{0,0,255});
        }
        cv::putText(shown,state.frozen?"CLICK: near L/R, far L/R | s save":"f freeze | NO MOTOR",{4,15},cv::FONT_HERSHEY_PLAIN,.85,{0,0,255});
        cv::resize(shown,shown,{640,480});cv::imshow(window,shown);
        const int key=cv::waitKey(20)&255;
        // 车上旧版GTK/X11可能不支持VISIBLE查询，或在远程窗口映射前返回0。
        // 标定不连接电机，直接使用明确的退出键/终端信号，避免刚开窗即误取消。
        if(key=='q'||key==27)break;
        if(key=='f'&&!image.empty())state.frozen=true;
        if(key=='r'){state.points.clear();state.frozen=false;}
        if(key=='s'&&state.points.size()==4) {
            for(size_t i=0;i<4;++i)cfg.imagePoints[i]={state.points[i].x,state.points[i].y};
            try {
                RaceGroundMap verified(cfg);(void)verified;
                // 临时文件写完才替换；取消/无效点不会破坏既有标定。
                const auto temp=path+".tmp";
                {std::ofstream out(temp);if(!out)throw std::runtime_error("Cannot write ground calibration");
                 out<<nlohmann::json(cfg).dump(2)<<'\n';out.close();if(!out)throw std::runtime_error("Ground calibration disk failure");}
                std::filesystem::rename(temp,path);
                cv::imwrite(path+".png",image);
                std::cout<<"[Ground] Saved "<<path<<"; 下一步 bash ./run_car.sh bypass competition\n";
                cv::destroyWindow(window);return 0;
            }catch(const std::exception &e){std::cerr<<"[Ground] "<<e.what()<<"; press r to retry\n";}
        }
    }
    cv::destroyWindow(window);std::cout<<"[Ground] Cancelled; existing file preserved\n";return 1;
}
