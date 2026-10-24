// 低分停车牌诊断预览，构建时不链接 control 或 joystick；不能驱动车辆。
#include "detection.hpp"
#include "predeal.hpp"
#include "stop_candidate_review.hpp"
#include <opencv2/opencv.hpp>
#include <nlohmann/json.hpp>
#include <csignal>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

namespace { volatile std::sig_atomic_t stopped=0;void stop(int) { stopped=1; } }
int main(int argc,char**) {
    if(argc!=1) { std::cerr<<"This is a camera-only diagnostic; no options or drive mode accepted.\n";return 2; }
    std::signal(SIGINT,stop);std::signal(SIGTERM,stop);std::signal(SIGHUP,stop);
    try {
        namespace fs=std::filesystem;
        const auto stamp=std::chrono::system_clock::now().time_since_epoch().count();
        const auto directory=fs::path("../preview-records")/std::to_string(stamp);fs::create_directories(directory);
        std::ofstream log(directory/"candidates.jsonl");
        if(!log) throw std::runtime_error("Cannot open preview log");
        Detection detector("../res/models/yolov3_mobilenet_v1",.30f);
        Predeal predeal;
        cv::VideoCapture camera("/dev/video0",cv::CAP_V4L2);
        if(!camera.isOpened()) throw std::runtime_error("Cannot open camera; exit other camera programs first");
        camera.set(cv::CAP_PROP_FRAME_WIDTH,320);camera.set(cv::CAP_PROP_FRAME_HEIGHT,240);
        camera.set(cv::CAP_PROP_BUFFERSIZE,1);
        std::cout<<"NO CONTROL | STOP CANDIDATE REVIEW ONLY | floor=0.30, ordinary threshold=0.65 | s:sample, q/ESC:quit\n"<<directory<<std::endl;
        int frame=0,samples=0;
        while(!stopped) {
            cv::Mat image;if(!camera.read(image)||image.empty()) break;
            if(image.size()!=cv::Size(320,240)) cv::resize(image,image,{320,240});
            predeal.correct(image);detector.inference(image);
            auto shown=image.clone();auto entries=nlohmann::json::array();
            for(const auto &d:detector.results) {
                if(d.type!=5) continue;
                const auto review=reviewStopCandidate(image,{d.x,d.y,d.width,d.height});
                const bool ordinary=d.score>=.65f;
                entries.push_back({{"type",d.type},{"score",d.score},{"x",d.x},{"y",d.y},{"width",d.width},{"height",d.height},
                    {"ordinary_threshold",ordinary},{"appearance_match",review.matches},{"appearance_reason",review.reason},
                    {"red_fraction",review.redFraction},{"near_row_145",d.y+d.height>=145}});
                const auto box=cv::Rect(d.x,d.y,d.width,d.height)&cv::Rect(0,0,320,240);
                if(box.empty())continue;
                const auto color=ordinary?cv::Scalar(0,255,0):review.matches?cv::Scalar(0,200,255):cv::Scalar(0,0,255);
                cv::rectangle(shown,box,color,1);
                std::ostringstream text;text<<"stop "<<std::fixed<<std::setprecision(2)<<d.score<<' '
                    <<(ordinary?"NORMAL":review.matches?"REVIEW":"REJECT");
                cv::putText(shown,text.str(),{box.x,std::max(12,box.y-2)},cv::FONT_HERSHEY_PLAIN,.8,color,1);
            }
            nlohmann::json row={{"frame",++frame},{"no_control",true},{"candidates",entries}};
            log<<row.dump()<<'\n';if(frame%10==0)log.flush();
            cv::putText(shown,"NO CONTROL | diagnostic only",{4,232},cv::FONT_HERSHEY_PLAIN,.8,{0,255,255},1);
            cv::imshow("Stop candidate review - NO CONTROL",shown);
            const int key=cv::waitKey(10);
            if(key==27||key=='q'||key=='Q')break;
            if(key=='s') {
                const auto path=directory/("sample-"+std::to_string(++samples));fs::create_directories(path);
                cv::imwrite((path/"camera.png").string(),image);
                std::ofstream(path/"candidates.json")<<row.dump(2);log.flush();
                std::cout<<"Saved "<<path<<std::endl;
            }
        }
        std::cout<<"[Stop] diagnostic preview finished.\n";return 0;
    } catch(const std::exception &e) { std::cerr<<e.what()<<'\n';return 1; }
}
