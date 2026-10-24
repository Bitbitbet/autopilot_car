// 仅供保存图片诊断，不创建 CarControl、VideoCapture 或手柄设备。
// 在独立目录内复用原 Detection，从 NMS 原始输出读取当前阈值以下的候选。
#include "detection.hpp"
#include "stop_candidate_review.hpp"
#include <nlohmann/json.hpp>
#include <opencv2/imgcodecs.hpp>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>

extern "C" int raceImageProbe(const char *manifestPath,const char *outputPath,
                               char *error,size_t errorCapacity) noexcept {
    try {
        namespace fs=std::filesystem;
        std::ifstream input(manifestPath);nlohmann::json manifest;
        if(!input) throw std::runtime_error("Cannot read saved-image manifest");
        input>>manifest;
        const auto entries=manifest.at("images");
        const int repeats=manifest.value("repeats",3);
        if(!entries.is_array()||entries.empty()||entries.size()>32||repeats<1||repeats>3)
            throw std::runtime_error("Saved-image probe bounds exceeded");
        // 配置必须是准备脚本在隔离目录中生成的固定相对路径。
        const std::string model="../res/models/yolov3_mobilenet_v1";
        Detection detector(model,0.f);
        std::ofstream output(outputPath);
        if(!output) throw std::runtime_error("Cannot create probe result file");
        for(const auto &entry:entries) {
            const auto file=entry.at("path").get<std::string>();
            const auto image=cv::imread(file,cv::IMREAD_COLOR);
            if(image.empty()||image.size()!=cv::Size(320,240))
                throw std::runtime_error("Expected unchanged 320x240 saved camera image: "+file);
            for(int repeat=0;repeat<repeats;++repeat) {
                detector.inference(image);
                auto raw=detector.get_output(0);
                int64_t count=1;
                for(auto dimension:raw.shape) {
                    if(dimension<0||dimension>24576||count>24576/std::max<int64_t>(dimension,1))
                        throw std::runtime_error("Unexpected NMS output shape");
                    count*=dimension;
                }
                if(count%6!=0) throw std::runtime_error("NMS output must contain six floats per row");
                auto data=raw.value();
                auto candidates=nlohmann::json::array();
                float bestStop=0;
                for(int64_t i=0;i<count;i+=6) {
                    bool finite=true;
                    for(int j=0;j<6;++j) finite&=std::isfinite(data[i+j]);
                    if(!finite||data[i]<0||data[i]>14||data[i+1]<=0||data[i+1]>1) continue;
                    const int type=static_cast<int>(data[i]);const float score=data[i+1];
                    const int x=static_cast<int>(data[i+2]),y=static_cast<int>(data[i+3]);
                    const int width=static_cast<int>(data[i+4]-data[i+2]);
                    const int height=static_cast<int>(data[i+5]-data[i+3]);
                    const bool valid=x>=0&&y>=0&&width>0&&height>0&&x<320&&y<240&&width<=320-x&&height<=240-y;
                    if(type==5) bestStop=std::max(bestStop,score);
                    const auto review=type==5&&valid?reviewStopCandidate(image,{x,y,width,height}):StopCandidateReview{};
                    candidates.push_back({{"type",type},{"score",score},{"raw_xyxy",{data[i+2],data[i+3],data[i+4],data[i+5]}},
                        {"x",x},{"y",y},{"width",width},{"height",height},{"valid_box",valid},
                        {"passes_065",score>=.65f},{"stop_near_145",type==5&&valid&&y+height>=145},
                        {"appearance_match",review.matches},{"appearance_reason",review.reason},{"red_fraction",review.redFraction}});
                }
                output<<nlohmann::json({{"id",entry.at("id")},{"path",file},{"repeat",repeat},
                    {"best_stop_score",bestStop},{"raw_shape",raw.shape},{"candidates",candidates},
                    {"source","original NMS output; NMS may have its own internal threshold; not pre-NMS scores"}}).dump()<<'\n';
                output.flush();
                if(!output) throw std::runtime_error("Probe output write failed");
            }
        }
        return 0;
    } catch(const std::exception &e) {
        if(error&&errorCapacity) { std::strncpy(error,e.what(),errorCapacity-1);error[errorCapacity-1]='\0'; }
        return 1;
    } catch(...) {
        if(error&&errorCapacity) { std::strncpy(error,"Unknown saved-image probe failure",errorCapacity-1);error[errorCapacity-1]='\0'; }
        return 2;
    }
}
