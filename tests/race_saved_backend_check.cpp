// 在目标车上用保存图片检查正式 AI 后端；不链接车辆控制或打开摄像头。
#include "race_preview.hpp"
#include <iostream>
#include <thread>

int main(int argc,char **argv) {
    try {
        if(argc!=3)throw std::invalid_argument("usage: race_saved_backend_check manifest.json output.jsonl");
        std::ifstream input(argv[1]);nlohmann::json manifest;input>>manifest;
        if(!manifest.at("images").is_array()||manifest["images"].empty()||manifest["images"].size()>32)
            throw std::invalid_argument("invalid saved-image list");
        RaceConfig cfg;auto worker=createRacePerception(cfg);
        std::ofstream output(argv[2]);if(!output)throw std::runtime_error("cannot write output");
        uint64_t sequence=0;
        for(const auto &entry:manifest["images"]) {
            const auto image=cv::imread(entry.at("path").get<std::string>());
            if(image.size()!=cv::Size(320,240)||image.type()!=CV_8UC3)throw std::runtime_error("invalid saved image");
            const auto geometryStart=raceTime();
            for(int i=0;i<10;++i) {
                const auto geometry=detectCrosswalkGeometry(image);
                if(geometry.stripes<0)throw std::runtime_error("invalid geometry result");
            }
            const double geometryMs=(raceTime()-geometryStart)*100;
            for(int repeat=0;repeat<3;++repeat) {
                const auto start=raceTime();worker->submit(image,++sequence,start);RaceSnapshot result;
                do {
                    std::this_thread::sleep_for(std::chrono::milliseconds(2));result=worker->snapshot();
                    if(result.failed)throw std::runtime_error(result.error);
                    if(raceTime()-start>10)throw std::runtime_error("saved-image inference timed out");
                }while(result.sequence!=sequence);
                auto row=raceDetectionJson(result,raceTime(),cfg);
                row["id"]=entry.at("id");row["repeat"]=repeat;row["elapsed_ms"]=(raceTime()-start)*1000;
                row["geometry_only_average_ms"]=geometryMs;
                output<<row.dump()<<'\n';output.flush();
                std::cout<<entry.at("id")<<" repeat="<<repeat<<" geometry="<<result.crosswalk.accepted<<'\n';
            }
        }
        if(!output)throw std::runtime_error("saved-image output failed");
    }catch(const std::exception &e) { std::cerr<<e.what()<<'\n';return 1; }
}
