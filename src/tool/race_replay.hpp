#pragma once
#include "race_perception.hpp"
#include <fstream>
#include <map>

// 离线检测记录按视频帧编号读取；没有新记录时保留旧时间戳以验证过期保护。
class RaceReplay {
    std::map<uint64_t,std::vector<PredictResult>> records;
    RaceSnapshot latest;
public:
    explicit RaceReplay(const std::string &path) {
        std::ifstream in(path);
        if(!in) throw std::runtime_error("Cannot read detections: "+path);
        uint64_t previous=0;
        for(std::string line;std::getline(in,line);) {
            if(line.empty()) continue;
            if(line.size()>1000000) throw std::runtime_error("Detection line too large");
            const auto j=nlohmann::json::parse(line);
            const auto frame=j.at("frame").get<uint64_t>();
            if(frame<=previous||frame>10000000) throw std::invalid_argument("Detection frames must increase from 1");
            previous=frame;
            auto &items=records[frame];
            if(!j.at("detections").is_array()||j.at("detections").size()>1000)
                throw std::invalid_argument("Invalid detection list");
            for(const auto &v:j.at("detections")) {
                PredictResult d{};
                d.type=v.at("type").get<int>(); d.score=v.at("score").get<float>();
                d.x=v.at("x").get<int>(); d.y=v.at("y").get<int>();
                d.width=v.at("width").get<int>(); d.height=v.at("height").get<int>();
                items.push_back(d);
            }
        }
    }
    RaceSnapshot take(uint64_t frame,double time) {
        const auto it=records.find(frame);
        if(it!=records.end()) latest={frame,time,it->second,false,{}};
        return latest;
    }
};
