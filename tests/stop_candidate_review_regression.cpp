#include "stop_candidate_review.hpp"
#include <opencv2/imgcodecs.hpp>
#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <filesystem>

void check(bool value,const char *message) { if(!value)throw std::runtime_error(message); }
int main(int argc,char **argv) {
    try {
        check(argc==1||argc==3,"usage: stop_candidate_review_regression [probe.jsonl saved-sample-root]");
        cv::Mat image(240,320,CV_8UC3,cv::Scalar(50,50,50));
        const cv::Rect box(50,40,120,120);
        cv::circle(image,{110,100},58,{35,35,200},-1);
        check(!reviewStopCandidate(image,box).matches,"solid red without white bar must fail");
        cv::rectangle(image,{65,90,90,20},{220,220,220},-1);
        check(reviewStopCandidate(image,box).matches,"red sign with central white bar must pass");
        check(!reviewStopCandidate(image,{-5,40,120,120}).matches,"out of bounds box must fail");
        cv::Mat green(240,320,CV_8UC3,cv::Scalar(35,180,35));
        cv::rectangle(green,{65,90,90,20},{220,220,220},-1);
        check(!reviewStopCandidate(green,box).matches,"green sign cannot substitute for red stop");
        int real=0,passed=0;
        if(argc==3) {
            std::ifstream input(argv[1]);check(bool(input),"recorded NMS result missing");
            for(std::string line;std::getline(input,line);) {
                auto row=nlohmann::json::parse(line);
                if(row.at("repeat").get<int>()!=0)continue;
                // 样本目录由调用方提供，不依赖某台电脑的盘符和采样日期。
                auto id=row.at("id").get<std::string>();
                if(id.rfind("sample-",0)!=0)continue;
                const auto sample=std::filesystem::path(argv[2])/("ai-sample-"+id.substr(7))/"camera.png";
                auto frame=cv::imread(sample.string());check(!frame.empty(),"saved camera image missing");
                for(const auto &candidate:row.at("candidates")) if(candidate.at("type").get<int>()==5) {
                    ++real;
                    const auto review=reviewStopCandidate(frame,{candidate.at("x"),candidate.at("y"),candidate.at("width"),candidate.at("height")});
                    passed+=review.matches;
                    check(review.matches,"saved stop candidate did not pass appearance review");
                }
            }
            check(real==10&&passed==10,"expected ten recorded stop candidates");
        }
        std::cout<<"PASS stop appearance: synthetic red/no-bar/green/bounds and "<<passed<<'/'<<real<<" saved candidates\n";
    }catch(const std::exception &e) { std::cerr<<"FAIL "<<e.what()<<'\n';return 1; }
}
