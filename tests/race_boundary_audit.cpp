#include "race_manager.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>

// 边界审计有意保留检不出的场景，并报告覆盖缺口，不能把“程序没崩溃”当作识别通过。
int main(int argc,char **argv) {
    try {
        if(argc!=2) return 2;
        const std::filesystem::path out=argv[1]; std::filesystem::create_directories(out);
        RaceConfig cfg; cfg.stopAfterCross=false;
        nlohmann::json cases=nlohmann::json::array();
        auto road=[] {
            cv::Mat bin=cv::Mat::zeros(240,320,CV_8UC1);
            cv::rectangle(bin,{20,0,280,240},cv::Scalar(255),-1);
            return bin;
        };
        auto fork=[&] { auto b=road(); cv::rectangle(b,{135,0,50,160},cv::Scalar(0),-1); return b; };
        auto audit=[&](std::string id,std::string group,std::string description,cv::Mat bin,
                       int truth,const std::vector<cv::Rect> &regions=std::vector<cv::Rect>{}) {
            Track track; track.rowCutUp=10; track.pointsEdgeLeft={{219,20}}; track.pointsEdgeRight={{219,299}};
            const auto path=raceBranchPath(bin,track,0,cfg,regions);
            const bool detected=path.splitRows>=cfg.minBranchRows;
            const std::string status=truth<0?"ambiguous":detected==(truth!=0)?"matched":truth?"missed":"false_positive";
            // 同一画面还检查镜像，防止只对某一侧成立。
            cv::Mat mirrored; cv::flip(bin,mirrored,1);
            std::vector<cv::Rect> mirroredRegions;
            for(auto r:regions) { r.x=320-r.x-r.width; mirroredRegions.push_back(r); }
            const bool mirror=raceBranchPath(mirrored,track,0,cfg,mirroredRegions).splitRows>=cfg.minBranchRows;
            if(mirror!=detected) throw std::runtime_error("mirror asymmetry: "+id);
            if(group=="baseline"&&status!="matched") throw std::runtime_error("baseline failure: "+id);
            if(!cv::imwrite((out/(id+".png")).string(),bin)) throw std::runtime_error("image write failed");
            cases.push_back({{"id",id},{"group",group},{"description",description},{"truth",truth},
                {"detected_fork",detected},{"split_rows",path.splitRows},{"result",status},{"image",id+".png"},
                {"mirror_agrees",true}});
        };
        audit("straight","baseline","普通直道",road(),0);
        audit("solid_fork","baseline","持续至搜索远端的两条宽支路",fork(),1);
        auto circle=road(); cv::circle(circle,{160,55},60,cv::Scalar(0),12);
        audit("clipped_ring","baseline","顶部裁切的牌面圆环",circle,0);
        auto island=road(); cv::rectangle(island,{135,50,50,110},cv::Scalar(0),-1);
        audit("printed_island","baseline","白底有限黑色图案",island,0);
        // 同一二值图可以同时代表打印图案和真实的有限分隔岛：几何单独无法确定语义。
        audit("rejoining_road","rejoin","两条路在画面内重新合流，与有限黑色图案同像",island,1);
        audit("printed_vertical_bar","ambiguity","顶端被裁掉的粗竖条，与持续道路分隔带同像",fork(),-1);
        for(int width:{20,25,29,30,31,40}) {
            auto b=road(); cv::rectangle(b,{20+width,0,50,160},cv::Scalar(0),-1);
            audit("branch_width_"+std::to_string(width),"narrow","左支路宽度 "+std::to_string(width)+" 像素",b,1);
        }
        for(int gap:{5,10,11,12,13,20}) {
            auto b=road(); cv::rectangle(b,{160-gap/2,0,gap,160},cv::Scalar(0),-1);
            audit("divider_width_"+std::to_string(gap),"narrow","分隔带宽度 "+std::to_string(gap)+" 像素",b,1);
        }
        for(int height:{2,4,6,10,20}) {
            auto b=fork(); cv::rectangle(b,{0,80,320,height},cv::Scalar(0),-1);
            audit("occlusion_"+std::to_string(height),"occlusion","横向遮挡 "+std::to_string(height)+" 行",b,1);
        }
        for(int width:{80,100,120,160}) {
            cv::Mat b=cv::Mat::zeros(240,320,CV_8UC1);
            std::vector<cv::Point> surface{{20,239},{299,239},{160+width/2,0},{160-width/2,0}};
            cv::fillConvexPoly(b,surface,cv::Scalar(255));
            std::vector<cv::Point> divider{{145,0},{174,0},{184,160},{135,160}};
            cv::fillConvexPoly(b,divider,cv::Scalar(0));
            audit("perspective_"+std::to_string(width),"perspective","远端总路宽 "+std::to_string(width)+" 像素",b,1);
        }
        for(int end:{11,30,60,100}) {
            auto b=road(); cv::rectangle(b,{135,end,50,160-end},cv::Scalar(0),-1);
            audit("merge_row_"+std::to_string(end),"rejoin","分隔带在第 "+std::to_string(end)+" 行结束",b,1);
        }
        audit("roadside_sign","baseline","路侧牌框不覆盖分隔带",fork(),1,{{0,0,45,100}});
        audit("sign_over_divider","mask_overlap","新鲜牌框与真实分隔带完全重叠",fork(),1,{{120,0,80,175}});
        int matched=0,missed=0,falsePositive=0,ambiguous=0;
        for(const auto &c:cases) {
            const auto status=c["result"].get<std::string>();
            matched+=status=="matched"; missed+=status=="missed";
            falsePositive+=status=="false_positive"; ambiguous+=status=="ambiguous";
        }
        nlohmann::json report={{"cases",cases},{"case_count",cases.size()},{"matched",matched},
            {"missed",missed},{"false_positive",falsePositive},{"ambiguous",ambiguous},
            {"mirror_checks",cases.size()},{"configuration",cfg},
            {"interpretation","合成覆盖审计，不是实车准确率。missed 是需要进一步验证的能力缺口。"}};
        std::ofstream file(out/"boundary-audit.json"); file<<report.dump(2);
        if(!file) throw std::runtime_error("report write failed");
        std::cout<<"AUDIT cases="<<cases.size()<<" matched="<<matched<<" missed="<<missed
                 <<" false_positive="<<falsePositive<<" ambiguous="<<ambiguous
                 <<" mirror_checks="<<cases.size()<<" baseline_checks=passed\n";
    } catch(const std::exception &e) { std::cerr<<"FAIL "<<e.what()<<'\n'; return 1; }
}
