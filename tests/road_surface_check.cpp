#include "road_binary.hpp"
#include "track.hpp"
#include "trackview_pilot.hpp"
#include <opencv2/imgcodecs.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>

static void require(bool ok,const char *why) { if(!ok)throw std::runtime_error(why); }
int main(int argc,char **argv) {
    try {
        // Equivalent physical lanes must produce identical geometry in both modes.
        cv::Mat white(240,320,CV_8UC3,cv::Scalar(20,20,20));
        cv::rectangle(white,{70,0,180,240},cv::Scalar(220,220,220),-1);
        cv::Mat black;cv::bitwise_not(white,black);
        for(int threshold:{-1,127}) {
            auto a=roadBinary(white,threshold,false),b=roadBinary(black,threshold,true);
            require(cv::countNonZero(a!=b)==0,"Surface modes disagree on equivalent lane");
            Track t;t.handle(b);require(t.pointsEdgeLeft.size()>60&&t.pointsEdgeRight.size()>60,"Black lane has no edges");
        }
        cv::Mat glare=black.clone();
        cv::ellipse(glare,{150,130},{28,36},0,0,360,cv::Scalar(220,220,220),-1);
        auto repaired=roadBinary(glare,127,true),plain=roadBinary(glare,127,true,false);
        require(plain.at<uint8_t>(130,150)==0&&repaired.at<uint8_t>(130,150)==255,"Interior reflection not repaired");
        // A divider reaching the far border and the body reaching the near border stay excluded.
        cv::rectangle(glare,{145,0,30,80},cv::Scalar(220,220,220),-1);
        cv::rectangle(glare,{130,205,60,35},cv::Scalar(220,220,220),-1);
        repaired=roadBinary(glare,127,true);
        require(repaired.at<uint8_t>(30,155)==0&&repaired.at<uint8_t>(230,155)==0,"Boundary-connected exclusion filled");
        TrackviewConfig cfg;cfg.roadSurface="black";cfg.validate();
        require(cfg.toJson()["roadSurface"]=="black","Surface omitted from runtime log");
        cfg.roadSurface="invalid";bool rejected=false;try{cfg.validate();}catch(...){rejected=true;}
        require(rejected,"Invalid surface accepted");
        for(int i=1;i<argc;++i) {
            const std::filesystem::path path=argv[i];auto image=cv::imread(path.string());
            require(!image.empty()&&image.size()==cv::Size(320,240),"Bad field image");
            for(int mode:{0,1,2}) {
                const bool dark=mode>0,repair=mode==2;
                auto bin=roadBinary(image,-1,dark,repair);Track track;track.handle(bin);AutoPilot pilot;
                const auto profile=path.parent_path()/"width-profile.csv";
                if(std::filesystem::exists(profile)) {
                    std::ifstream input(profile);std::string line;std::getline(input,line);
                    std::vector<std::pair<int,float>> widths;
                    while(std::getline(input,line)) {
                        std::replace(line.begin(),line.end(),',',' ');std::istringstream row(line);
                        int y,observations,pending;float width;
                        if(row>>y>>width>>observations>>pending && observations>=3&&pending==0)widths.emplace_back(y,width);
                    }
                    pilot.seedCalibratedWidths(widths);
                }
                for(int n=0;n<20;++n)pilot.update(track.pointsEdgeLeft,track.pointsEdgeRight,1.f/30,0.f);
                const auto label=mode==0?"white":mode==1?"black-raw":"black";
                std::cout<<path.filename()<<" "<<label<<" left="<<track.pointsEdgeLeft.size()<<" right="<<track.pointsEdgeRight.size()
                    <<" dual="<<pilot.dualPoints<<" single="<<pilot.singlePoints<<" quality="<<pilot.confidence
                    <<" valid="<<pilot.valid<<" ready="<<pilot.ready<<" reason="<<pilot.pathIssue<<" center="<<pilot.center<<'\n';
                if(mode==2&&path.parent_path().filename()=="event-3"&&path.filename()=="870-camera.jpg")
                    require(pilot.valid&&pilot.ready&&pilot.center<130,"Field reflection still sends left bend to the right");
                if(mode==2&&path.parent_path().filename()=="event-2"&&path.filename()=="615-camera.jpg")
                    require(pilot.valid&&pilot.ready&&pilot.dualPoints>50,"Field crosswalk still cuts the lane");
                if(mode==2&&path.parent_path().filename()=="event-3"&&path.filename()=="884-camera.jpg")
                    require(!pilot.valid,"Out-of-view final frame invented a valid path");
                auto overlay=image.clone();track.drawImage(overlay);
                for(const auto &p:pilot.centerEdge)cv::circle(overlay,{p.y,p.x},1,{0,0,255},-1);
                const auto stem=path.parent_path()/(path.stem().string()+"-checked-"+label);
                cv::imwrite(stem.string()+".png",overlay);cv::imwrite(stem.string()+"-binary.png",bin);
            }
        }
        std::cout<<"PASS road surface polarity and field inspection\n";
    }catch(const std::exception &e){std::cerr<<"FAIL "<<e.what()<<'\n';return 1;}
}
