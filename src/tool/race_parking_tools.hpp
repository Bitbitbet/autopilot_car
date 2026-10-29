#pragma once
#include "race_parking_geometry.hpp"
#include "race_ground.hpp"
#include "race_trial.hpp"
#include "control.hpp"
#include <atomic>
#include <memory>

// Explicit diagnostic only. It never claims negative commands prove real reversing.
template<class Preprocess,class StopFlag>
int raceCheckReverse(cv::VideoCapture &video,Preprocess &predeal,StopFlag &stop) {
    auto car=CarControl::create();if(!car)throw std::runtime_error("Reverse check cannot open vehicle control");
    struct Guard {std::shared_ptr<CarControl> car;~Guard(){try{car->latchStop();}catch(...){}}} guard{car};
    car->setCommandTimeout(250);car->resetVelocity();
    RaceTrialDeadline deadline(.5,[car]{car->pauseMotion();});
    const char *window="reverse check - ONE OPERATOR TRIGGERED PULSE";
    cv::namedWindow(window,cv::WINDOW_AUTOSIZE);cv::Mat image;
    std::cout<<"倒车短测：只有按r才发送一次-0.08 m/s、0.5秒的直行倒车指令，计时独立于界面。\n"
        <<"请放在车后留有空间的直道，观察实际车身移动。完成后按y确认确实后退，n记录不能倒车；空格或q立即结束。\n"
        <<"负速度指令不是实测倒车能力；本模式不自动转向或执行入库。\n";
    bool resultSaved=false;int status=1;
    try {
        while(!stop.load()) {
            if(!video.read(image)||image.empty())throw std::runtime_error("Reverse check camera unavailable");
            if(image.size()!=cv::Size(320,240))cv::resize(image,image,{320,240});
            predeal.correct(image);
            const bool running=deadline.armed()&&!deadline.expired();
            if(car->hasFault())throw std::runtime_error("Reverse check serial fault; no capability confirmation saved");
            car->carControl(running?-.08f:0.f,PWMSERVOMID);
            cv::Mat shown;cv::resize(image,shown,{640,480});
            cv::putText(shown,running?"REVERSE PULSE":deadline.expired()?"STOPPED: y=observed reverse / n=not reversed":"r=one 0.5s pulse; q/SPACE=quit",{8,24},cv::FONT_HERSHEY_PLAIN,1,{0,255,255});
            cv::imshow(window,shown);const int key=cv::waitKey(10)&255;
            if(key==27||key=='q'||key=='Q'||key==' ')break;
            if(key=='r'&&!deadline.armed()) {
                // Invalidate an older human confirmation before a new physical test.
                std::ofstream pending("parking-reverse-check.json.tmp");
                pending<<nlohmann::json({{"version",1},{"reverse_verified",false},{"pending",true},
                    {"verification_source","new pulse requested; physical result not confirmed"}}).dump(2)<<'\n';pending.close();
                if(!pending)throw std::runtime_error("Cannot record pending reverse check");
                std::filesystem::rename("parking-reverse-check.json.tmp","parking-reverse-check.json");
                cv::imwrite("parking-reverse-before.png",image);deadline.arm();
            }
            if((key=='y'||key=='n')&&deadline.expired()) {
                car->pauseMotion();
                nlohmann::json result={{"version",1},{"reverse_verified",key=='y'},
                    {"verification_source","operator observed vehicle motion; not inferred from command"},
                    {"command_speed_mps",-.08},{"pulse_seconds",.5},{"measured_speed_mps",nullptr},
                    {"vehicle_scope","current physical vehicle only"}};
                std::ofstream file("parking-reverse-check.json.tmp");file<<result.dump(2)<<'\n';file.close();
                if(!file)throw std::runtime_error("Cannot save reverse check");
                std::filesystem::rename("parking-reverse-check.json.tmp","parking-reverse-check.json");
                cv::imwrite("parking-reverse-after.png",image);resultSaved=true;status=key=='y'?0:1;
                std::cout<<"[Parking] Human reverse result saved: "<<(key=='y'?"VERIFIED":"NOT VERIFIED")<<'\n';break;
            }
        }
        deadline.finish();car->latchStop();cv::destroyWindow(window);
    }catch(...){deadline.finish();cv::destroyWindow(window);throw;}
    if(!resultSaved)std::cout<<"[Parking] Reverse capability remains unconfirmed; no new result saved.\n";
    return status;
}

// Camera-only taught geometry preview. Two mouth corners locate the known bay in
// the current ground frame; the photograph is never used as a metric calibration.
template<class Preprocess,class StopFlag>
int racePreviewParking(cv::VideoCapture &video,Preprocess &predeal,const RaceGroundCalibration &calibration,StopFlag &stop) {
    RaceParkingDimensions dims;RaceParkingSpace space(dims);RaceGroundMap map(calibration);
    struct Clicks {bool frozen=false;std::vector<cv::Point2f> points;};Clicks state;
    const char *window="parking bay 2 preview - NO MOTOR";cv::namedWindow(window,cv::WINDOW_AUTOSIZE);
    cv::setMouseCallback(window,[](int event,int x,int y,int,void *context){
        auto &s=*static_cast<Clicks*>(context);
        if(event==cv::EVENT_LBUTTONDOWN&&s.frozen&&s.points.size()<2&&x<640&&y<480&&x>=0&&y>=0)
            s.points.emplace_back(x/2.f,y/2.f);
    },&state);
    cv::Mat image;RaceParkingRoute route;cv::Point2f origin,along{0,1},right{1,0};bool taught=false;
    std::cout<<"停车路径预览，不连接电机。小车摆在停车区中央通道、2号位之前，车头朝出口方向。\n"
        <<"f冻结，点击2号车位靠通道的两个口角：近角、远角；p规划；s保存预览；r重选；q退出。\n"
        <<"尺寸：通道45cm，车位四边41cm、垂直进深38cm；停车3秒。图中路线仅是几何预览，不能当成已执行停车。\n";
    auto world=[&](cv::Point2f p){return origin+right*(p.x+dims.aisle/2)+along*p.y;};
    while(!stop.load()) {
        if(!state.frozen){if(!video.read(image)||image.empty())throw std::runtime_error("Parking preview camera unavailable");
            if(image.size()!=cv::Size(320,240))cv::resize(image,image,{320,240});
            predeal.correct(image);}
        cv::Mat shown;cv::resize(image,shown,{640,480});
        for(size_t i=0;i<state.points.size();++i){cv::circle(shown,state.points[i]*2,5,{0,0,255},-1);cv::putText(shown,std::to_string(i+1),state.points[i]*2+cv::Point2f(6,-6),cv::FONT_HERSHEY_PLAIN,1,{0,0,255});}
        if(taught) {
                auto visible=[](cv::Point2f p){return std::isfinite(p.x)&&std::isfinite(p.y)&&std::abs(p.x)<5000&&std::abs(p.y)<5000;};
                auto draw=[&](const std::vector<RaceParkingPose> &poses,cv::Scalar color) {
                for(size_t i=1;i<poses.size();++i) {
                    const auto a=map.toImage(world({poses[i-1].x,poses[i-1].z})),b=map.toImage(world({poses[i].x,poses[i].z}));
                    if(visible(a)&&visible(b))cv::line(shown,a*2,b*2,color,2);
                }
            };
            const auto bay=space.bay();for(size_t i=0;i<bay.size();++i){const auto a=map.toImage(world(bay[i])),b=map.toImage(world(bay[(i+1)%bay.size()]));
                if(visible(a)&&visible(b))cv::line(shown,a*2,b*2,{0,255,255},2);}
            draw(route.entry,{0,255,0});
        }
        cv::putText(shown,taught?route.reason:"f freeze | near/far mouth corners | p plan",{5,22},cv::FONT_HERSHEY_PLAIN,1,{0,0,255});
        cv::putText(shown,"NO MOTOR | experimental steering geometry | s save r reset q quit",{5,465},cv::FONT_HERSHEY_PLAIN,.9,{0,0,255});
        cv::imshow(window,shown);const int key=cv::waitKey(20)&255;
        if(key=='q'||key==27)break;
        if(key=='f')state.frozen=true;
        if(key=='r'){state.points.clear();state.frozen=false;taught=false;route={};map.setMetricScale(1);}
        if(key=='p'&&state.points.size()==2) {
            try {
                map.setMetricScale(1);auto a=map.toGround(state.points[0]),b=map.toGround(state.points[1]);
                const float length=cv::norm(b-a),scale=dims.side/length;
                if(!std::isfinite(scale)||scale<.8f||scale>1.3f)throw std::runtime_error("Mouth length contradicts ground calibration; recheck corners/calibration");
                map.setMetricScale(scale);a=map.toGround(state.points[0]);b=map.toGround(state.points[1]);
                if(b.y<=a.y||std::abs(b.x-a.x)>.10f)throw std::runtime_error("Face forward along aisle; click near then far mouth corner");
                origin=a;along=(b-a)*(1.f/dims.side);right={along.y,-along.x};
                const auto offset=-origin;
                const RaceParkingPose start{offset.dot(right)-dims.aisle/2,offset.dot(along),std::atan2(along.x,along.y)};
                route=racePlanParking(space,start);taught=true;
                std::cout<<"[Parking] "<<route.reason<<" nodes="<<route.expanded<<" scale="<<scale<<'\n';
            }catch(const std::exception &e){taught=false;route={};std::cerr<<"[Parking] "<<e.what()<<'\n';}
        }
        if(key=='s'&&taught) {
            nlohmann::json entry=nlohmann::json::array(),exit=entry;
            for(const auto &p:route.entry)entry.push_back({p.x,p.z,p.yaw});
            for(const auto &p:route.exit)exit.push_back({p.x,p.z,p.yaw});
            nlohmann::json result={{"version",1},{"bay",2},{"side_m",dims.side},{"depth_m",dims.depth},
                {"stagger_m",dims.stagger()},{"aisle_m",dims.aisle},{"stop_seconds",3},
                {"all_bays_assumed_empty",true},{"preview_only",true},{"drives_vehicle",false},
                {"route_found",route.valid},{"reason",route.reason},{"entry",entry},{"reverse_exit",exit},
                {"front_to_rear_axle_experimental_m",dims.frontToRearAxle},{"full_lock_experimental_degrees",dims.fullLockDegrees},
                {"ground_scale",map.scale()},{"lens_fingerprint",calibration.lensFingerprint},
                {"mouth_image_points",{{state.points[0].x,state.points[0].y},{state.points[1].x,state.points[1].y}}},
                {"lane_origin_ground_m",{origin.x,origin.y}},{"lane_forward_axis",{along.x,along.y}}};
            std::ofstream output("parking-preview.json");output<<result.dump(2)<<'\n';output.close();
            if(!output)throw std::runtime_error("Cannot save parking preview");
            cv::imwrite("parking-preview-camera.png",image);cv::imwrite("parking-preview-overlay.png",shown);
            std::cout<<"[Parking] Saved parking-preview.json and camera/overlay; preview only.\n";
        }
    }
    cv::destroyWindow(window);return 0;
}
