#pragma once
#include "race_runtime.hpp"
#include "race_mission.hpp"
#include "race_virtual_boundary.hpp"
#include "race_parking_actuator.hpp"
#include "race_parking_control.hpp"
#include "race_parking_detection.hpp"
#include "race_parking_test.hpp"
#include "race_ground_motion.hpp"
#include "race_manager.hpp"
#include "trackview_pilot.hpp"
#include "trackview_display.hpp"
#include "race_ui.hpp"
#include "control.hpp"
#include <filesystem>
#include <fstream>
#include <thread>
#include <future>
#include <deque>

// The established v6 convention is image-right -> PWM above midpoint. Parking
// bicycle curvature is positive LEFT, so its sign must be inverted here.
inline uint16_t raceParkingServo(float curvature,const RaceParkingDimensions &d) {
    const float angle=std::atan(d.wheelbase*curvature),lock=d.fullLockDegrees*float(CV_PI)/180;
    return uint16_t(std::lround(std::clamp(float(PWMSERVOMID)-400*angle/lock,float(PWMSERVOMIN),float(PWMSERVOMAX))));
}

template<class Preprocess,class StopFlag>
int raceDriveParking(cv::VideoCapture &video,Preprocess &predeal,const RaceGroundCalibration &calibration,
                     const TrackviewConfig &config,const RaceConfig &raceConfig,bool reverseVerified,StopFlag &stop,bool missionEnabled=false,bool labMission=false,bool previewOnly=false) {
    if(!labMission&&!reverseVerified&&!previewOnly)throw std::runtime_error("Parking requires observed reverse capability, not just a negative command");
    if(!config.diagnostics)throw std::runtime_error("Parking drive requires diagnostics=true");
    auto perception=createRacePerception(raceConfig);
    const auto stamp=std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    const auto directory=std::filesystem::path(config.logDirectory)/((missionEnabled?"mission-":"parking-")+std::to_string(stamp));
    std::filesystem::create_directories(directory);
    std::ofstream telemetry(directory/"parking-control.jsonl");if(!telemetry)throw std::runtime_error("Cannot open parking telemetry");
    if(!labMission)std::ofstream(directory/"ground-calibration.json")<<nlohmann::json(calibration).dump(2);
    std::ofstream(directory/"trackview-config.json")<<config.toJson().dump(2);
    std::ofstream(directory/"race-config.json")<<nlohmann::json(raceConfig).dump(2);
    auto car=std::make_shared<RaceParkingActuator>(!previewOnly);
    struct Guard {std::shared_ptr<RaceParkingActuator> car;~Guard(){try{car->latchStop();}catch(...){}}} guard{car};
    car->setCommandTimeout(250);car->pauseMotion();
    TrackviewDisplay display;
    struct Input {std::mutex mutex;std::vector<cv::Point2f> clicks;} input;
    const char *window=missionEnabled?"mission - a start | n at start: lap/finish | SPACE stop | s sample":
        "parking - AUTO visible left/right bay | a run | f manual | g rescan";
    // Explicit client sizing avoids clipped AUTO-sized windows with remote X11
    // and Windows DPI scaling; the status strip is part of this same image.
    cv::namedWindow(window,cv::WINDOW_NORMAL|cv::WINDOW_KEEPRATIO);
    cv::resizeWindow(window,640,560);
    cv::setMouseCallback(window,[](int event,int x,int y,int,void *data){
        if(event!=cv::EVENT_LBUTTONDOWN||x<0||x>=640||y<0||y>=480)return;
        auto &q=*static_cast<Input*>(data);std::lock_guard lock(q.mutex);
        if(q.clicks.size()<2)q.clicks.emplace_back(float(x),float(y));
    },&input);
    if(missionEnabled)std::cout<<"[Mission] "<<directory<<"\n"<<(previewOnly?"NO MOTOR | ":"")
        <<(labMission?"实验室单圈；不套用比赛停车路线。":"比赛两圈；第一圈停车/出库左转，第二圈走外圈。")
        <<" a开始；n仅在实际起点确认计圈/终点；空格停；g复位；s保存当前图。\n";
    else std::cout<<"[Parking] "<<directory<<"\n默认从当前车载图自动找左右两侧完整可见车位口并规划，不需逐次点选；出现黄色车位和绿色路线后按a执行。\n"
        <<"按g重新识别；f进入手动点选诊断。后方有道闸时不要求后退，车身扫掠不得超过起始后沿。\n"
        <<"入库->实测停稳3秒->倒车回通道->恢复原巡线向前。空格立即停；中断后r重新点位。s保存静止样本。\n"
        <<"入口左转和出口地面箭头转弯尚待车载样本，不能把本次车位动作当作整场自动停车验收。\n";
    std::jthread driving([&]{
        RaceParkingTestReport test;
        auto saveTest=[&]{std::ofstream(directory/"parking-test-summary.json")<<test.toJson().dump(2);};
        try {
            RaceParkingDimensions dims;RaceParkingSpace space(dims);RaceGroundMap map=labMission?RaceGroundMap{}:RaceGroundMap(calibration);
            RaceParkingControl controller(space,raceConfig.parkingSpeed);RaceParkingPoseFallback poseFallback;
            RaceManager manager(raceConfig);RaceTaskPilot pilot(config),virtualPilot(config);
            RaceVirtualSteeringFeedback virtualFeedback;
            RaceMission mission(labMission);bool missionAutomatic=false;
            RaceNavigationConfig navConfig;RaceVirtualBoundary virtualBoundary(navConfig);Track matchedTrack;
            std::ofstream missionLog;if(missionEnabled)missionLog.open(directory/"mission-state.jsonl");
            Track track;track.rowCutUp=config.rowCutUp;track.rowCutBottom=config.rowCutBottom;
            std::unique_ptr<RaceGroundMotion> motion;
            RaceGroundMotionResult measured;RaceParkingRoute route;
            cv::Mat frozenImage;std::vector<cv::Point2f> corners;
            RaceParkingFrame bayFrame;bool frozen=false,taught=false,armed=false,continueLane=false;
            double last=raceTime(),completedAt=-1;uint64_t frame=0;uint16_t previousServo=PWMSERVOMID;
            float cruise=.10f,previousSpeed=0;std::string previousState;
            std::string firstMotionFailure;
            std::string teachingFeedback,teachingDetails;
            bool autoEnabled=!missionEnabled,autoPlanning=false;double lastScan=-1e30;int stableMouths=0;
            RaceParkingMouthCandidate previousMouth;std::vector<RaceParkingMouthCandidate> autoMouths;
            RaceParkingBaySearch baySearch;bool searchAdvance=false,searchRequested=false;
            bool standaloneExitGateSeen=false,standaloneExitReleased=false;
            std::ofstream searchLog(directory/"parking-bay-search.jsonl");
            cv::Size cameraSize;
            struct EvidenceFrame {cv::Mat image;nlohmann::json metadata;};
            std::deque<EvidenceFrame> recent;
            std::future<void> evidenceWriter;int evidenceCount=0;
            auto world=[&](cv::Point2f p){return bayFrame.world(p,dims.aisle);};
            while(!stop.load()) {
                if(car->hasFault())throw std::runtime_error("Parking serial watchdog/fault: restart required");
                cv::Mat image;if(!video.read(image)||image.empty())throw std::runtime_error("Parking camera unavailable");
                const double captured=raceTime();
                if(image.size()!=cameraSize) {
                    cameraSize=image.size();
                    std::ofstream(directory/"camera-input.json")<<nlohmann::json({{"raw_width",cameraSize.width},{"raw_height",cameraSize.height},
                        {"processing_width",640},{"processing_height",480},{"display_width",640},{"display_height",560},
                        {"resize_before_lens_correction",false},{"lens_fingerprint",calibration.lensFingerprint}}).dump(2);
                    std::cout<<"[Parking camera] input="<<cameraSize.width<<'x'<<cameraSize.height<<" -> processing=640x480; display=640x560\n";
                }
                if(cameraSize.width*3!=cameraSize.height*4)
                    throw std::runtime_error("Parking camera aspect ratio changed: restore calibrated 4:3 capture or recalibrate; refusing stretched geometry");
                raceNormalizeFrame(image);
                predeal.correct(image);++frame;perception->submit(image,frame,captured);
                const auto bin=predeal.binarize(image);track.handle(bin);
                bool mouthsScanned=false;std::vector<RaceParkingMouthCandidate> liveMouths;
                auto scanMouths=[&]() -> const std::vector<RaceParkingMouthCandidate>& {
                    if(!mouthsScanned){liveMouths=raceDetectParkingMouths(image,map,dims);mouthsScanned=true;}return liveMouths;
                };
                const auto ai=perception->snapshot();
                const double before=raceTime();float dt=float(before-last);last=before;
                if(!armed&&dt>0&&dt<=.25f) {
                    const float angle=(float(PWMSERVOMID)-previousServo)/400*dims.fullLockDegrees*float(CV_PI)/180;
                    const float distance=previousSpeed*dt;
                    baySearch.advance(distance,distance*std::tan(angle)/dims.wheelbase);
                }
                RaceObstaclePreviewPlan virtualPlan;RaceBypassDecision virtualDecision;
                if(missionEnabled&&!mission.parkingSearch()&&!mission.parkingRun()&&!ai.image.empty()) {
                    const auto matchedBinary=predeal.binarize(ai.image);matchedTrack.handle(matchedBinary);
                    virtualPlan=virtualBoundary.update(ai,before,matchedBinary,matchedTrack,raceConfig,missionAutomatic,previousSpeed);
                    virtualDecision=virtualBoundary.decision(virtualPlan,missionAutomatic);
                }
                if(missionEnabled)mission.observeSigns(ai,manager.aiReady(ai,before));
                RaceGroundArrow groundArrow;
                if(missionEnabled){
                    groundArrow=raceGroundArrow(image,track,config.roadSurface=="black");
                    mission.observeExitArrow(groundArrow,frame,before-captured<=.18);
                }
                auto traffic=manager.update(ai,before,dt,missionEnabled?missionAutomatic:true,bin,track,image,
                    virtualDecision.authorizeCone,true,missionEnabled?mission.routeDirection():0,missionEnabled,
                    missionEnabled&&((mission.parkingApproach()&&mission.routeDirection()==-1)||mission.exitArrowTurnAuthorized()),
                    missionEnabled&&mission.parkingSpeedLimited()?raceConfig.parkingSpeed:0);
                if(missionEnabled)mission.observeGate(traffic);
                else if(controller.done()) {
                    if(traffic.gateWaiting)standaloneExitGateSeen=true;
                    if(standaloneExitGateSeen&&traffic.gateOpened)standaloneExitReleased=true;
                }
                const float taskCruise=missionEnabled?mission.speedLimit(cruise,raceConfig.parkingSpeed):
                    standaloneExitReleased?cruise:std::min(cruise,raceConfig.parkingSpeed);
                const bool areaLimited=missionEnabled?mission.parkingSpeedLimited():!standaloneExitReleased;
                pilot.setCurveSpeed(areaLimited?std::min(taskCruise,raceConfig.parkingSpeed):config.curveSpeed);
                if(missionEnabled) {
                    auto visibleCandidates=missionAutomatic&&mission.parkingZone()&&!taught?
                        scanMouths():std::vector<RaceParkingMouthCandidate>{};
                    baySearch.filter(visibleCandidates);
                    const bool visibleMouth=!visibleCandidates.empty();
                    mission.observe(ai,traffic,visibleMouth);
                    if(mission.parkingSearch()&&!taught&&!frozen&&!controller.failed()) {
                        if(continueLane){car->pauseMotion();previousSpeed=0;}
                        continueLane=false;searchAdvance=false;autoEnabled=true;
                    }
                }
                if(motion) {RaceSnapshot current;current.image=image;current.sequence=frame;current.captured=captured;
                    if(traffic.fresh)current.detections=ai.detections; // conservatively exclude recent objects from ground texture
                    // No zero-motion reanchoring: parking cannot lose a piece of the path.
                    measured=motion->update(current,true,false);}
                RaceParkingObservation observation;
                if(taught) {
                    observation={bayFrame.local({measured.pose.x,measured.pose.z,measured.pose.angle},dims.aisle),captured,frame,measured.epoch,measured.valid};
                }
                if(display.failed.load())throw std::runtime_error("Parking display failed");
                const auto commandEpoch=car->motionEpoch();
                {std::lock_guard lock(input.mutex);if(frozen){for(auto p:input.clicks)if(corners.size()<2)corners.push_back(p);}input.clicks.clear();}
                auto keys=display.takeKeys();
                if(missionEnabled&&mission.parkingSearch()&&taught&&route.valid&&measured.valid&&!traffic.stop)keys.push_back('a');
                if(!missionEnabled&&searchRequested&&taught&&route.valid&&!armed&&measured.valid&&!traffic.stop)keys.push_back('a');
                if(std::find(keys.begin(),keys.end(),' ')==keys.end()&&autoEnabled&&!taught&&!frozen&&!armed&&(!continueLane||searchAdvance)&&!controller.failed()&&before-lastScan>=.1) {
                    const double scanGap=before-lastScan;lastScan=before;map.setMetricScale(1);autoMouths=scanMouths();
                    baySearch.filter(autoMouths);
                    if(autoMouths.empty()) {stableMouths=0;teachingFeedback="AUTO_WAIT_FULL_VISIBLE_BAY";}
                    else {
                        const auto &m=autoMouths.front();
                        const float travel=std::abs(previousSpeed)*std::min(float(scanGap),.3f);
                        const bool same=stableMouths&&m.side==previousMouth.side&&cv::norm(m.nearGround-previousMouth.nearGround)<.015+travel&&
                            cv::norm(m.farGround-previousMouth.farGround)<.015+travel;
                        stableMouths=same?stableMouths+1:1;previousMouth=m;
                        teachingFeedback="AUTO_CONFIRMING_BAY_MARKINGS";
                        if(stableMouths>=3) {
                            if(searchAdvance){car->pauseMotion();previousSpeed=0;continueLane=false;searchAdvance=false;}
                            frozenImage=image.clone();frozen=true;autoPlanning=true;autoEnabled=false;
                            corners={m.nearImage,m.farImage};keys.push_back('p');
                        }
                    }
                }
                for(int key:keys) {
                    if(key>='A'&&key<='Z')key=key-'A'+'a';
                    if(labMission&&(key=='f'||key=='p'))continue;
                    if(key==' ') {if(missionEnabled){mission.stop();missionAutomatic=false;}car->pauseMotion();controller.abort();armed=false;continueLane=false;searchAdvance=searchRequested=false;autoEnabled=false;previousSpeed=0;}
                    else if(missionEnabled&&key=='n')mission.markStart();
                    else if(missionEnabled&&key=='a'&&!mission.running()&&!mission.failed()&&!mission.complete()) {
                        if(manager.aiReady(ai,before)&&pilot.ready&&car->resumeMotion(commandEpoch)) {
                            mission.start();missionAutomatic=true;continueLane=true;autoEnabled=false;
                        }
                    }
                    else if((key=='r'||key=='g')&&!armed&&(!continueLane||(missionEnabled&&(!missionAutomatic||mission.complete()||mission.failed())))) {
                        if(missionEnabled){mission=RaceMission(labMission);missionAutomatic=false;continueLane=false;virtualBoundary=RaceVirtualBoundary(navConfig);}
                        test.reset();saveTest();
                        car->pauseMotion();space=RaceParkingSpace(dims);controller=RaceParkingControl(space,raceConfig.parkingSpeed);route={};motion.reset();measured={};taught=frozen=false;corners.clear();map.setMetricScale(1);completedAt=-1;firstMotionFailure.clear();recent.clear();teachingFeedback.clear();teachingDetails.clear();
                        standaloneExitGateSeen=standaloneExitReleased=false;
                        autoEnabled=!missionEnabled&&key=='g';autoPlanning=false;stableMouths=0;
                        baySearch.reset();searchAdvance=searchRequested=false;
                        std::lock_guard lock(input.mutex);input.clicks.clear();
                    } else if(key=='f'&&!armed&&!continueLane&&!taught) {
                        frozen=true;autoEnabled=autoPlanning=false;frozenImage=image.clone();corners.clear();teachingFeedback.clear();teachingDetails.clear();std::lock_guard lock(input.mutex);input.clicks.clear();
                    } else if(key=='p') {
                        if(armed||continueLane){std::cout<<"[Parking] Stop before replanning.\n";continue;}
                        if(taught&&route.valid&&!controller.failed()){teachingFeedback="A_START_PARKING";std::cout<<"[Parking] Route ready; press a to run or g to rescan.\n";continue;}
                        if(!frozen){teachingFeedback="P_PRESS_F_TO_FREEZE";continue;}
                        if(corners.size()!=2){teachingFeedback="P_CLICK_TWO_POINTS_FIRST";continue;}
                        std::vector<std::vector<cv::Point2f>> choices{corners};
                        if(autoPlanning){choices.clear();for(const auto &m:autoMouths)choices.push_back({m.nearImage,m.farImage});}
                        for(const auto &choice:choices) {
                        corners=choice;
                        map.setMetricScale(1);auto a=map.toGround(corners[0]),b=map.toGround(corners[1]);
                        const auto rawNear=a,rawFar=b;
                        const auto check=raceCheckParkingMouth(a,b,dims);const float scale=check.scale;
                        teachingFeedback=check.reason;
                        teachingDetails="near="+std::to_string(std::lround(check.near.y*100))+"cm, far="+
                            std::to_string(std::lround(check.far.y*100))+"cm";
                        nlohmann::json attempt={{"frame",frame},{"reason",check.reason},{"valid",check.valid},
                            {"automatic",autoPlanning},{"image_points",{{corners[0].x,corners[0].y},{corners[1].x,corners[1].y}}},
                            {"raw_ground_points",{{a.x,a.y},{b.x,b.y}}},{"scale",scale},
                            {"parking_clearance_m",dims.clearance},{"goal_extra_margin_m",0.0},
                            {"predicted_boundary_policy","warning_continue"},{"route_preference","early_turn_into_selected_bay_far_divider_clearance"},
                            {"scaled_near_m",{check.near.x,check.near.y}},{"scaled_far_m",{check.far.x,check.far.y}}};
                        std::ofstream(directory/"parking-teaching-attempt.json")<<attempt.dump(2);
                        cv::imwrite((directory/"parking-teaching-attempt.png").string(),frozenImage);
                        if(!check.valid) {
                            if(autoPlanning)baySearch.reject(rawNear,rawFar);
                            test.planned(false,check.reason);saveTest();
                            std::cout<<"[Parking] "<<teachingFeedback<<" | "<<teachingDetails<<"; r reset, f freeze, reselect.\n";
                            continue;
                        }
                        map.setMetricScale(scale);a=check.near;b=check.far;
                        bayFrame.locate(a,b);const auto start=bayFrame.local({},dims.aisle);
                        space=RaceParkingSpace(dims);space.keepRearWithinStart(start);controller=RaceParkingControl(space,raceConfig.parkingSpeed);
                        route=racePlanTrackableParking(space,start,raceConfig.parkingSpeed);
                        test.planned(route.valid,route.reason);saveTest();
                        teachingFeedback=route.valid?"A_START_PARKING":route.reason;
                        teachingDetails=std::string("nodes=")+std::to_string(route.expanded)+" candidates="+std::to_string(route.geometricCandidates)+" rejected="+std::to_string(route.modelRejected);
                        attempt["planner_reason"]=route.reason;attempt["expanded"]=route.expanded;
                        attempt["geometric_candidates"]=route.geometricCandidates;attempt["model_rejected"]=route.modelRejected;
                        attempt["searches"]=nlohmann::json::array();
                        for(const auto &s:route.searches)attempt["searches"].push_back({{"curvature_scale",s.curvatureScale},
                            {"heuristic_weight",s.heuristicWeight},{"expanded",s.expanded},{"candidates",s.candidates},
                            {"model_rejected",s.rejected},{"reason",s.reason},{"entry_curvature_preview_m",s.curvaturePreview}});
                        std::ofstream(directory/"parking-teaching-attempt.json")<<attempt.dump(2);
                        std::cout<<"[Parking] "<<route.reason<<" nodes="<<route.expanded<<" candidates="<<route.geometricCandidates<<" model_rejected="<<route.modelRejected<<'\n';
                        if(!route.valid){if(autoPlanning)baySearch.reject(rawNear,rawFar);continue;}
                        motion=std::make_unique<RaceGroundMotion>(calibration);motion->setMetricScale(scale);
                        motion->enableParkingNoiseTolerance();
                        RaceSnapshot reference;reference.image=frozenImage;reference.sequence=frame;reference.captured=raceTime();
                        measured=motion->update(reference,true,false);taught=true;frozen=false;
                        nlohmann::json points=nlohmann::json::array();for(auto p:route.entry)points.push_back({p.x,p.z,p.yaw});
                        std::ofstream(directory/"parking-route.json")<<nlohmann::json({{"bay",autoPlanning?nlohmann::json(nullptr):nlohmann::json(2)},{"entry",points},{"reverse_reference","visual_entry_poses_with_marked_command_fallback"},
                            {"selection_source",autoPlanning?"fresh_live_white_T_junction_geometry":"manual_current_frame"},
                            {"recognizes_bay_number",false},{"rear_boundary_m",space.rearLimit()},
                            {"parking_clearance_m",dims.clearance},{"goal_extra_margin_m",0.0},
                            {"predicted_boundary_policy","warning_continue"},{"route_preference","early_turn_into_selected_bay_far_divider_clearance"},
                            {"entry_curvature_preview_m",route.entryCurvaturePreview},{"observed_mouth_tolerance_m",RaceParkingSpace::observedMouthTolerance},
                            {"origin_m",{bayFrame.origin.x,bayFrame.origin.y}},{"forward_axis",{bayFrame.along.x,bayFrame.along.y}},{"ground_scale",scale},
                            {"parking_side",bayFrame.side<0?"left":"right"},{"canonical_frame_handedness",bayFrame.handedness()},
                            {"reverse_verified",reverseVerified},{"reverse_source","operator_report_or_saved_vehicle_observation"},
                            {"wheelbase_measured_m",dims.wheelbase},{"full_lock_experimental_degrees",dims.fullLockDegrees},
                            {"steering_model_source","field_visual_response_trial_not_wheel_angle_calibration"},
                            {"front_to_rear_axle_experimental_m",dims.frontToRearAxle},{"all_bays_assumed_empty",true}}).dump(2);
                        cv::imwrite((directory/"taught-bay.png").string(),frozenImage);
                        break;
                        }
                        if(autoPlanning&&!taught) {
                            searchLog<<nlohmann::json({{"frame",frame},{"event","PLAN_FAILED_FIND_NEXT_BAY"},
                                {"reason",teachingFeedback},{"rejection_tracking","applied_command_estimate_not_measured"}}).dump()<<'\n';
                            frozen=false;corners.clear();route={};map.setMetricScale(1);stableMouths=0;autoEnabled=true;
                            controller=RaceParkingControl(RaceParkingSpace(dims),raceConfig.parkingSpeed);test.reset();saveTest();
                            if((missionEnabled&&missionAutomatic)||searchRequested) {
                                if(missionEnabled)mission.nextParkingBay();
                                if(car->resumeMotion(car->motionEpoch())){continueLane=true;searchAdvance=true;}
                            }
                            teachingFeedback=searchAdvance?"FIND_NEXT_BAY_CONTINUE":"A_FIND_NEXT_BAY";
                        }
                        autoPlanning=false;
                    } else if(key=='a'&&!missionEnabled&&!armed&&!taught&&autoEnabled&&baySearch.hasRejected()) {
                        if(!traffic.stop&&manager.aiReady(ai,before)&&car->resumeMotion(commandEpoch)) {
                            searchAdvance=searchRequested=true;continueLane=true;teachingFeedback="FIND_NEXT_BAY_CONTINUE";
                        }
                    } else if(key=='a'&&!armed&&!continueLane) {
                        std::string blocked;
                        if(controller.failed())blocked=std::string("G_RESCAN_AFTER_")+controller.status().reason;
                        else if(controller.done())blocked="ALREADY_COMPLETED";
                        else if(!taught||!route.valid)blocked="WAIT_VALID_ROUTE";
                        else if(traffic.stop)blocked=traffic.reason;
                        else if(!measured.valid)blocked=std::string("WAIT_MOTION_")+measured.reason;
                        else if(raceTime()-captured>.18)blocked="WAIT_FRESH_CAMERA";
                        if(!blocked.empty()) {
                            test.startBlocked(blocked);saveTest();
                            teachingFeedback=blocked;std::cout<<"[Parking] Start blocked: "<<blocked<<'\n';continue;
                        }
                        if(!traffic.stop&&measured.valid&&controller.arm(route,observation,raceTime(),reverseVerified||previewOnly)) {
                            if(car->resumeMotion(commandEpoch)){armed=true;test.start();saveTest();if(missionEnabled){mission.parkingStarted();virtualBoundary=RaceVirtualBoundary(navConfig);}}
                            else {controller.abort();std::cout<<"[Parking] Stop epoch changed; reteach.\n";}
                        } else {teachingFeedback="G_RESCAN_CURRENT_POSE_OR_STALE_ROUTE";test.startBlocked(teachingFeedback);saveTest();std::cout<<"[Parking] Start blocked: current pose/route continuity; press g to rescan.\n";}
                    } else if(key=='+'||key=='=')cruise=std::min(config.maxSpeed,cruise+.05f);
                    else if(key=='-'||key=='_')cruise=std::max(config.minSpeed,cruise-.05f);
                    else if(key=='s'&&(missionEnabled||std::abs(previousSpeed)<1e-5f)) {
                        cv::imwrite((directory/("sample-"+std::to_string(frame)+".png")).string(),image);
                    }
                }
                RaceParkingCommand command;
                const auto visualObservation=observation;
                if(armed) {
                    const float appliedAngle=(float(PWMSERVOMID)-previousServo)/400*dims.fullLockDegrees*float(CV_PI)/180;
                    observation=poseFallback.update(observation,raceTime(),previousSpeed,bayFrame.handedness()*std::tan(appliedAngle)/dims.wheelbase,dims);
                    command=controller.update(observation,raceTime(),dt,!traffic.stop,std::min(taskCruise,traffic.speedCap));
                }
                else if(controller.failed())command=controller.status();
                else command.reason=frozen?(corners.size()==2?"P_PLAN_ROUTE":"CLICK_NEAR_THEN_FAR_WHITE_LINE_TIPS"):taught?"A_START_PARKING":"F_FREEZE_AT_BAY_2";
                if(!armed&&!continueLane&&!controller.failed()&&!teachingFeedback.empty())command.reason=teachingFeedback.c_str();
                uint16_t servo=PWMSERVOMID;float speed=0;
                std::string actuatorReason="STOPPED";uint16_t requestedServo=PWMSERVOMID;
                if(armed&&!controller.failed()) {
                    const bool hold=std::string(command.state)=="PARK_HOLD";
                    // An invalid motion frame stops translation; it must not
                    // also centre the wheels and force a fresh steering wait.
                    const auto requested=hold?previousServo:raceParkingServo(bayFrame.handedness()*command.curvature,dims);
                    requestedServo=requested;
                    servo=uint16_t(std::lround(std::clamp(float(requested),float(previousServo)-600*std::min(dt,.1f),float(previousServo)+600*std::min(dt,.1f))));
                    speed=command.speed;
                    actuatorReason=command.speed==0?"CONTROLLER_STOP":std::abs(int(servo)-int(requested))>35?"STEERING_SLEW_MOVING":"TRACKING";
                    // A command calculated from a frame that became old during processing cannot move.
                    if(raceTime()-captured>.18||raceTime()-ai.captured>raceConfig.maxDetectionAge)speed=0;
                    if(controller.done()) {
                        if(completedAt<0)completedAt=raceTime();
                        // Rejoining uses the original lane quality/curve limits and all traffic stops.
                        if(raceTime()-completedAt>=.5){continueLane=true;armed=false;pilot.resetControl();if(missionEnabled)mission.parkingFinished();}
                    }
                }
                if(continueLane||(missionEnabled&&!mission.parkingSearch()&&!mission.parkingRun())) {
                    pilot.update(traffic.selectedBranch?traffic.path.left:track.pointsEdgeLeft,
                        traffic.selectedBranch?traffic.path.right:track.pointsEdgeRight,dt,previousSpeed);
                    command={0,0,missionEnabled?mission.name():"PARK_REJOIN_WAIT","WAIT_TRUSTED_LANE",0};
                    if(raceDirectionSeeking(traffic)&&raceTime()-captured<=.18) {
                        speed=raceDirectionSpeed(traffic,pilot,taskCruise,dt,previousSpeed);
                        servo=raceDirectionServo(traffic,PWMSERVOMID);
                        command.state=traffic.state;command.reason=traffic.reason;
                    }
                    else if(!traffic.stop&&pilot.ready&&!pilot.lost&&raceTime()-captured<=.18) {
                        speed=pilot.speedFor(std::min(taskCruise,traffic.speedCap),dt);servo=raceDirectionServo(traffic,pilot.servo);
                        command.state=missionEnabled&&!controller.done()?mission.name():"PARK_REJOIN_LANE";
                        command.reason=missionEnabled?"MISSION_LANE_FOLLOWING":"PARKED_THREE_SECONDS_AND_EXITED_CONTINUE_FORWARD";
                    }
                    else if((missionEnabled||searchAdvance||continueLane)&&!traffic.stop&&raceTime()-captured<=.18) {
                        // Missing paint is recoverable in competition. Keep the
                        // current turn until current edges can take over again.
                        speed=pilot.speedForDirection(std::min(taskCruise,traffic.speedCap),dt,previousSpeed);
                        servo=pilot.valid?pilot.servo:previousServo;
                        command.state=missionEnabled?mission.name():"PARK_FIND_NEXT_BAY";command.reason="LANE_GEOMETRY_RECOVERY_CONTINUE";
                    }
                    if(searchAdvance){speed=std::min(speed,taskCruise);command.state="PARK_FIND_NEXT_BAY";command.reason="REJECTED_BAY_SKIPPED_SEARCHING";}
                }
                if(missionEnabled&&!mission.parkingSearch()&&!mission.parkingRun()) {
                    if(virtualDecision.involved) {
                        std::vector<PointX> left,right;for(auto p:virtualPlan.left)left.emplace_back(p.x,int(std::lround(p.y)));
                        for(auto p:virtualPlan.right)right.emplace_back(p.x,int(std::lround(p.y)));
                        virtualPilot.update(left,right,dt,previousSpeed);
                        if(virtualDecision.follow&&virtualPilot.ready&&!traffic.stop) {
                            speed=virtualPilot.speedFor(std::min(taskCruise,traffic.speedCap),dt);
                            servo=virtualFeedback.update(virtualPlan,virtualPilot.error,virtualPilot.servo,previousServo,dt,previousSpeed);command.state=virtualDecision.state;command.reason=virtualDecision.reason;
                        } else {speed=0;servo=previousServo;command.reason=virtualDecision.reason;}
                    }
                    const int required=mission.routeDirection()<0?RoadLeft:mission.routeDirection()==2?RoadStraight:0;
                    if(required&&groundArrow.allowed&&!(groundArrow.allowed&required)&&traffic.geometrySplitRows>=raceConfig.minBranchRows) {
                        command.reason="GROUND_ARROW_ROUTE_CONFLICT_CONTINUE_SIGN_ROUTE";
                    }
                }
                if(missionEnabled) {
                    if(controller.failed())mission.fault(controller.status().reason);
                    if(mission.failed()||mission.complete()||!missionAutomatic) {speed=0;if(mission.complete()||mission.failed())car->pauseMotion();}
                }
                if(traffic.stop) {speed=0;command.reason=traffic.reason;}
                if(raceTime()-captured>.18||!manager.aiReady(ai,raceTime())) {
                    speed=0;if(armed||continueLane)command.reason="PARK_FINAL_FRAME_OR_AI_STALE";
                }
                if(controller.failed()){speed=0;servo=PWMSERVOMID;armed=false;car->pauseMotion();}
                if(stop.load())speed=0;
                if(car->hasFault())throw std::runtime_error("Parking control command timed out");
                const float previousCommandSpeed=previousSpeed;
                if(!virtualDecision.follow||speed==0)virtualFeedback.reset();
                car->carControl(speed,servo);previousServo=servo;previousSpeed=speed;
                test.observe(command.state,command.reason,speed,actuatorReason,!observation.estimated,controller.terminalRecoveryUsed());
                if(!armed)poseFallback.reset();
                if(missionLog) {
                    auto row=mission.json();row["frame"]=frame;row["no_control"]=previewOnly;row["speed_command"]=speed;row["pwm"]=servo;
                    row["virtual_correction_pwm"]=virtualFeedback.correction();row["virtual_error_px"]=virtualPilot.error;
                    row["traffic_state"]=traffic.state;row["traffic_reason"]=traffic.reason;row["parking_state"]=command.state;
                    row["guide_direction"]=traffic.guideDirection;row["guide_pwm"]=traffic.guidePwm;row["lane_blend"]=traffic.laneBlend;
                    row["cruise_setting"]=cruise;row["task_cruise_limit"]=taskCruise;
                    row["gate_waiting"]=traffic.gateWaiting;row["gate_opened"]=traffic.gateOpened;
                    row["decision_reason"]=command.reason;row["arrow_allowed"]=groundArrow.allowed;row["arrow_box"]={groundArrow.box.x,groundArrow.box.y,groundArrow.box.width,groundArrow.box.height};
                    row["bypass"]=raceObstacleJson(virtualPlan,navConfig);missionLog<<row.dump()<<'\n';
                }
                const double now=raceTime();
                if(controller.active()&&!measured.valid&&firstMotionFailure.empty())firstMotionFailure=measured.reason;
                const auto failure=controller.status();
                nlohmann::json row={{"frame",frame},{"time",now},{"state",command.state},{"reason",command.reason},
                    {"captured",captured},{"controller_reason",controller.failed()?failure.reason:command.reason},{"first_motion_failure",firstMotionFailure},
                    {"armed",armed},{"frozen",frozen},{"command_speed",speed},{"pwm",servo},{"segment",command.segment},
                    {"parking_side",bayFrame.side<0?"left":"right"},{"task_cruise_limit",taskCruise},
                    {"requested_pwm",requestedServo},{"requested_speed",command.speed},{"actuator_reason",actuatorReason},
                    {"predicted_boundary_warning",command.predictedBoundaryWarning},
                    {"entry_heading_recovery",command.entryHeadingRecovery},{"heading_error_rad",command.headingError},
                    {"entry_mouth_recovery",command.entryMouthRecovery},
                    {"exit_heading_recovery",command.exitHeadingRecovery},
                    {"inside_observed_envelope",space.freeObserved(observation.pose)},
                    {"inside_mouth_transition",space.freeEntryMouth(observation.pose)},
                    {"base_curvature",command.baseCurvature},{"requested_curvature",command.requestedCurvature},
                    {"physical_curvature",bayFrame.handedness()*command.curvature},{"pose_frame","canonical_left_bay"},
                    {"pose",{observation.pose.x,observation.pose.z,observation.pose.yaw}},{"motion_valid",measured.valid},
                    {"pose_estimated",observation.estimated},
                    {"terminal_recovery",controller.terminalRecoveryUsed()},
                    {"pose_source",observation.estimated?"APPLIED_COMMAND_ESTIMATE":"VISUAL_ODOMETRY"},
                    {"visual_pose",{visualObservation.pose.x,visualObservation.pose.z,visualObservation.pose.yaw}},
                    {"motion_epoch",measured.epoch},{"motion_reason",measured.reason},{"motion_inliers",measured.inliers},
                    {"motion_residual_m",measured.residual},{"motion_residual_limit_m",.008},{"motion_fit_scale",measured.fitScale},
                    {"motion_fit_translation_m",measured.fitTranslation},{"motion_fit_angle_rad",measured.fitAngle},
                    {"motion_reference_age_s",measured.referenceAge},{"motion_features",measured.features},
                    {"motion_matches",measured.matches},{"motion_span_x_m",measured.spanX},{"motion_span_z_m",measured.spanZ},
                    {"motion_backup_used",measured.backupReferenceUsed},{"motion_backup_attempts",measured.backupAttempts},
                    {"fully_inside_bay",taught&&measured.valid&&space.parked(visualObservation.pose)},
                    {"control_pose_inside_bay",taught&&space.parked(observation.pose)},
                    {"rear_boundary_m",taught?nlohmann::json(space.rearLimit()):nlohmann::json(nullptr)},
                    {"observed_rear_tolerance_m",RaceParkingSpace::observedRearTolerance},{"traffic_state",traffic.state},
                    {"traffic_reason",traffic.reason},{"ai_sequence",ai.sequence},{"ai_age_s",now-ai.captured}};
                telemetry<<row.dump()<<'\n';
                if(!telemetry)throw std::runtime_error("Parking telemetry disk write failed");
                recent.push_back({image.clone(),row});if(recent.size()>12)recent.pop_front();
                const bool firstStop=previousCommandSpeed!=0&&speed==0&&
                    (std::string(command.state)=="PARK_HOLD"||controller.failed());
                if(firstStop&&evidenceCount<3&&(!evidenceWriter.valid()||
                   evidenceWriter.wait_for(std::chrono::seconds(0))==std::future_status::ready)) {
                    if(evidenceWriter.valid())evidenceWriter.get();
                    auto frames=recent;const auto destination=directory/("parking-stop-evidence-"+std::to_string(++evidenceCount));
                    // Encode only in a bounded background task, after the zero-speed
                    // command. Evidence I/O must not create a new visual-frame gap.
                    evidenceWriter=std::async(std::launch::async,[frames=std::move(frames),destination]{
                        try {
                            std::filesystem::create_directories(destination);nlohmann::json saved=nlohmann::json::array();
                            for(const auto &item:frames) {
                                const auto name="frame-"+std::to_string(item.metadata.at("frame").template get<uint64_t>())+".png";
                                if(!cv::imwrite((destination/name).string(),item.image))throw std::runtime_error("PNG evidence write failed");
                                auto metadata=item.metadata;metadata["image"]=name;saved.push_back(std::move(metadata));
                            }
                            std::ofstream(destination/"frames.json")<<saved.dump(2);
                        }catch(const std::exception &e){std::cerr<<"[Parking evidence] "<<e.what()<<'\n';}
                    });
                }
                if(previousState!=command.state) {
                    saveTest();std::cout<<"[Parking test] "<<test.progress()<<'\n';
                    std::cout<<"[Parking] "<<command.state<<" | "<<command.reason<<'\n';previousState=command.state;telemetry.flush();
                    if(std::string(command.state)=="PARK_EXIT"&&speed==0) {
                        nlohmann::json poses=nlohmann::json::array();for(auto p:controller.reverseReference())poses.push_back({p.x,p.z,p.yaw});
                        std::ofstream(directory/"parking-reverse-route.json")<<nlohmann::json({{"source",controller.usedPoseEstimate()?"spatially_resampled_visual_and_estimated_entry_poses":"spatially_resampled_measured_entry_poses"},
                            {"command_pose_estimate_used",controller.usedPoseEstimate()},
                            {"poses",poses},{"motion_epoch",measured.epoch},{"rear_boundary_m",space.rearLimit()}}).dump(2);
                    }
                    if(controller.done()||controller.failed()) {
                        std::ofstream(directory/"parking-result.json")<<row.dump(2);
                        if(speed==0)cv::imwrite((directory/"parking-result-camera.png").string(),image);
                    }
                }
                if(display.frameDue()) {
                cv::Mat shown;cv::resize(frozen?frozenImage:image,shown,{640,480});
                if(missionEnabled) {
                    std::string options=groundArrow.allowed?"":"?";
                    if(groundArrow.allowed&RoadLeft)options+="L ";if(groundArrow.allowed&RoadStraight)options+="S ";if(groundArrow.allowed&RoadRight)options+="R ";
                    cv::putText(shown,"GROUND "+options,{6,22},cv::FONT_HERSHEY_PLAIN,1,{0,255,255});
                    if(groundArrow.allowed)cv::rectangle(shown,{groundArrow.box.x,groundArrow.box.y,groundArrow.box.width,groundArrow.box.height},{255,255,0},1);
                    for(size_t i=1;i<virtualPlan.centers.size();++i)cv::line(shown,
                        {int(virtualPlan.centers[i-1].y),virtualPlan.centers[i-1].x},
                        {int(virtualPlan.centers[i].y),virtualPlan.centers[i].x},{255,255,0},2);
                }
                for(size_t i=0;i<corners.size();++i)if(frozen) {
                    cv::circle(shown,corners[i],6,{0,0,255},-1);
                    cv::putText(shown,i==0?"1 NEAR":"2 FAR",corners[i]+cv::Point2f(8,-8),cv::FONT_HERSHEY_PLAIN,1.2,{0,0,255});
                }
                if(taught) {
                    auto displayPose=measured.pose;
                    if(observation.estimated) {
                        const auto pose=bayFrame.physical(observation.pose,dims.aisle);
                        displayPose={pose.x,pose.z,pose.yaw};
                    }
                    auto project=[&](cv::Point2f p){return map.toImage(displayPose.local(world(p)));};
                    auto line=[&](cv::Point2f a,cv::Point2f b,cv::Scalar color){
                        a=project(a);b=project(b);
                        if(std::isfinite(a.x)&&std::isfinite(a.y)&&std::isfinite(b.x)&&std::isfinite(b.y)&&
                           std::abs(a.x)<5000&&std::abs(a.y)<5000&&std::abs(b.x)<5000&&std::abs(b.y)<5000)cv::line(shown,a,b,color,2);
                    };
                    const auto bay=space.bay();for(size_t i=0;i<bay.size();++i)line(bay[i],bay[(i+1)%bay.size()],{0,255,255});
                    for(size_t i=1;i<route.entry.size();++i)line({route.entry[i-1].x,route.entry[i-1].z},{route.entry[i].x,route.entry[i].z},{0,255,0});
                }
                cv::copyMakeBorder(shown,shown,0,80,0,0,cv::BORDER_CONSTANT,{0,0,0});
                cv::putText(shown,std::string(command.state)+" "+command.reason,{5,500},cv::FONT_HERSHEY_PLAIN,1,{0,255,255});
                cv::putText(shown,missionEnabled?std::string(mission.name())+(previewOnly?" NO MOTOR":""):frozen&&!teachingDetails.empty()?teachingDetails:test.progress(),{5,522},cv::FONT_HERSHEY_PLAIN,1,{0,255,255});
                cv::putText(shown,"SPACE stop | g reset | n lap/finish | s sample | q quit",{5,546},cv::FONT_HERSHEY_PLAIN,1,{0,255,255});
                display.publish(shown,{});
                }
            }
            car->pauseMotion();
            test.finish("OPERATOR_EXIT");saveTest();
        }catch(const std::exception &e) {
            try{car->latchStop();}catch(...){}display.failed=true;
            test.finish(std::string("EXCEPTION: ")+e.what());try{saveTest();}catch(...){}
            std::ofstream(directory/"parking-error.txt")<<e.what()<<'\n';std::cerr<<"[Parking] "<<e.what()<<'\n';
        }
        display.finished=true;
    });
    try {
        while(!stop.load()&&!display.finished.load()) {
            cv::Mat image,binary;if(display.snapshot(image,binary))cv::imshow(window,image);
            const int key=raceUiKey(cv::waitKey(20));display.progress();
            if(raceUiQuit(key)){stop=true;car->pauseMotion();break;}
            if(key==' ')car->pauseMotion();
            display.post(key);
        }
    }catch(...){stop=true;car->latchStop();driving.join();cv::destroyWindow(window);throw;}
    stop=true;car->pauseMotion();driving.join();cv::destroyWindow(window);
    return display.failed?1:0;
}
