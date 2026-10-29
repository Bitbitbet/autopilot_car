#include "race_parking_control.hpp"
#include "race_parking_detection.hpp"
#include "race_parking_test.hpp"
#include "race_ground_motion.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>
static void require(bool value,const char *why){if(!value)throw std::runtime_error(why);}
int main(int argc,char **argv) {
    try {
        if(argc==3&&std::string(argv[1])=="--exit-field") {
            const auto folder=std::filesystem::path(argv[2]);nlohmann::json saved;
            std::ifstream(folder/"parking-route.json")>>saved;
            RaceParkingRoute route;route.valid=true;route.entryCurvaturePreview=saved.at("entry_curvature_preview_m");
            for(auto p:saved.at("entry"))route.entry.push_back({p[0].get<float>(),p[1].get<float>(),p[2].get<float>()});
            route.exit=route.entry;std::reverse(route.exit.begin(),route.exit.end());
            RaceParkingSpace space;space.keepRearWithinStart(route.entry.front());RaceParkingControl control(space);
            std::ifstream input(folder/"parking-control.jsonl");std::string line;bool armed=false,exit=false;double previous=0;
            RaceParkingObservation o;RaceParkingCommand command;float wheel=0;
            while(std::getline(input,line)) {
                const auto row=nlohmann::json::parse(line);const int frame=row.at("frame");
                if(!row.at("armed").get<bool>())continue;
                const auto p=row.at("pose");o={{p[0].get<float>(),p[1].get<float>(),p[2].get<float>()},
                    row.at("captured").get<double>(),uint64_t(frame),row.at("motion_epoch").get<int>(),row.at("motion_valid").get<bool>()};
                const double now=row.at("time");
                if(!armed){require(control.arm(route,o,now,true),"Cannot arm recorded exit test");armed=true;previous=now;continue;}
                command=control.update(o,now,float(now-previous),true);previous=now;
                require(!control.failed(),command.reason);
                if(std::string(command.reason)=="DWELL_COMPLETE_REVERSE_NEXT_FRAME") {
                    exit=true;wheel=(1500-row.at("pwm").get<float>())/400*33*float(CV_PI)/180;
                    std::cout<<"recorded entry+dwell reproduced; exit frame="<<frame<<" measured points="<<control.reverseReference().size()<<'\n';break;
                }
            }
            require(exit,"Actual entry and dwell did not reach exit");
            auto recordedControl=control;double recordedTime=previous;bool corrected=false;
            while(std::getline(input,line)) {
                const auto row=nlohmann::json::parse(line);const int frame=row.at("frame");if(frame>350)break;
                const auto p=row.at("pose");RaceParkingObservation sample{{p[0].get<float>(),p[1].get<float>(),p[2].get<float>()},
                    row.at("captured").get<double>(),uint64_t(frame),row.at("motion_epoch").get<int>(),row.at("motion_valid").get<bool>()};
                const double now=row.at("time");const auto c=recordedControl.update(sample,now,float(now-recordedTime),true);recordedTime=now;
                require(!recordedControl.failed(),c.reason);
                if(frame==350) {
                    require(c.exitHeadingRecovery&&c.curvature>0,"Reverse still counter-steers after heading overshoot");
                    std::cout<<"actual frame350 recorded PWM="<<row.at("pwm")<<" corrected PWM="
                        <<1500-400*std::atan(.2f*c.curvature)/(33*float(CV_PI)/180)<<" heading_error="<<c.headingError<<'\n';
                    corrected=true;
                }
            }
            require(corrected,"Missing actual reverse counter-steer frame");
            nlohmann::json recorded;std::ifstream(folder/"parking-reverse-route.json")>>recorded;
            const auto &resampled=control.reverseReference();
            const auto first=recorded.at("poses").front(),last=recorded.at("poses").back();
            require(resampled.size()>20&&resampled.size()<recorded.at("poses").size(),"Actual dense reverse path not spatially resampled");
            require(std::hypot(resampled.front().x-first[0].get<float>(),resampled.front().z-first[1].get<float>())<1e-5f&&
                std::hypot(resampled.back().x-last[0].get<float>(),resampled.back().z-last[1].get<float>())<1e-5f,"Resampling changed measured endpoints");
            double time=previous;int reverseFrames=0;float worst=0;std::string phase;
            for(int n=0;n<3500&&!control.done()&&!control.failed();++n) {
                const float target=std::atan(.2f*command.curvature),rate=33*float(CV_PI)/180*600/400;
                wheel+=std::clamp(target-wheel,-rate*.08f,rate*.08f);
                o.pose=raceParkingAdvance(o.pose,command.speed*.08f,std::tan(wheel)/.2f,space.dimensions());
                require(space.freeEntryMouth(o.pose),"Reverse model leaves maneuver area");
                reverseFrames+=command.speed<0;worst=std::max(worst,std::abs(command.headingError));
                time+=.08;o.captured=time;++o.sequence;command=control.update(o,time,.08f,true);
                if(phase!=command.state){std::cout<<command.state<<' '<<command.reason<<" pose="<<o.pose.x<<','<<o.pose.z<<','<<o.pose.yaw<<'\n';phase=command.state;}
            }
            require(control.done()&&reverseFrames>20,"Measured reverse route cannot return to aisle");
            std::cout<<"PASS actual 154-point entry/dwell replay then reverse plant model; max heading error="<<worst
                <<" reverse_frames="<<reverseFrames<<"; physical exit validation pending\n";return 0;
        }
        if(argc==3&&std::string(argv[1])=="--mouth-recovery") {
            const auto folder=std::filesystem::path(argv[2]);nlohmann::json saved;
            std::ifstream(folder/"parking-route.json")>>saved;
            RaceParkingRoute route;route.valid=true;route.entryCurvaturePreview=saved.at("entry_curvature_preview_m");
            for(auto p:saved.at("entry"))route.entry.push_back({p[0].get<float>(),p[1].get<float>(),p[2].get<float>()});
            route.exit=route.entry;std::reverse(route.exit.begin(),route.exit.end());
            RaceParkingSpace space;space.keepRearWithinStart(route.entry.front());RaceParkingControl control(space);
            std::ifstream input(folder/"parking-control.jsonl");std::string line;bool armed=false;double previous=0;
            RaceParkingObservation o;RaceParkingCommand command;float wheel=0;bool sawFaultFrame=false;
            while(std::getline(input,line)) {
                const auto row=nlohmann::json::parse(line);const int frame=row.at("frame");
                if(frame>187)break;
                if(!row.at("armed").get<bool>()&&frame!=187)continue;
                const auto p=row.at("pose");o={{p[0].get<float>(),p[1].get<float>(),p[2].get<float>()},
                    row.at("captured").get<double>(),uint64_t(frame),row.at("motion_epoch").get<int>(),row.at("motion_valid").get<bool>()};
                const double now=row.at("time");
                if(!armed){require(control.arm(route,o,now,true),"Cannot arm actual mouth replay");armed=true;previous=now;continue;}
                command=control.update(o,now,float(now-previous),true);previous=now;
                require(!control.failed(),command.reason);
                if(frame==186)wheel=(1500-row.at("pwm").get<float>())/400*33*float(CV_PI)/180;
                if(frame==187) {
                    require(!space.freeObserved(o.pose)&&space.freeEntryMouth(o.pose),"Recorded 10.8mm mouth fault not exercised");
                    require(command.speed>0&&command.entryMouthRecovery&&command.requestedCurvature>3.f,
                        "Actual fault still stops instead of verified stronger left turn");
                    sawFaultFrame=true;std::cout<<"actual frame 187: "<<command.reason<<" speed="<<command.speed<<" target_curvature="<<command.requestedCurvature<<'\n';
                }
            }
            require(sawFaultFrame,"Missing actual stop frame");
            bool returned=false,reversed=false;double time=previous;std::string phase;
            for(int n=0;n<3500&&!control.done()&&!control.failed();++n) {
                const float target=std::atan(.2f*command.curvature),rate=33*float(CV_PI)/180*600/400;
                wheel+=std::clamp(target-wheel,-rate*.08f,rate*.08f);
                o.pose=raceParkingAdvance(o.pose,command.speed*.08f,std::tan(wheel)/.2f,space.dimensions());
                require(space.freeEntryMouth(o.pose),"Recovery model left the bounded paint transition");
                returned|=space.free(o.pose);reversed|=command.speed<0;
                time+=.08;o.captured=time;++o.sequence;command=control.update(o,time,.08f,true);
                if(phase!=command.state){std::cout<<command.state<<' '<<command.reason<<" pose="<<o.pose.x<<','<<o.pose.z<<','<<o.pose.yaw<<'\n';phase=command.state;}
            }
            require(control.done()&&returned&&reversed,"Continuation from actual stop fails entry/dwell/measured reverse model");
            require(!space.freeEntryMouth({.25f,.1f,0})&&!space.freeEntryMouth({-.7f,.2f,1.2f}),"Road/deep outer edge expanded");
            auto behind=route.entry.front();behind.z-=.01f;
            require(!space.freeEntryMouth(behind),"Rear boundary expanded");
            std::cout<<"PASS recorded stop now requests recovery; bounded continuation model completes entry/3s/reverse; outer/rear bounds retained; physical validation pending\n";
            return 0;
        }
        if(argc==3&&std::string(argv[1])=="--heading-field") {
            const auto folder=std::filesystem::path(argv[2]);nlohmann::json saved;
            std::ifstream(folder/"parking-route.json")>>saved;
            RaceParkingRoute route;route.valid=true;route.entryCurvaturePreview=saved.at("entry_curvature_preview_m");
            for(auto p:saved.at("entry"))route.entry.push_back({p[0].get<float>(),p[1].get<float>(),p[2].get<float>()});
            route.exit=route.entry;std::reverse(route.exit.begin(),route.exit.end());
            RaceParkingSpace space;space.keepRearWithinStart(route.entry.front());RaceParkingControl control(space);
            std::ifstream input(folder/"parking-control.jsonl");std::string line;bool armed=false;double previous=0;int verified=0;
            while(std::getline(input,line)) {
                const auto row=nlohmann::json::parse(line);const int frame=row.at("frame");
                if(!row.at("armed").get<bool>()||frame>164)continue;
                const auto p=row.at("pose");RaceParkingObservation o{{p[0].get<float>(),p[1].get<float>(),p[2].get<float>()},
                    row.at("captured").get<double>(),uint64_t(frame),row.at("motion_epoch").get<int>(),row.at("motion_valid").get<bool>()};
                const double now=row.at("time");
                if(!armed){require(control.arm(route,o,now,true),"Cannot arm field observation replay");armed=true;previous=now;continue;}
                const auto c=control.update(o,now,float(now-previous),true);previous=now;
                require(!control.failed(),c.reason);
                if(frame==105||frame==133||frame==164) {
                    const float pwm=1500-400*std::atan(.2f*c.curvature)/(33.f*float(CV_PI)/180);
                    std::cout<<"frame="<<frame<<" recorded_pwm="<<row.at("pwm")<<" new_pwm="<<pwm
                        <<" heading_error="<<c.headingError<<" base="<<c.baseCurvature<<" requested="<<c.requestedCurvature<<'\n';
                    require(c.entryHeadingRecovery&&c.curvature>0,"Lagging entry still reverses or cancels steering");
                    if(frame==164)require(c.requestedCurvature>2.3f,"Final lagging pose retains insufficient correction");
                    ++verified;
                }
            }
            require(verified==3,"Missing actual problem frames");
            std::cout<<"PASS recorded-pose command replay at frames 105/133/164; not a counterfactual vehicle trajectory\n";return 0;
        }
        if(argc==3&&std::string(argv[1])=="--motion-field") {
            const auto folder=std::filesystem::path(argv[2]);
            RaceGroundCalibration calibration;calibration.load((folder/"ground-calibration.json").string(),false);
            nlohmann::json route,frames;
            std::ifstream(folder/"parking-route.json")>>route;
            const auto evidence=folder/"parking-stop-evidence-1";
            std::ifstream(evidence/"frames.json")>>frames;
            RaceGroundMotion original(calibration),repaired(calibration);
            original.setMetricScale(route.at("ground_scale"));repaired.setMetricScale(route.at("ground_scale"));
            repaired.enableParkingNoiseTolerance();
            int recordedFailures=0,valid=0;RaceGroundMotionResult measured;
            std::cout<<"Replay uses saved camera images without unsaved historical AI exclusion boxes; baseline may differ from live fit.\n";
            for(const auto &row:frames) {
                RaceSnapshot snapshot;snapshot.image=cv::imread((evidence/row.at("image").get<std::string>()).string());
                snapshot.sequence=row.at("frame");snapshot.captured=row.at("captured");
                const auto baseline=original.update(snapshot,true,false);measured=repaired.update(snapshot,true,false);
                if(&row==&frames.front())continue;
                if(row.at("motion_reason")=="RIGID_RESIDUAL_HIGH") {
                    const float residual=row.at("motion_residual_m");
                    require(residual>.005f&&residual<=.008f,"Recorded residual not covered by parking noise budget");
                    ++recordedFailures;
                }
                valid+=measured.valid;
                std::cout<<snapshot.sequence<<" old="<<baseline.reason<<" new="<<measured.reason<<" residual="<<measured.residual
                    <<" scale="<<measured.fitScale<<" pose="<<measured.pose.x<<','<<measured.pose.z<<','<<measured.pose.angle<<'\n';
                require(measured.valid,"Saved parking frame still loses visual motion");
                require(measured.epoch==1,"Field recovery changed coordinate epoch");
            }
            require(recordedFailures>0&&valid>=10,"No recorded residual fault or insufficient valid image replay");
            // Reacquisition must deliver the whole visual displacement over a
            // brief hold, not treat a stopped command as zero physical motion.
            RaceParkingSpace space;const RaceParkingPose start{0,-.4f,0};
            const auto plan=racePlanTrackableParking(space,start);require(plan.valid,"Recovery test needs feasible route");
            RaceParkingControl control(space);RaceParkingObservation o{start,1,1,1,true};
            require(control.arm(plan,o,1,true),"Recovery test arm failed");
            o.sequence=2;o.captured=1.08;o.valid=false;
            require(control.update(o,1.08,.08f,true).speed==0,"Invalid frame did not hold motion");
            for(int i=3;i<=4;++i){o.sequence=i;o.captured+=.08;control.update(o,o.captured,.08f,true);}
            o.sequence=5;o.captured=1.32;o.valid=true;o.pose.z+=.012f;
            const auto resumed=control.update(o,1.32,.08f,true);
            require(!control.failed()&&resumed.speed>0,"Valid 320ms visual bridge cannot resume");
            // Opt-in must not accept zoom changes or grossly nonrigid motion.
            std::vector<cv::Point2f> a,b;
            for(int y=0;y<5;++y)for(int x=0;x<5;++x){a.push_back({x*.3f,y*.2f});b.push_back(a.back()*1.01247f);}
            RaceGroundPose pose;int inliers;float residual;
            require(!RaceGroundMotion::fit(a,b,pose,inliers,residual)&&residual>.005f,
                "Small scene-scale disturbance did not exercise old residual failure");
            require(RaceGroundMotion::fit(a,b,pose,inliers,residual,nullptr,.08f,.008f),
                "Parking noise policy still rejects bounded 1.247% scene-scale disturbance");
            for(size_t i=0;i<a.size();++i)b[i]=a[i]*1.2f;
            pose={};
            require(!RaceGroundMotion::fit(a,b,pose,inliers,residual,nullptr,.08f,.008f),"Parking tolerance accepted 20% scale change");
            std::cout<<"PASS 11 saved moving-image pairs; recorded residual covered; synthetic mild disturbance old-fail/new-pass; 320ms hold recovery; gross scale rejection retained\n";
            return 0;
        }
        if(argc==2&&std::string(argv[1])=="--test-report") {
            RaceParkingTestReport report;
            report.planned(true,"MODEL_CHECKED_FORWARD_ROUTE");
            report.observe("PARK_COMPLETE","synthetic before start",0,"STOPPED");
            auto row=report.toJson().at("retained_attempts").back();
            require(row.at("route_generated").get<bool>()&&!row.at("returned_to_aisle_observed").get<bool>(),
                "Generated route or an unarmed state falsely reports physical parking success");
            report.start();report.observe("PARK_ENTRY","VISUAL_FORWARD_ROUTE",.1f,"TRACKING");
            report.observe("PARK_ENTRY","VISUAL_FORWARD_ROUTE",0,"STEERING_SLEW_WAIT");
            report.observe("PARK_FAULT","PARK_PREDICTED_BODY_CROSSES_LINE",0,"STOPPED");
            report.finish("OPERATOR_EXIT");report.reset();
            report.planned(true,"MODEL_CHECKED_FORWARD_ROUTE");report.start();
            report.observe("PARK_DWELL","FULL_BODY_INSIDE_STOPPING",0,"STOPPED");
            row=report.toJson().at("retained_attempts").back();
            require(row.at("full_body_inside_observed").get<bool>()&&!row.at("three_second_dwell_confirmed").get<bool>(),
                "Entering dwell falsely reports three seconds completed");
            report.observe("PARK_EXIT","DWELL_COMPLETE_REVERSE_NEXT_FRAME",0,"STOPPED");
            report.observe("PARK_EXIT","VISUAL_REVERSE_ROUTE",-.08f,"TRACKING");
            report.observe("PARK_COMPLETE","RETURNED_TO_AISLE",0,"STOPPED");report.finish("OPERATOR_EXIT");
            const auto rows=report.toJson().at("retained_attempts");
            require(rows.size()==2&&rows[0].at("outcome")=="FAULT"&&rows[0].at("reason")=="PARK_PREDICTED_BODY_CROSSES_LINE",
                "Reset/exit loses previous attempt fault");
            require(rows[0].at("steering_wait_frames")==1&&!rows[0].at("full_body_inside_observed").get<bool>(),
                "Interrupted entry marked as parked");
            require(rows[1].at("outcome")=="COMPLETE"&&rows[1].at("three_second_dwell_confirmed").get<bool>()&&
                rows[1].at("reverse_motion_commanded").get<bool>()&&!rows[1].at("lane_motion_resumed").get<bool>(),
                "Stage summary fails to distinguish full parking from lane rejoin");
            report.reset();report.planned(false,"NO_MODEL_TRACKABLE_FORWARD_ROUTE");report.finish("OPERATOR_EXIT");
            require(report.toJson().at("retained_attempts").back().at("reason")=="NO_MODEL_TRACKABLE_FORWARD_ROUTE",
                "Closing without starting destroys the planning failure");
            std::cout<<"PASS parking test report: no success from planning alone, dwell timing stages, separate retry/fault and blocked-start evidence\n";return 0;
        }
        if(argc==4&&(std::string(argv[1])=="--field-route"||std::string(argv[1])=="--response-route")) {
            const bool responseCheck=std::string(argv[1])=="--response-route";
            const auto folder=std::filesystem::path(argv[3]);nlohmann::json saved;
            RaceParkingPose start;
            if(std::filesystem::exists(folder/"parking-route.json")) {
                std::ifstream(folder/"parking-route.json")>>saved;
                const auto p=saved.at("entry").front();start={p[0].get<float>(),p[1].get<float>(),p[2].get<float>()};
            } else {
                std::ifstream(folder/"parking-teaching-attempt.json")>>saved;
                const auto a=saved.at("scaled_near_m"),b=saved.at("scaled_far_m");
                const cv::Point2f origin(a[0].get<float>(),a[1].get<float>()),far(b[0].get<float>(),b[1].get<float>());
                const auto along=(far-origin)*(1.f/.41f);const cv::Point2f right(along.y,-along.x);const auto offset=-origin;
                start={offset.dot(right)-.225f,offset.dot(along),std::atan2(along.x,along.y)};
            }
            RaceParkingSpace space;space.keepRearWithinStart(start);
            if(responseCheck) {
                RaceParkingDimensions assumed;assumed.fullLockDegrees=45;
                RaceParkingSpace previousSpace(assumed);previousSpace.keepRearWithinStart(start);
                RaceParkingRoute previousRoute;previousRoute.valid=true;
                previousRoute.entryCurvaturePreview=saved.value("entry_curvature_preview_m",0.f);
                for(const auto &p:saved.at("entry"))previousRoute.entry.push_back({p[0].get<float>(),p[1].get<float>(),p[2].get<float>()});
                previousRoute.exit=previousRoute.entry;std::reverse(previousRoute.exit.begin(),previousRoute.exit.end());
                RaceParkingControl previousControl(previousSpace);RaceParkingObservation sample{start,1,1,1,true};
                require(previousControl.arm(previousRoute,sample,1,true),"Cannot arm old response model");
                double time=1;float physicalWheel=0;
                for(int n=0;n<3500&&!previousControl.done()&&!previousControl.failed();++n) {
                    time+=.08;sample.captured=time;++sample.sequence;
                    const auto command=previousControl.update(sample,time,.08f,true);
                    // Same PWM has 33deg physical response, while old mapping
                    // assumes 45deg. This plant is independent of its controller.
                    const float target=std::atan(assumed.wheelbase*command.curvature)*33.f/45.f;
                    const float rate=33.f*float(CV_PI)/180*600/400;
                    physicalWheel+=std::clamp(target-physicalWheel,-rate*.08f,rate*.08f);
                    sample.pose=raceParkingAdvance(sample.pose,command.speed*.08f,std::tan(physicalWheel)/assumed.wheelbase,assumed);
                }
                std::cout<<"old 45deg controller with 33deg plant: "<<previousControl.status().state<<' '<<previousControl.status().reason<<'\n';
                // An ideal noise-free model can recover this mismatch. Record
                // its outcome rather than claiming it reproduces the field fault.
                require(std::abs(space.dimensions().fullLockDegrees-33.f)<1e-5f,"New response model not applied");
            }
            const auto route=racePlanTrackableParking(space,start);
            std::cout<<"field route="<<route.reason<<" expanded="<<route.expanded<<" candidates="<<route.geometricCandidates<<'\n';
            if(route.valid&&saved.contains("entry")) {
                auto crossing=[](const std::vector<RaceParkingPose> &path,float angle) {
                    for(size_t i=1;i<path.size();++i)if(path[i].yaw>=angle&&path[i-1].yaw<angle) {
                        const float t=(angle-path[i-1].yaw)/(path[i].yaw-path[i-1].yaw);
                        return path[i-1].z+t*(path[i].z-path[i-1].z);
                    }
                    return 99.f;
                };
                std::vector<RaceParkingPose> old;
                for(const auto &p:saved.at("entry"))old.push_back({p[0].get<float>(),p[1].get<float>(),p[2].get<float>()});
                std::cout<<"turn at yaw=.35 rad: old front z="<<crossing(old,.35f)<<" new front z="<<crossing(route.entry,.35f)
                    <<" earlier_m="<<crossing(old,.35f)-crossing(route.entry,.35f)<<'\n';
                if(folder.filename()=="parking-1793274393822") {
                    require(crossing(old,.35f)-crossing(route.entry,.35f)>.03f,"Current route did not turn materially earlier");
                    nlohmann::json stopped;std::ifstream(folder/"parking-result.json")>>stopped;
                    const auto p=stopped.at("pose");RaceParkingPose observed{p[0].get<float>(),p[1].get<float>(),p[2].get<float>()};
                    require(!space.free(observed)&&space.freeObserved(observed),"0.6mm observed mouth crossing still locks parking");
                    observed.z-=.02f;
                    require(!space.freeObserved(observed),"Mouth tolerance hides a 2cm crossing");
                }
            }
            if(!route.valid) {
                for(float scale:{1.f,.9f,.8f,.7f}) {
                    const auto geometric=racePlanParking(space,start,scale);
                    std::cout<<"diagnostic curvature="<<scale<<" geometric="<<geometric.reason<<'\n';
                    if(!geometric.valid)continue;
                    RaceParkingControl debug(space);RaceParkingObservation sample{start,1,1,1,true};debug.arm(geometric,sample,1,true);
                    double time=1;
                    for(int n=0;n<3500&&!debug.done()&&!debug.failed();++n) {
                        time+=.04;sample.captured=time;++sample.sequence;const auto c=debug.update(sample,time,.04f,true);
                        sample.pose=raceParkingAdvance(sample.pose,c.speed*.04f,c.curvature,space.dimensions());
                    }
                    std::cout<<"diagnostic result="<<debug.status().reason<<" pose="<<sample.pose.x<<','<<sample.pose.z<<','<<sample.pose.yaw<<" parked="<<space.parked(sample.pose)<<" goalYaw="<<geometric.entry.back().yaw<<'\n';
                }
            }
            require(route.valid,"Field start lacks a route under continuous steering controller");
            RaceParkingControl controller(space);RaceParkingObservation o{start,1,1,1,true};require(controller.arm(route,o,1,true),"Cannot arm new field route");
            float wheel=0;double t=1;int waits=0;float maxStep=0;std::string state;
            nlohmann::json trace=nlohmann::json::array();
            for(int n=0;n<3500&&!controller.done()&&!controller.failed();++n) {
                t+=.08;o.captured=t;++o.sequence;const auto c=controller.update(o,t,.08f,true);
                const auto &d=space.dimensions();const float target=std::atan(d.wheelbase*c.curvature);
                const float stepLimit=d.fullLockDegrees*float(CV_PI)/180*600/400*.08f;
                const float prior=wheel;wheel+=std::clamp(target-wheel,-stepLimit,stepLimit);
                maxStep=std::max(maxStep,std::abs(wheel-prior));
                const bool waiting=std::abs(target-wheel)>d.fullLockDegrees*float(CV_PI)/180*35/400;waits+=waiting&&c.speed!=0;
                const float speed=waiting?0:c.speed;
                o.pose=raceParkingAdvance(o.pose,speed*.08f,std::tan(wheel)/d.wheelbase,d);
                require(space.free(o.pose),"New field model violates strict body/rear boundary");
                if(state!=c.state){std::cout<<t<<' '<<c.state<<' '<<c.reason<<'\n';state=c.state;}
                trace.push_back({{"time",t},{"pose",{o.pose.x,o.pose.z,o.pose.yaw}},{"speed",speed},{"wheel",wheel},{"state",c.state}});
            }
            require(controller.done(),"Field smooth route model cannot complete entry/dwell/exit");
            require(waits==0,"Continuous steering still creates stop-go waits");
            const auto output=std::filesystem::path(argv[2]);std::filesystem::create_directories(output);
            nlohmann::json path=nlohmann::json::array();for(auto pose:route.entry)path.push_back({pose.x,pose.z,pose.yaw});
            std::ofstream(output/"parking-smooth-field.json")<<nlohmann::json({{"entry",path},{"trace",trace},
                {"steering_wait_frames",waits},{"max_wheel_step_rad",maxStep},{"physical_validation",false}}).dump(2);
            std::cout<<"PASS current field start at 80ms cycle, continuous steering, strict body/rear bounds, entry/3s/reverse exit; steering_wait_frames="<<waits<<'\n';return 0;
        }
        if(argc==3&&std::string(argv[1])=="--stationary-start") {
            const auto folder=std::filesystem::path(argv[2]);nlohmann::json saved;
            std::ifstream(folder/"parking-route.json")>>saved;
            RaceParkingRoute route;route.valid=true;
            for(const auto &p:saved.at("entry"))route.entry.push_back({p[0].get<float>(),p[1].get<float>(),p[2].get<float>()});
            route.exit=route.entry;std::reverse(route.exit.begin(),route.exit.end());
            RaceParkingSpace space;space.keepRearWithinStart(route.entry.front());RaceParkingControl controller(space);
            std::ifstream input(folder/"parking-control.jsonl");std::string line;bool armed=false,sawOldFault=false;int checked=0;
            double previous=0;
            while(std::getline(input,line)) {
                const auto row=nlohmann::json::parse(line);const auto frame=row.at("frame").get<int>();
                // This is the observed stationary steering-only interval. Later
                // frames were recorded after the original fault and are not a
                // substitute for the unobserved response to the repaired command.
                if(frame<131||frame>134)continue;
                require(row.at("command_speed").get<float>()==0,"Evidence includes translation commands");
                const auto p=row.at("pose");RaceParkingObservation o{{p[0].get<float>(),p[1].get<float>(),p[2].get<float>()},
                    row.at("captured").get<double>(),uint64_t(frame),row.at("motion_epoch").get<int>(),row.at("motion_valid").get<bool>()};
                const double now=row.at("time").get<double>();
                if(!armed){require(controller.arm(route,o,now,true),"Stationary actual observation cannot arm");armed=true;}
                else {
                    sawOldFault|=!space.free(o.pose);
                    const auto c=controller.update(o,now,float(now-previous),true);
                    require(!controller.failed(),c.reason);require(c.speed>0,"Noise repair has no feasible forward command");++checked;
                }
                previous=now;
            }
            require(armed&&sawOldFault&&checked==3,"Real false rear-plane fault not exercised");
            auto outside=route.entry.front();outside.z-=.003f;
            require(!space.freeObserved(outside),"3mm rear violation hidden by noise allowance");
            require(!space.free(outside),"Planner/prediction rear plane relaxed");
            outside=route.entry.front();outside.x=.23f;
            require(!space.freeObserved(outside),"Side boundary relaxed by rear allowance");
            std::cout<<"PASS actual stationary start frames 131-134; 0.054mm false rear violation tolerated; 3mm/side violations rejected\n";
            return 0;
        }
        if(argc==4&&std::string(argv[1])=="--attempt") {
            const auto folder=std::filesystem::path(argv[3]);nlohmann::json attempt;
            std::ifstream(folder/"parking-teaching-attempt.json")>>attempt;
            const auto a=attempt.at("scaled_near_m"),b=attempt.at("scaled_far_m");
            cv::Point2f origin(a[0].get<float>(),a[1].get<float>()),far(b[0].get<float>(),b[1].get<float>());
            const auto along=(far-origin)*(1.f/.41f);const cv::Point2f right(along.y,-along.x);const auto offset=-origin;
            const RaceParkingPose start{offset.dot(right)-.225f,offset.dot(along),std::atan2(along.x,along.y)};
            RaceParkingSpace space;space.keepRearWithinStart(start);std::filesystem::create_directories(argv[2]);
            std::cout<<"actual clicked start="<<start.x<<','<<start.z<<','<<start.yaw<<" rear="<<space.rearLimit()<<'\n';
            nlohmann::json report=nlohmann::json::array();
            for(float scale:{1.f,.9f,.8f,.7f}) {
                const auto route=racePlanParking(space,start,scale);nlohmann::json row={{"scale",scale},{"geometry",route.reason},{"expanded",route.expanded}};
                std::cout<<"scale="<<scale<<" geometry="<<route.reason<<" expanded="<<route.expanded<<'\n';
                if(route.valid) {
                    RaceParkingControl control(space);RaceParkingObservation o{start,1,1,1,true};control.arm(route,o,1,true);
                    float wheel=0;double t=1;std::string state;
                    for(int n=0;n<3500&&!control.done()&&!control.failed();++n) {
                        t+=.04;o.captured=t;++o.sequence;const auto command=control.update(o,t,.04f,true);
                        const float target=std::atan(.2f*command.curvature);wheel+=std::clamp(target-wheel,-.047124f,.047124f);
                        const float speed=std::abs(target-wheel)>.0687f?0:command.speed;
                        o.pose=raceParkingAdvance(o.pose,speed*.04f,std::tan(wheel)/.2f,{});
                        if(state!=command.state){std::cout<<" t="<<t<<" state="<<command.state<<" reason="<<command.reason<<" segment="<<command.segment<<'\n';state=command.state;}
                        if(control.failed())row["model_stop"]={{"reason",command.reason},{"segment",command.segment},{"time",t},{"pose",{o.pose.x,o.pose.z,o.pose.yaw}}};
                    }
                    row["model_complete"]=control.done();
                }
                report.push_back(row);
            }
            std::ofstream(std::filesystem::path(argv[2])/"parking-current-attempt-analysis.json")<<report.dump(2);
            const auto resolved=racePlanTrackableParking(space,start);
            std::cout<<"combined result="<<resolved.reason<<" total nodes="<<resolved.expanded<<" candidates="<<resolved.geometricCandidates<<" rejected="<<resolved.modelRejected<<'\n';
            for(auto s:resolved.searches)std::cout<<"search weight="<<s.heuristicWeight<<" curvature="<<s.curvatureScale<<" nodes="<<s.expanded<<" candidates="<<s.candidates<<" rejected="<<s.rejected<<" reason="<<s.reason<<'\n';
            require(resolved.valid,"Current actual clicked pose has no model-trackable route");
            require(!resolved.searches.empty()&&resolved.geometricCandidates>=1,"Current search diagnostics missing");
            nlohmann::json path=nlohmann::json::array();for(auto p:resolved.entry)path.push_back({p.x,p.z,p.yaw});
            std::ofstream(std::filesystem::path(argv[2])/"parking-current-resolved.json")<<nlohmann::json({
                {"valid",resolved.valid},{"expanded",resolved.expanded},{"geometric_candidates",resolved.geometricCandidates},
                {"model_rejected",resolved.modelRejected},{"start",{start.x,start.z,start.yaw}},
                {"rear_boundary_m",space.rearLimit()},{"entry",path},{"physical_validation",false}}).dump(2);
            RaceGroundCalibration calibration;calibration.load((folder/"ground-calibration.json").string(),false);
            RaceGroundMap map(calibration);map.setMetricScale(attempt.at("scale").get<float>());
            auto overlay=cv::imread((folder/"parking-teaching-attempt.png").string());
            auto pixel=[&](cv::Point2f p){return map.toImage(origin+right*(p.x+.225f)+along*p.y);};
            auto line=[&](cv::Point2f a,cv::Point2f b,cv::Scalar color) {
                a=pixel(a);b=pixel(b);if(std::isfinite(a.x)&&std::isfinite(a.y)&&std::isfinite(b.x)&&std::isfinite(b.y)&&
                    std::abs(a.x)<5000&&std::abs(a.y)<5000&&std::abs(b.x)<5000&&std::abs(b.y)<5000)cv::line(overlay,a,b,color,1);
            };
            auto bay=space.bay();for(size_t i=0;i<bay.size();++i)line(bay[i],bay[(i+1)%bay.size()],{0,255,255});
            for(size_t i=1;i<resolved.entry.size();++i)line({resolved.entry[i-1].x,resolved.entry[i-1].z},{resolved.entry[i].x,resolved.entry[i].z},{0,255,0});
            cv::imwrite((std::filesystem::path(argv[2])/"parking-current-resolved.png").string(),overlay);
            std::cout<<"resolved candidates="<<resolved.geometricCandidates<<" rejected="<<resolved.modelRejected<<" expanded="<<resolved.expanded<<'\n';
            std::cout<<"PASS actual clicked parking attempt produces a model-trackable forward entry/exit\n";return 0;
        }
        if(argc>=4&&std::string(argv[1])=="--mouth-samples") {
            std::filesystem::create_directories(argv[2]);
            for(int i=3;i<argc;++i) {
                const auto folder=std::filesystem::path(argv[i]);RaceGroundCalibration calibration;
                calibration.load((folder/"ground-calibration.json").string(),false);RaceGroundMap map(calibration);
                const auto image=cv::imread((folder/(std::filesystem::exists(folder/"taught-bay.png")?"taught-bay.png":"parking-teaching-attempt.png")).string());require(!image.empty(),"Missing actual bay sample");
                std::vector<cv::Point2f> junctions;const auto mouths=raceDetectParkingMouths(image,map,{},&junctions);
                for(auto j:junctions)std::cout<<"junction="<<j<<'\n';
                std::cout<<folder.filename()<<" detected mouths="<<mouths.size()<<'\n';
                auto overlay=image.clone();
                bool forwardRoute=false;
                for(const auto &m:mouths) {
                    std::cout<<"mouth near="<<m.nearGround<<" far="<<m.farGround<<" support="<<m.support<<'\n';
                    cv::circle(overlay,m.nearImage,3,{0,0,255},-1);cv::circle(overlay,m.farImage,3,{0,255,0},-1);
                    require(m.nearGround.y>.4f,"Clipped foreground marking accepted as a complete bay mouth");
                    const auto check=raceCheckParkingMouth(m.nearGround,m.farGround);
                    const auto along=(check.far-check.near)*(1.f/.41f);const cv::Point2f right(along.y,-along.x);
                    const auto offset=-check.near;RaceParkingPose start{offset.dot(right)-.225f,offset.dot(along),std::atan2(along.x,along.y)};
                    RaceParkingSpace bounded;bounded.keepRearWithinStart(start);const auto route=racePlanTrackableParking(bounded,start);
                    std::cout<<"automatic forward route="<<route.reason<<" expanded="<<route.expanded<<'\n';
                    if(route.valid) {
                        RaceParkingControl control(bounded);RaceParkingObservation o{start,1,1,5,true};
                        require(control.arm(route,o,1,true),"Auto-selected route cannot arm");double t=1;float wheel=0;
                        for(int n=0;n<3500&&!control.done()&&!control.failed();++n) {
                            t+=.04;o.captured=t;++o.sequence;const auto command=control.update(o,t,.04f,true);
                            const float requested=std::atan(.20f*command.curvature);
                            wheel+=std::clamp(requested-wheel,-.047124f,.047124f);
                            const float speed=std::abs(requested-wheel)>.0687f?0:command.speed;
                            o.pose=raceParkingAdvance(o.pose,speed*.04f,std::tan(wheel)/.20f,{});
                            require(bounded.free(o.pose),"Auto-selected simulation crossed body/rear boundary");
                            if(control.failed())std::cout<<"auto controller stop="<<command.reason<<" t="<<t<<" segment="<<command.segment<<" pose="<<o.pose.x<<','<<o.pose.z<<','<<o.pose.yaw<<" contained="<<bounded.parked(o.pose)<<" start="<<start.x<<','<<start.z<<','<<start.yaw<<'\n';
                        }
                        forwardRoute|=control.done();
                    }
                }
                cv::imwrite((std::filesystem::path(argv[2])/("parking-auto-sample-"+std::to_string(i-2)+".png")).string(),overlay);
                require(!mouths.empty(),"No mouth found in actual sample");
                require(forwardRoute,"Automatic sample has no forward-only gate-constrained route");
                require(raceDetectParkingMouths(cv::Mat(240,320,CV_8UC3,cv::Scalar(0,0,0)),map).empty(),"Blank image gives a bay");
                require(raceDetectParkingMouths(cv::Mat(240,320,CV_8UC3,cv::Scalar(255,255,255)),map).empty(),"White image gives a bay");
            }
            std::cout<<"PASS saved-image parking mouth detection and blank/white rejection; development samples only\n";return 0;
        }
        RaceParkingDimensions d;RaceParkingSpace space(d);
        require(std::abs(d.clearance-.005f)<1e-6f,"Field-trial footprint allowance is not the requested 5mm");
        const auto closeMouth=raceCheckParkingMouth({-.175f,.02f},{-.175f,.43f},d);
        require(closeMouth.valid,"Visible 2cm mouth unnecessarily requires backing toward gate");
        require(raceCheckParkingMouth({-.175f,.15f},{-.175f,.56f},d).valid,"Backing up does not clear close-mouth guard");
        require(std::string(raceCheckParkingMouth({-.175f,.56f},{-.175f,.15f},d).reason)=="P_NEAR_FAR_ORDER_REVERSED",
            "Actual reversed clicks lack a distinct explanation");
        require(std::abs(d.stagger()-.153948f)<1e-5,"41cm side/38cm depth mismatch");
        auto bay=space.bay();for(size_t i=0;i<4;++i)require(std::abs(cv::norm(bay[i]-bay[(i+1)%4])-.41f)<1e-5,"Bay is not the measured rhombus");
        require(space.free({0,-.4f,0})&&!space.parked({0,-.4f,0}),"Aisle staging is not distinct from parked");
        require(!space.free({-.5f,-.2f,.7f}),"Body crosses bay divider/outside road");
        RaceParkingPose pose{0,-.35f,.2f};auto next=raceParkingAdvance(pose,.08f,2.f,d);
        auto back=raceParkingAdvance(next,-.08f,2.f,d);
        require(std::hypot(back.x-pose.x,back.z-pose.z)<1e-5&&std::abs(back.yaw-pose.yaw)<1e-5,"Reverse arc is not inverse of forward arc");
        const auto plan=racePlanTrackableParking(space,{0,-.4f,0});
        std::cout<<"planner nodes="<<plan.expanded<<" reason="<<plan.reason<<" points="<<plan.entry.size()<<'\n';
        require(plan.valid&&!plan.entry.empty()&&plan.exit.size()==plan.entry.size(),"Measured bay lacks preview entry/reverse route");
        require(space.parked(plan.entry.back()),"Parking success is not full-body containment");
        for(size_t i=0;i<plan.entry.size();++i) {
            require(space.free(plan.entry[i]),"Planned body crosses road/bay boundary");
            const auto &a=plan.entry[i],&b=plan.exit[plan.entry.size()-1-i];
            require(a.x==b.x&&a.z==b.z&&a.yaw==b.yaw,"Exit is not the matched reverse route");
        }
        require(!racePlanParking(space,{1,0,0}).valid,"Unknown outside area accepted as aisle");
        // Use the measured 1793269950092 start for the finite-slew gate case.
        RaceParkingSpace gateSpace(d);const RaceParkingPose gateStart{.026019454f,-.271356314f,-.088376746f};
        gateSpace.keepRearWithinStart(gateStart);
        require(gateSpace.free(gateStart)&&!gateSpace.free({gateStart.x,gateStart.z-.005f,gateStart.yaw}),
            "Rear gate boundary permits backing behind initial footprint");
        const auto gateRoute=racePlanTrackableParking(gateSpace,gateStart);
        std::cout<<"gate route="<<gateRoute.reason<<" expanded="<<gateRoute.expanded<<'\n';
        require(gateRoute.valid,"Visible near mouth lacks a forward-only gate-constrained route");
        for(const auto &p:gateRoute.entry)require(gateSpace.free(p),"Entry swings body behind initial gate boundary");
        {
            RaceParkingControl gateControl(gateSpace);RaceParkingObservation o{gateStart,1,1,4,true};
            require(gateControl.arm(gateRoute,o,1,true),"Gate-bounded route cannot arm");
            double t=1;float wheel=0;
            for(int i=0;i<3500&&!gateControl.done()&&!gateControl.failed();++i) {
                t+=.04;o.captured=t;++o.sequence;const auto command=gateControl.update(o,t,.04f,true);
                const float requested=std::atan(d.wheelbase*command.curvature);
                wheel+=std::clamp(requested-wheel,-.047124f,.047124f);
                const float speed=std::abs(requested-wheel)>.0687f?0:command.speed;
                o.pose=raceParkingAdvance(o.pose,speed*.04f,std::tan(wheel)/d.wheelbase,d);
                require(gateSpace.free(o.pose),"Closed-loop parking crosses starting rear boundary");
                if(gateControl.failed())std::cerr<<"gate controller fault="<<command.reason<<" t="<<t<<" segment="<<command.segment<<" pose="<<o.pose.x<<','<<o.pose.z<<','<<o.pose.yaw<<" contained="<<gateSpace.parked(o.pose)<<" goal="<<gateRoute.entry.back().x<<','<<gateRoute.entry.back().z<<','<<gateRoute.entry.back().yaw<<'\n';
            }
            require(gateControl.done(),"Gate-bounded forward entry/3s dwell/reverse exit did not finish");
        }
        auto invalid=d;invalid.depth=.5f;bool rejected=false;try{invalid.validate();}catch(...){rejected=true;}
        require(rejected,"Impossible rhombus accepted");
        RaceParkingDwell dwell;
        require(!dwell.ready(1,true,true,true,true)&&!dwell.ready(3.99,true,true,true,true),"Dwell completes before three stopped seconds");
        require(dwell.ready(4,true,true,true,true),"Three stopped seconds do not permit verified exit");
        require(!dwell.ready(4.1,false,true,true,true)&&!dwell.ready(4.2,true,true,true,true),"Lost observation does not reset dwell");
        require(!dwell.ready(10,true,true,true,false),"Unknown reversing silently permits exit");
        require(!dwell.ready(11,true,false,true,true)&&!dwell.ready(12,true,true,false,true),"Moving/outside vehicle counted as parked");
        RaceParkingObservation observation{plan.entry.front(),1,1,7,true};
        RaceParkingControl unverified(space);
        require(!unverified.arm(plan,observation,1,false),"Unverified reverse can arm real parking");
        RaceParkingControl controller(space);require(controller.arm(plan,observation,1,true),"Verified parking cannot arm");
        double time=1,dwellStarted=-1,exitStarted=-1;bool entered=false,reversed=false;float actualWheelAngle=0;
        for(int step=0;step<3500&&!controller.done()&&!controller.failed();++step) {
            time+=.04;observation.captured=time;++observation.sequence;
            auto command=controller.update(observation,time,.04f,true);
            if(std::string(command.state)=="PARK_DWELL"&&dwellStarted<0)dwellStarted=time;
            if(command.speed>0)entered=true;
            if(command.speed<0){reversed=true;if(exitStarted<0)exitStarted=time;}
            // Independent actuator model: wheel angle changes at 67.5 deg/s,
            // corresponding to the command PWM slew; motion waits for steering.
            const float targetAngle=std::atan(d.wheelbase*command.curvature);
            actualWheelAngle+=std::clamp(targetAngle-actualWheelAngle,-.047124f,.047124f);
            const float actualSpeed=std::abs(targetAngle-actualWheelAngle)>.0687f?0:command.speed;
            observation.pose=raceParkingAdvance(observation.pose,actualSpeed*.04f,std::tan(actualWheelAngle)/d.wheelbase,d);
            require(space.free(observation.pose),"Ideal closed-loop simulation crossed a bay/aisle edge");
            if(controller.failed())std::cerr<<"controller fault="<<command.reason<<" t="<<time<<" segment="<<command.segment<<" pose="<<observation.pose.x<<','<<observation.pose.z<<','<<observation.pose.yaw<<'\n';
        }
        require(controller.done()&&entered&&reversed,"Ideal visual closed-loop entry/dwell/exit did not complete");
        require(exitStarted-dwellStarted>=3,"Reverse begins before measured stationary three-second dwell");
        RaceParkingControl interrupted(space);observation={plan.entry.front(),1,1,7,true};
        require(interrupted.arm(plan,observation,1,true),"Hold test cannot arm");
        auto hold=interrupted.update(observation,1.05,.05f,false);
        require(hold.speed==0,"Traffic stop does not suppress parking");
        observation.captured=1.8;++observation.sequence;
        require(interrupted.update(observation,1.8,.05f,false).speed==0&&interrupted.failed(),"Long traffic interruption restarts without reteach");
        RaceParkingControl discontinuous(space);observation={plan.entry.front(),1,1,7,true};discontinuous.arm(plan,observation,1,true);
        observation.epoch=8;observation.captured=1.05;++observation.sequence;
        require(discontinuous.update(observation,1.05,.05f,true).speed==0&&discontinuous.failed(),"Odometry reset permits motion");
        require(std::string(discontinuous.status().reason)=="PARK_ODOMETRY_EPOCH_CHANGED",
            "Disarmed UI overwrites original parking fault");
        discontinuous.abort();
        require(std::string(discontinuous.status().reason)=="PARK_ODOMETRY_EPOCH_CHANGED",
            "Operator stop after a fault destroys the first failure reason");
        RaceParkingControl stale(space);observation={plan.entry.front(),1,1,7,true};stale.arm(plan,observation,1,true);
        require(stale.update(observation,1.2,.05f,true).speed==0,"Stale visual pose permits motion");
        std::cout<<"PASS ideal measured-pose closed loop, entry/3s dwell/reverse exit, traffic hold and odometry reset/stale guards\n";
        std::cout<<"PASS measured bay geometry, full-body union containment, reversible preview path and verified three-second dwell\n";
    }catch(const std::exception &e){std::cerr<<"FAIL "<<e.what()<<'\n';return 1;}
}
