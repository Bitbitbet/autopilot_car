#include "trackview_pilot.hpp"
#include "track.hpp"
#include <iostream>
#include <limits>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <unistd.h>
void check(bool ok,const char *message) { if(!ok) throw std::runtime_error(message); }
using Edges=std::pair<std::vector<PointX>,std::vector<PointX>>;
Edges lane(int shift=0,bool clipRight=false,bool clipBoth=false) {
    Edges e;
    for(int row=219;row>=30;row-=2) {
        const int width=70+row/2;
        e.first.emplace_back(row,clipBoth?0:160+shift-width/2);
        e.second.emplace_back(row,(clipRight||clipBoth)?319:160+shift+width/2);
    }
    return e;
}
void feed(AutoPilot &p,const Edges &e,int frames=15,float dt=1.f/30,float speed=0) {
    for(int i=0;i<frames;++i) p.update(e.first,e.second,dt,speed);
}
float drive(AutoPilot &p,const Edges &e,float cruise,int frames=90,float dt=1.f/30) {
    float speed=0;
    for(int i=0;i<frames;++i) {
        p.update(e.first,e.second,dt,speed);speed=p.speedFor(cruise,dt);
    }
    return speed;
}
Edges readEdges(const std::filesystem::path &path) {
    std::ifstream in(path); check(bool(in),"missing现场 edge fixture");
    Edges e; std::string line; std::getline(in,line);
    while(std::getline(in,line)) {
        std::replace(line.begin(),line.end(),',',' ');
        std::istringstream record(line); std::string side; int row,col;
        if(record>>side>>row>>col) (side=="left"?e.first:e.second).emplace_back(row,col);
    }
    return e;
}
int main(int argc, char **argv) {
    try {
        Track t; t.rowCutBottom=20;
        cv::Mat black=cv::Mat::zeros(240,320,CV_8UC1); t.handle(black);
        check(t.pointsEdgeLeft.empty(),"black frame accepted");
        cv::Mat road=black.clone();
        cv::rectangle(road,{40,0},{279,239},cv::Scalar(255),-1); t.handle(road);
        check(!t.pointsEdgeLeft.empty() && t.pointsEdgeLeft.front().x==219,"bottom crop/scan start wrong");
        t.handle(true,0); t.handle(true,65535);
        bool rejected=false;
        try { t.handle(cv::Mat::zeros(240,320,CV_8UC3)); } catch(const std::invalid_argument &) { rejected=true; }
        check(rejected && t.pointsEdgeLeft.empty(),"invalid input retained stale track");
        std::cout<<"PASS bounds, bottom crop, rescan and invalid frame\n";

        {
            TrackviewConfig ceiling;
            ceiling.validate();
            check(ceiling.maxSpeed==1.5f && ceiling.cruiseSpeed==0.25f &&
                ceiling.curveSpeed==0.15f && ceiling.degradedSpeed==0.12f,
                "1.5 m/s ceiling changed other default speeds");
            const auto path=std::filesystem::temp_directory_path()/
                ("trackview-v6-ceiling-"+std::to_string(getpid())+".json");
            std::ofstream(path)<<R"({"maxSpeed":1.5})";
            TrackviewConfig loaded;loaded.load(path.string(),true);
            std::filesystem::remove(path);
            check(loaded.maxSpeed==1.5f,"1.5 m/s JSON ceiling rejected");
            TrackviewConfig excessive;excessive.maxSpeed=1.51f;
            bool denied=false;try {excessive.validate();}catch(const std::invalid_argument &){denied=true;}
            check(denied,"ceiling above 1.5 m/s accepted");
            AutoPilot faster(loaded);
            feed(faster,lane());
            check(std::fabs(drive(faster,lane(),1.5f,210)-1.5f)<0.001f,
                "straight command failed to reach 1.5 m/s");
            check(std::fabs(drive(faster,lane(),2.f,210)-1.5f)<0.001f,
                "request exceeded 1.5 m/s ceiling");
            check(faster.speedFor(0.25f,1.f/30)<=0.2501f,
                "lower operator limit was not immediate");
            const auto bend=lane(65);
            check(std::fabs(drive(faster,bend,1.5f,210)-0.15f)<0.001f,
                "1.5 m/s ceiling changed the curve target");
            feed(faster,lane());
            const auto single=lane(0,true);
            check(std::fabs(drive(faster,single,1.5f,210)-0.12f)<0.001f,
                "1.5 m/s ceiling changed the degraded target");
            faster.update({}, {}, 1.f/30);
            check(faster.speedFor(1.5f,1.f/30)==0,
                "path loss failed to stop with 1.5 m/s ceiling");
            std::cout<<"PASS 1.5 m/s default/JSON ceiling, command clamp, unchanged curve/quality targets and path-loss stop\n";
        }

        AutoPilot straight; feed(straight,lane());
        check(straight.valid && straight.ready && std::abs(straight.center-160)<=1,"straight not centered");
        check(std::abs(int(straight.servo)-1500)<=5,"straight servo");
        float speed=0; for(int i=0;i<90;++i) speed=straight.speedFor(0.5f,1.f/30);
        check(speed>0.49f,"straight speed did not reach limit");
        feed(straight,lane(30,true));
        check(straight.valid && straight.singlePoints>0,"clipped boundary did not become single side");
        check(std::abs(straight.center-190)<3,"row-specific width reconstruction wrong");
        for(int i=0;i<90;++i) speed=straight.speedFor(0.5f,1.f/30);
        check(speed<=0.1201f,"single-side target not reached after quality braking");
        AutoPilot unlearned; feed(unlearned,lane(0,true));
        check(!unlearned.valid,"single side used invented width before learning");
        AutoPilot fullWhite; feed(fullWhite,lane(0,false,true));
        check(!fullWhite.valid && fullWhite.lost,"full-width white accepted as road");
        std::cout<<"PASS real vs clipped edges, width profile and degraded speed\n";

        AutoPilot mixed; feed(mixed,lane());
        auto fewSingle=lane();
        for(size_t i=0;i<6;++i) fewSingle.second[i].y=319;
        feed(mixed,fewSingle);
        check(mixed.valid && mixed.ready && mixed.singlePoints==6 && !mixed.degraded,
            "a few single-edge rows still degrade the entire path");
        float mixedVelocity=0;
        for(const float requested:{0.25f,0.35f,0.5f}) {
            for(int i=0;i<90;++i) mixedVelocity=mixed.speedFor(requested,1.f/30);
            check(std::fabs(mixedVelocity-requested)<0.001f,"healthy mixed path failed to follow cruise limit");
        }
        check(std::string(mixed.qualityLimitReason())=="LIMIT","healthy path limit reason missing");
        const auto legacyPath=std::filesystem::temp_directory_path()/
            ("trackview-v4-config-"+std::to_string(getpid())+".json");
        std::ofstream(legacyPath)<<R"({"mixedSpeed":0.01})";
        TrackviewConfig legacy;legacy.load(legacyPath.string(),true);
        std::filesystem::remove(legacyPath);
        check(!legacy.toJson().contains("mixedSpeed"),"obsolete mixed cap still saved in v5 configuration");
        AutoPilot legacyMixed(legacy);feed(legacyMixed,lane());feed(legacyMixed,fewSingle);
        for(int i=0;i<90;++i) mixedVelocity=legacyMixed.speedFor(0.35f,1.f/30);
        check(std::fabs(mixedVelocity-0.35f)<0.001f,"legacy mixedSpeed still limits motion");
        auto manySingle=lane();
        for(size_t i=0;i<40;++i) manySingle.second[i].y=319;
        feed(mixed,manySingle);
        for(int i=0;i<90;++i) mixedVelocity=mixed.speedFor(0.5f,1.f/30);
        check(mixed.degraded && mixedVelocity<=0.1201f,"large single share not capped after braking");
        auto manyRejected=lane();
        for(size_t i=0;i<35;++i) { manyRejected.first[i].y=0;manyRejected.second[i].y=319; }
        feed(mixed,manyRejected);
        check(mixed.valid && mixed.degraded && mixed.speedFor(0.5f,1.f/30)<=0.1201f,
            "large rejected share not capped");
        check(std::string(mixed.qualityLimitReason())=="REJECT_SHARE","rejection reason wrong");
        auto lowConfidence=lane();
        for(size_t i=0;i<30;++i) {lowConfidence.first[i].y=0;lowConfidence.second[i].y=319;}
        for(size_t i=30;i<50;++i) lowConfidence.second[i].y=319;
        feed(mixed,lowConfidence);
        check(mixed.valid && !mixed.degraded && mixed.confidence<0.65f &&
            mixed.speedFor(0.5f,1.f/30)<=0.1201f,"low quality escaped protective cap");
        check(std::string(mixed.qualityLimitReason())=="LOW_QUALITY","low quality reason missing");
        TrackviewConfig custom; custom.maxSingleFraction=0.05f;
        AutoPilot strict(custom);feed(strict,lane());feed(strict,fewSingle);
        check(strict.degraded,"maxSingleFraction configuration ignored");
        TrackviewConfig invalidShare;invalidShare.maxSingleFraction=1.f;
        rejected=false;try {invalidShare.validate();}catch(...){rejected=true;}
        check(rejected,"invalid share accepted");
        std::cout<<"PASS healthy mixed cruise 0.25/0.35/0.5, legacy mixedSpeed ignored; genuine quality caps preserved\n";

        TrackviewConfig smoothConfig;smoothConfig.curveSpeed=0.20f;
        AutoPilot smooth(smoothConfig);
        check(drive(smooth,lane(),0.5f)>0.49f,"cannot prepare straight speed");
        auto hardCurve=lane(65);
        smooth.update(hardCurve.first,hardCurve.second,1.f/30,0.5f);
        const float firstBrake=smooth.speedFor(0.5f,1.f/30);
        check(firstBrake<0.5f && firstBrake>0.47f && firstBrake>smooth.speedTarget,
            "curve slowdown still jumps directly to its target");
        check(std::string(smooth.speedReason).find("CURVE_")==0,"curve reason missing");
        check(std::fabs(drive(smooth,hardCurve,0.5f)-0.20f)<0.001f,"curve target not reached");
        smooth.setCurveSpeed(0.30f);
        check(std::fabs(drive(smooth,hardCurve,0.5f)-0.30f)<0.001f,"runtime curve target ignored");
        check(smooth.speedFor(0.10f,1.f/30)<=0.1001f,"operator lower limit not applied immediately");

        AutoPilot qualityBrake;
        check(drive(qualityBrake,lane(),0.5f)>0.49f,"cannot prepare quality test");
        qualityBrake.update(manySingle.first,manySingle.second,1.f/30,0.5f);
        const float firstQuality=qualityBrake.speedFor(0.5f,1.f/30);
        check(firstQuality<0.5f && firstQuality>0.44f && qualityBrake.speedTarget<=0.1201f,
            "valid-path quality slowdown is not gradual");
        check(drive(qualityBrake,manySingle,0.5f,30)<=0.1201f,"quality target not reached");
        qualityBrake.update(lane().first,lane().second,1.f/30);
        check(std::string(qualityBrake.qualityLimitReason())=="QUALITY_RECOVERY",
            "quality limit cleared on one good frame");
        check(qualityBrake.speedFor(0.5f,1.f/30)<=0.1201f,"one good frame caused speed jump");
        check(drive(qualityBrake,lane(),0.5f,30)>0.12f,"stable quality recovery never released");
        qualityBrake.setDegradedSpeed(0.22f);
        check(std::fabs(drive(qualityBrake,manySingle,0.5f)-0.22f)<0.001f,
            "runtime quality target ignored");
        qualityBrake.update({}, {}, 1.f/30);
        check(qualityBrake.speedFor(0.5f,1.f/30)==0,"loss incorrectly uses soft braking");

        // 仅远处出现弯道的测试场景：速度预瞄的图像行范围可单独调整。
        Edges farBend=lane();
        for(size_t i=0;i<farBend.first.size();++i) {
            const int row=farBend.first[i].x,shift=std::max(0,80-row)*4/5;
            farBend.first[i].y+=shift;farBend.second[i].y+=shift;
        }
        TrackviewConfig nearConfig;nearConfig.curveLookaheadRow=100;
        AutoPilot fullPreview,nearPreview(nearConfig);
        feed(fullPreview,farBend);feed(nearPreview,farBend);
        check(fullPreview.severity>0 && nearPreview.severity==0 &&
            std::fabs(fullPreview.curveHeading)>std::fabs(nearPreview.curveHeading),
            "preview setting did not suppress far-only bend speed signal");
        check(!nearPreview.curveWindowFallback,"usable near preview unexpectedly fell back");
        TrackviewConfig sparsePreview;sparsePreview.curveLookaheadRow=200;
        AutoPilot fallback(sparsePreview);feed(fallback,farBend);
        check(fallback.curveWindowFallback && fallback.severity>0,
            "insufficient preview quietly disabled curve braking");
        AutoPilot smallError;feed(smallError,lane(5));
        check(smallError.rawSeverity>0 && smallError.severity==0,
            "small straight tracking errors still trigger curve slowdown");
        for(const float dt:{1.f/15,1.f/30,1.f/60}) {
            AutoPilot rate(smoothConfig);drive(rate,lane(),0.5f,int(3/dt),dt);
            float previous=0.5f;
            for(int i=0;i<int(1/dt);++i) {
                rate.update(hardCurve.first,hardCurve.second,dt,previous);
                const float current=rate.speedFor(0.5f,dt);
                check(current>=previous-smoothConfig.deceleration*dt-0.0001f,
                    "curve deceleration exceeds configured rate");
                previous=current;
            }
            check(std::fabs(previous-0.20f)<0.005f,"braking changes with frame rate");
        }
        TrackviewConfig badBrake;badBrake.deceleration=-1;
        rejected=false;try {badBrake.validate();}catch(...){rejected=true;}
        check(rejected,"negative deceleration accepted");
        TrackviewConfig badPreview;badPreview.curveLookaheadRow=240;
        rejected=false;try {badPreview.validate();}catch(...){rejected=true;}
        check(rejected,"invalid preview row accepted");
        std::cout<<"PASS smooth curve/quality braking, recovery hold, adjustable targets, preview, deadband and 15/30/60 FPS rates\n";

        // 复现停车等待超过 v2 宽度记录 15 秒有效期的情形。
        AutoPilot paused; feed(paused,lane());
        feed(paused,{},1260,1.f/15); paused.resetControl();
        check(paused.lost && paused.speedFor(0.25f,1.f/15)==0,"blank pause failed to stop");
        feed(paused,lane(30,true),15,1.f/15);
        check(paused.valid && paused.ready && paused.singlePoints>0,"84s pause discarded learned width");
        check(std::abs(paused.center-190)<3,"post-pause single reconstruction changed");
        check(paused.speedFor(0.5f,1.f/15)<=0.1201f,"post-pause single speed unsafe");
        // 宽度观测冲突时先停用旧记录，多次观测一致后再重新学习。
        Edges narrow;
        for(int row=219;row>=30;row-=2) { narrow.first.emplace_back(row,130); narrow.second.emplace_back(row,190); }
        AutoPilot changed; feed(changed,lane()); feed(changed,narrow,1);
        auto singleNarrow=narrow;
        for(auto &v:singleNarrow.second) v.y=319;
        feed(changed,singleNarrow,1);
        check(!changed.valid && changed.missingWidthPoints>0,"contradictory profile used old width");
        feed(changed,narrow,15); feed(changed,singleNarrow,15);
        check(changed.valid && std::abs(changed.center-160)<2,"consistent new width was not relearned");
        AutoPilot shortSpan; Edges fragment;
        for(int row=219;row>=199;row-=2) { fragment.first.emplace_back(row,60); fragment.second.emplace_back(row,260); }
        feed(shortSpan,fragment);
        check(!shortSpan.valid && std::string(shortSpan.pathIssue)=="SPAN_SHORT","short fragment escaped coverage check");
        std::cout<<"PASS 84s pause, single-side cap, contradictory width revalidation and span rejection\n";

        AutoPilot a,b; feed(a,lane()); feed(b,lane()); feed(a,lane(30)); feed(b,lane(-30));
        check(a.servo>1500 && b.servo<1500,"mirror steering direction changed");
        check(std::abs(int(a.servo)+int(b.servo)-3000)<8,"mirror steering asymmetric");
        float curveSpeed=0; for(int i=0;i<90;++i) curveSpeed=a.speedFor(0.5f,1.f/30);
        check(curveSpeed<0.4f,"large error did not reduce speed");
        a.update({}, {}, 1.f/30);
        check(a.speedFor(0.5f,1.f/30)==0 && !a.lost,"first missing frame still moving");
        feed(a,{},7); check(a.lost,"time-based loss not reached");
        feed(a,lane(),1); check(!a.ready,"recovered on one frame");
        feed(a,lane(),15); check(a.ready,"stable recovery never ready");
        a.update(lane().first,lane().second,0.5f);
        check(a.lost && a.speedFor(0.5f,0.5f)==0,"stale frame allowed motion");
        std::cout<<"PASS mirrored turns, curve reduction, immediate stop, timed loss and recovery\n";

        AutoPilot p15,p30,p60;
        feed(p15,lane(),15,1.f/15); feed(p30,lane(),30,1.f/30); feed(p60,lane(),60,1.f/60);
        for(int i=1;i<=15;++i) feed(p15,lane(i*2),1,1.f/15);
        for(int i=1;i<=30;++i) feed(p30,lane(i),1,1.f/30);
        for(int i=1;i<=60;++i) feed(p60,lane(i/2),1,1.f/60);
        check(std::abs(int(p15.servo)-int(p30.servo))<15 &&
              std::abs(int(p60.servo)-int(p30.servo))<15,"frame-rate-dependent PD output");
        TrackviewConfig bad; bad.turnD=std::numeric_limits<float>::quiet_NaN();
        rejected=false; try {bad.validate();} catch(...) {rejected=true;} check(rejected,"NaN config accepted");
        std::cout<<"PASS 15/30/60 FPS consistency and configuration validation\n";
        if (argc > 1) {
            const std::filesystem::path out(argv[1]);
            // 使用之前故障时保存的图像；宽度先验为合成数据，不是实车标定值。
            for(const auto &name:{"curve-entry","curve-seam"}) {
                cv::Mat image=cv::imread((out.parent_path()/"trial_fixes_20261012"/"fixtures"/name/"binary.png").string(),0);
                check(!image.empty(),"missing saved curve fixture");
                int centers[2]{};
                for(int mirror=0;mirror<2;++mirror) {
                    AutoPilot p;
                    Edges learned;
                    for(int row=219;row>=20;row-=2) {
                        int width=40+row;
                        learned.first.emplace_back(row,160-width/2);
                        learned.second.emplace_back(row,160+width/2);
                    }
                    feed(p,learned);
                    cv::Mat frame=image.clone(); if(mirror) cv::flip(image,frame,1);
                    Track scanner; scanner.handle(frame);
                    feed(p,{scanner.pointsEdgeLeft,scanner.pointsEdgeRight});
                    centers[mirror]=p.center;
                    check(p.valid && p.ready,"saved curve has no usable path with learned width");
                    std::cout<<"FIXTURE "<<name<<" mirror="<<mirror<<" center="<<p.center
                        <<" dual="<<p.dualPoints<<" single="<<p.singlePoints<<" pwm="<<p.servo<<'\n';
                }
                check(std::abs(centers[0]+centers[1]-319)<10,"saved fixture mirror center inconsistent");
            }
            // 使用本次记录的真实边界坐标，并明确采用合成的宽度先验。
            // v2 未保存宽度模型，因此这里只验证过期行为，不验证真实物理标定。
            const auto fieldRun=out/"现场记录"/"1791885777078-8366"/"event-2";
            if(std::filesystem::exists(fieldRun/"1872-edges.csv")) {
                auto actual=readEdges(fieldRun/"1872-edges.csv");
                for(int mirror=0;mirror<2;++mirror) {
                    AutoPilot p; Edges prior;
                    for(int row=219;row>=20;row-=2) {
                        int width=40+row;
                        prior.first.emplace_back(row,160-width/2);
                        prior.second.emplace_back(row,160+width/2);
                    }
                    feed(p,prior); feed(p,{},1260,1.f/15); p.resetControl();
                    auto edges=actual;
                    if(mirror) {
                        edges.first.clear(); edges.second.clear();
                        for(auto v:actual.second) edges.first.emplace_back(v.x,319-v.y);
                        for(auto v:actual.first) edges.second.emplace_back(v.x,319-v.y);
                    }
                    feed(p,edges,15,1.f/15);
                    check(p.valid && p.ready && p.singlePoints>0,"saved stopped frame cannot reconstruct after pause");
                    check(p.speedFor(0.5f,1.f/15)<=0.1201f,"actual single-side fixture exceeded cap");
                    std::cout<<"现场 event2 frame1872 synthetic-prior mirror="<<mirror
                        <<" valid="<<p.valid<<" dual="<<p.dualPoints<<" single="<<p.singlePoints
                        <<" rejected="<<p.rejectedPoints<<" center="<<p.center<<'\n';
                }
                AutoPilot noPrior; feed(noPrior,actual,15,1.f/15);
                check(!noPrior.valid && noPrior.missingWidthPoints>0 &&
                    std::string(noPrior.pathIssue)=="SPAN_SHORT","fixture without width unexpectedly accepted");
                std::cout<<"PASS stopped-frame expiry scenario and mirrored reconstruction; no-prior remains rejected\n";
            }
            if(argc>2) {
                const std::filesystem::path event(argv[2]);
                Edges prior;std::ifstream input(event/"width-profile.csv");
                check(bool(input),"missing v3 recorded width profile");
                std::string line;std::getline(input,line);
                while(std::getline(input,line)) {
                    std::replace(line.begin(),line.end(),',',' ');std::istringstream rec(line);
                    int row,observations,pending;float width;
                    if(rec>>row>>width>>observations>>pending && observations>=3 && pending==0) {
                        const int w=int(std::lround(width)),l=160-w/2;
                        prior.first.emplace_back(row,l);prior.second.emplace_back(row,l+w);
                    }
                }
                AutoPilot p;feed(p,prior);
                auto actual=readEdges(event/"3347-edges.csv");feed(p,actual);
                check(p.valid && p.ready && p.singlePoints>0 && !p.degraded,
                    "recorded mostly-dual scene still degrades");
                float velocity=0;for(int i=0;i<90;++i) velocity=p.speedFor(0.35f,1.f/30);
                check(velocity>0.25f && velocity<=0.3501f,"recorded healthy mixed scene still has extra 0.25 cap");
                std::cout<<"PASS recorded v3 frame3347 with recorded width prior: dual="<<p.dualPoints
                    <<" single="<<p.singlePoints<<" reject="<<p.rejectedPoints<<" confidence="<<p.confidence
                    <<" degraded="<<p.degraded<<" speed="<<velocity<<'\n';
            }
            // 完整程序回放测试：仅使用单位矩阵标定和合成视频。
            const auto replay=out/"replay";
            std::filesystem::create_directories(replay/"res"/"calibration");
            std::filesystem::create_directories(replay/"bin");
            cv::FileStorage calibration((replay/"res"/"calibration"/"calibration.xml").string(),cv::FileStorage::WRITE);
            cv::Mat matrix=(cv::Mat_<double>(3,3)<<200,0,160,0,200,120,0,0,1);
            calibration<<"cameraMatrix"<<matrix<<"distCoeffs"<<cv::Mat::zeros(1,5,CV_64F);
            calibration.release();
            cv::VideoWriter video((replay/"sample.avi").string(),cv::VideoWriter::fourcc('M','J','P','G'),30,{320,240});
            check(video.isOpened(),"cannot make replay video");
            for(int frame=0;frame<120;++frame) {
                cv::Mat binary=cv::Mat::zeros(240,320,CV_8UC1);
                if(frame<100) for(int row=0;row<240;++row) {
                    int shift=frame<60?0:std::min(30,(frame-60));
                    int width=80+row/2;
                    cv::line(binary,{160+shift-width/2,row},{160+shift+width/2,row},cv::Scalar(255));
                }
                cv::Mat bgr; cv::cvtColor(binary,bgr,cv::COLOR_GRAY2BGR); video.write(bgr);
            }
            video.release();
            std::cout<<"PASS saved curve fixtures and mirrored paths; generated executable replay\n";
        }
        return 0;
    } catch(const std::exception &e) { std::cerr<<"FAIL "<<e.what()<<'\n'; return 1; }
}
