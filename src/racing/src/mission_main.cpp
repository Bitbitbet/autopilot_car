#include "../tasks/race_runtime.hpp"
#include "../tasks/race_obstacle.hpp"
#include "../tasks/race_navigation_worker.hpp"
#include "../tasks/race_sample_writer.hpp"
#include "../tasks/race_fork_evidence.hpp"
#include "../tasks/race_parking_tools.hpp"
#include "../tasks/race_parking_drive.hpp"
#include "../tasks/race_bypass_control.hpp"
#include "../tasks/race_virtual_boundary.hpp"
/** 比赛集成入口：使用 master 原生640巡线，叠加异步感知、选路和任务仲裁。
 * racing 与 trackview_race 共用本入口。 */
#include "../tasks/race_manager.hpp"
#include "../tasks/race_replay.hpp"
#include "../tasks/race_preview.hpp"
#include "../tasks/race_trace.hpp"
#include "../tasks/race_ui.hpp"
#include "../tasks/race_trial.hpp"
#include "control.hpp"
#include "joystick.hpp"
#include "task_preprocess.hpp"
#include "task_track.hpp"
#include "trackview_pilot.hpp"
#include "trackview_display.hpp"
#include <chrono>
#include <csignal>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <thread>

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;
namespace {
std::atomic_bool stopping{false};
static_assert(std::atomic_bool::is_always_lock_free);
void onSignal(int) { stopping.store(true); }
}
enum class Mode { Manual, Auto };

// 在驾驶线程之后声明：析构时先禁止运动，再等待驾驶线程结束。
struct StopOnExit {
    std::shared_ptr<CarControl> car;
    ~StopOnExit() {
        stopping = true;
        if(car) { try { car->latchStop(); } catch(...) {} }
    }
};

class Diagnostics {
    struct Frame {
        int number; cv::Mat camera, binary;
        std::vector<PointX> left, right;
    };
    fs::path directory;
    std::ofstream csv;
    std::ofstream timing;
    std::deque<Frame> recent;
    bool enabled = false;
    int number = 0, eventNumber = 0;
public:
    explicit Diagnostics(const TrackviewConfig &config) {
        if (!config.diagnostics) return;
        try {
            const auto stamp = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count();
            directory = fs::path(config.logDirectory) / (std::to_string(stamp)+"-"+std::to_string(getpid()));
            fs::create_directories(directory);
            csv.open(directory/"control.csv");
            if (!csv) throw std::runtime_error("Cannot open diagnostic log");
            csv << "frame,time_s,dt_s,mode,panic,valid,ready,lost,degraded,confidence,dual,single,rejected,center,error,heading,severity,cruise,speed,pwm,path_issue,width_rows,missing_width,untrusted,width_range,jump,display_age_ms,single_fraction,quality_cap,cap_reason,speed_target,speed_reason,curve_limit,degraded_limit,curve_raw,curve_heading,curve_window_fallback\n";
            std::ofstream(directory/"config.json") << config.toJson().dump(2);
            timing.open(directory/"timing.csv");
            timing << "frame,capture_ms,preprocess_ms,track_ms,control_ms,record_publish_ms\n";
            enabled = true;
            std::cout << "[Log] " << directory << '\n';
        } catch(const std::exception &e) {
            std::cerr << "[Warn] Diagnostics unavailable: " << e.what() << '\n';
        }
    }
    fs::path outputDirectory() const { return enabled ? directory : fs::path{}; }
    fs::path displayTimingPath() const { return enabled ? directory/"display-timing.csv" : fs::path{}; }
    void add(const cv::Mat &image,const cv::Mat &bin,const Track &track) {
        ++number;
        if (!enabled) return;
        recent.push_back({number,image.clone(),bin.clone(),track.pointsEdgeLeft,track.pointsEdgeRight});
        if (recent.size()>15) recent.pop_front();
    }
    void record(double time,float dt,Mode mode,bool panic,const RaceTaskPilot &p,float cruise,float speed,uint16_t pwm,long long displayAge) {
        if (!enabled) return;
        csv << number << ',' << time << ',' << dt << ',' << (mode==Mode::Auto?"AUTO":"MANUAL")
            << ',' << panic << ',' << p.valid << ',' << p.ready << ',' << p.lost
            << ',' << p.degraded << ',' << p.confidence << ',' << p.dualPoints
            << ',' << p.singlePoints << ',' << p.rejectedPoints << ',' << p.center
            << ',' << p.error << ',' << p.heading << ',' << p.severity << ','
            << cruise << ',' << speed << ',' << pwm << ',' << p.pathIssue
            << ',' << p.learnedWidthRows() << ',' << p.missingWidthPoints
            << ',' << p.untrustedPoints << ',' << p.widthRangePoints << ',' << p.jumpPoints
            << ',' << displayAge << ',' << p.singleFraction() << ',' << p.qualitySpeedLimit()
            << ',' << p.qualityLimitReason() << ',' << p.speedTarget << ',' << p.speedReason
            << ',' << p.curveSpeed() << ',' << p.degradedSpeed() << ',' << p.rawSeverity
            << ',' << p.curveHeading << ',' << p.curveWindowFallback << '\n';
        if (number%15==0) csv.flush();
        if (!csv) { enabled=false; std::cerr << "[Warn] Diagnostic disk write failed.\n"; }
    }
    void recordTiming(double capture,double preprocess,double track,double control,double publish) {
        if(!enabled) return;
        timing << number << ',' << capture << ',' << preprocess << ',' << track
               << ',' << control << ',' << publish << '\n';
        if(number%15==0) timing.flush();
    }
    void event(const std::string &reason,const RaceTaskPilot &p,const cv::Mat &overlay={}) noexcept {
        if (!enabled) return;
        try {
            // 停车后才保存事件图像，避免图像编码延误运动指令。
            const auto path=directory/("event-"+std::to_string(++eventNumber));
            fs::create_directories(path);
            std::ofstream(path/"reason.txt") << reason << "\ncenter=" << p.center
                << " confidence=" << p.confidence << " dual=" << p.dualPoints
                << " single=" << p.singlePoints << " rejected=" << p.rejectedPoints
                << " path=" << p.pathIssue << " missing_width=" << p.missingWidthPoints << '\n';
            std::ofstream(path/"runtime-config.json") << p.runtimeConfig().dump(2);
            std::ofstream widths(path/"width-profile.csv");
            widths << "row,width,observations,pending_observations,age_s,refresh_due,calibrated_reference\n";
            for(const auto &w:p.widthProfile()) widths << w.row << ',' << w.width << ','
                << w.observations << ',' << w.pendingObservations << ',' << w.age << ',' << w.refreshDue << ',' << w.calibratedReference << '\n';
            for (const auto &f:recent) {
                cv::imwrite((path/(std::to_string(f.number)+"-camera.jpg")).string(),f.camera);
                cv::imwrite((path/(std::to_string(f.number)+"-binary.png")).string(),f.binary);
                std::ofstream edges(path/(std::to_string(f.number)+"-edges.csv"));
                edges << "side,row,col\n";
                for(const auto &v:f.left) edges << "left," << v.x << ',' << v.y << '\n';
                for(const auto &v:f.right) edges << "right," << v.x << ',' << v.y << '\n';
            }
            if (!overlay.empty()) cv::imwrite((path/"overlay.png").string(),overlay);
            std::ofstream points(path/"center.csv"); points << "row,col\n";
            for(const auto &v:p.centerEdge) points << v.x << ',' << v.y << '\n';
            csv.flush();
            std::cout << "[Saved] " << path << '\n';
        } catch(const std::exception &e) { std::cerr << "[Warn] Save diagnostics: " << e.what() << '\n'; }
    }
};

static void drawHud(cv::Mat &image,const RaceTaskPilot &pilot,Mode mode,bool panic,
                    float cruise,float speed,uint16_t servo,double fps,bool displaySlow,const RacePlan &race) {
    for(const auto &p:pilot.centerEdge)
        cv::circle(image,cv::Point(p.y,p.x),1,cv::Scalar(0,0,255),-1);
    int y=18;
    auto line=[&](const std::string &s) {
        cv::putText(image,s,{5,y},cv::FONT_HERSHEY_PLAIN,0.75,{0,255,255},1); y+=15;
    };
    auto fmt=[](float v) { std::ostringstream s; s<<std::fixed<<std::setprecision(2)<<v; return s.str(); };
    line(std::string(mode==Mode::Auto?"AUTO":"MANUAL")+(panic?" STOP LATCHED":"")+(pilot.lost?" LOST":""));
    line("speed="+fmt(speed)+" limit="+fmt(cruise)+" pwm="+std::to_string(servo));
    line("turn="+fmt(pilot.curveSpeed())+" qualityLimit="+fmt(pilot.degradedSpeed()));
    line(std::string(pilot.valid?"center=":"last center=")+std::to_string(pilot.center)+" err="+fmt(pilot.error));
    line("dual="+std::to_string(pilot.dualPoints)+" single="+std::to_string(pilot.singlePoints)+" reject="+std::to_string(pilot.rejectedPoints));
    line("quality="+fmt(pilot.confidence)+" curve="+fmt(pilot.severity)+" h="+fmt(pilot.curveHeading));
    line(std::string(pilot.valid?(pilot.degraded?"DEGRADED":(pilot.singlePoints>0?"MIXED":"DUAL")):"NO TRUSTED PATH")+(pilot.ready?" READY":" WAIT"));
    line("target="+fmt(pilot.speedTarget)+" "+(mode==Mode::Auto?pilot.speedReason:"MANUAL"));
    if(!pilot.valid) line(std::string(pilot.pathIssue)+" width="+std::to_string(pilot.learnedWidthRows())+" miss="+std::to_string(pilot.missingWidthPoints));
    if(displaySlow) line("DISPLAY DELAY: control independent");
    line("fps="+std::to_string(int(fps))+" weight="+pilot.weightName());
    line(std::string("race=")+race.state+" ai="+fmt(float(race.detectionAge)));
    line(race.reason);
    line("a:auto  SPACE:stop  +/-:limit  ESC:quit");
    line("[/]:turn  ,/.:quality  w:weight");
}

int raceVehicleMain(int argc,char **argv) {
    raceConfigureRuntime();
    std::signal(SIGINT,onSignal); std::signal(SIGTERM,onSignal); std::signal(SIGHUP,onSignal);
    std::shared_ptr<CarControl> car;
    std::unique_ptr<Diagnostics> log;
    RaceTaskPilot pilot;
    cv::Mat lastOverlay;
    try {
        std::string source="/dev/video0", configPath="trackview.json";
        std::string raceConfigPath="race.json", detectionPath,navigationConfigPath="navigation.json";
        bool navigationConfigRequired=false;
        bool calibrateGround=false,bypassEnabled=false,reverseCheck=false,parkingPreview=false,parkingDrive=false,reverseVerified=false;
        bool virtualBypass=false,missionDrive=false,missionLab=false;
        bool replayCorrected=false;
        std::string groundPath="ground-calibration.json";
        bool noControl=false, headless=false, replay=false, configRequired=false,legacyWindows=false;
        int displayFps=5;
        double trialSeconds=0;
        for(int i=1;i<argc;++i) {
            const std::string arg=argv[i];
            auto value=[&]() { if(++i>=argc) throw std::invalid_argument("Missing argument for "+arg); return std::string(argv[i]); };
            if(arg=="--race-config") raceConfigPath=value();
            else if(arg=="--navigation-config") { navigationConfigPath=value();navigationConfigRequired=true; }
            else if(arg=="--calibrate-ground") {calibrateGround=true;noControl=true;}
            else if(arg=="--reverse-check") reverseCheck=true;
            else if(arg=="--parking-preview") {parkingPreview=true;noControl=true;}
            else if(arg=="--parking-drive") parkingDrive=true;
            else if(arg=="--reverse-verified") reverseVerified=true;
            else if(arg=="--obstacle-bypass") bypassEnabled=true;
            else if(arg=="--virtual-boundary-bypass") virtualBypass=true;
            else if(arg=="--mission") missionDrive=true;
            else if(arg=="--mission-lab") {missionDrive=true;missionLab=true;}
            else if(arg=="--replay-corrected") replayCorrected=true;
            else if(arg=="--ground-calibration") groundPath=value();
            else if(arg=="--detections") detectionPath=value();
            else if(arg=="--config") { configPath=value(); configRequired=true; }
            else if(arg=="--replay") { source=value(); replay=noControl=true; }
            else if(arg=="--no-control") noControl=true;
            else if(arg=="--headless") headless=true;
            else if(arg=="--legacy-windows") legacyWindows=true;
            else if(arg=="--trial-seconds") {
                const auto text=value(); size_t used=0; trialSeconds=std::stod(text,&used);
                if(used!=text.size()||!std::isfinite(trialSeconds)||trialSeconds<.5||trialSeconds>3)
                    throw std::invalid_argument("--trial-seconds must be 0.5..3");
            }
            else if(arg=="--display-fps") {
                const auto text=value(); size_t used=0;
                displayFps=std::stoi(text,&used);
                if(used!=text.size()||displayFps<1||displayFps>10)
                    throw std::invalid_argument("--display-fps must be an integer from 1 to 10");
            }
            else if(arg=="--help") {
                std::cout<<"trackview_race [--config trackview.json] [--race-config race.json] [--no-control] [--headless] [--replay video --detections records.jsonl] [--display-fps 1..10] [--legacy-windows]\n"
                    <<"Default MANUAL; press a in image window. ESC/q quits; v toggles AI/binary.\n"
                    <<"Default one compact window at 5 FPS; Ctrl+C in terminal also requests stop. Replay never opens serial/gamepad.\n"
                    <<"--trial-seconds 0.5..3: one-shot low-speed stop timer; requires configured speeds <=0.10 m/s.\n"
                    <<"--navigation-config navigation.json: background pixel route preview in preview/drive modes.\n"
                    <<"--calibrate-ground: camera-only four-point ground calibration, never opens serial/NPU.\n"
                    <<"--parking-preview: camera-only taught bay-2 geometry and in/out route preview; never opens serial/NPU.\n"
                    <<"--reverse-check: one operator-triggered -0.08 m/s, 0.5s pulse; independent stop timer; no AI.\n"
                    <<"--parking-drive --reverse-verified: auto-locate a fully visible left bay; a starts entry/3s dwell/reverse exit/lane rejoin; f manual, g rescan.\n"
                    <<"--obstacle-bypass --ground-calibration ground-calibration.json: calibrated straight-cone trial; uses configured maxSpeed.\n"
                    <<"--virtual-boundary-bypass: matched-image Bezier virtual boundaries with master native 640 tracking; optional --no-control preview; no optical-flow odometry.\n"
                    <<"--mission / --mission-lab: two-lap route or single-loop lab; optional --no-control preview; n confirms physical start/finish.\n"; return 0;
            } else if(arg.rfind("--",0)==0) throw std::invalid_argument("Unknown option: "+arg);
            else source=arg;
        }
        if(!noControl && source.rfind("/dev/video",0)!=0)
            throw std::invalid_argument("Video files require --replay (no vehicle commands)");
        if(replayCorrected&&!replay)throw std::invalid_argument("--replay-corrected is only for offline corrected-frame replay");
        if(virtualBypass&&bypassEnabled)throw std::invalid_argument("Choose one bypass algorithm per run");
        TrackviewConfig config; config.load(configPath,configRequired);
        if(trialSeconds>0) RaceTrialDeadline::requireLowSpeed(config.maxSpeed,config.cruiseSpeed,config.curveSpeed,config.degradedSpeed);
        if(!detectionPath.empty()&&!replay)
            throw std::invalid_argument("Recorded detections require --replay; never use them to drive a car");
        if(replay&&detectionPath.empty())
            throw std::invalid_argument("Race replay requires --detections JSONL");
        RaceConfig raceConfig; raceConfig.load(raceConfigPath);
        RaceNavigationConfig navigationConfig;navigationConfig.load(navigationConfigPath,navigationConfigRequired);
        if(reverseVerified&&!parkingDrive&&!missionDrive)throw std::invalid_argument("--reverse-verified requires --parking-drive");
        if(calibrateGround||reverseCheck||parkingPreview||parkingDrive||missionDrive) {
            if(replay||headless||bypassEnabled||virtualBypass||trialSeconds>0||
               int(calibrateGround)+int(reverseCheck)+int(parkingPreview)+int(parkingDrive)+int(missionDrive)!=1||((reverseCheck||parkingDrive)&&noControl))
                throw std::invalid_argument("Calibration/parking preview/reverse check require separate live-camera window modes");
            if((parkingDrive||(missionDrive&&!missionLab&&!noControl))&&!reverseVerified)throw std::invalid_argument("Parking drive needs physically observed reverse verification");
            std::optional<RaceGroundCalibration> parkingGround;
            if(parkingPreview||parkingDrive||missionDrive){parkingGround.emplace();if(!missionLab)parkingGround->load(groundPath);}
            auto preprocessing=config.binary<0?std::make_shared<Predeal>():std::make_shared<Predeal>(config.binary);
            preprocessing->blackRoad=config.roadSurface=="black";
            if(parkingGround&&!missionLab)preprocessing->setGroundCalibration(*parkingGround);
            cv::VideoCapture video;video.open(source,cv::CAP_V4L2);if(!video.isOpened())video.open(source);
            if(!video.isOpened())throw std::runtime_error("Cannot open calibration camera");
            video.set(cv::CAP_PROP_FRAME_WIDTH,640);video.set(cv::CAP_PROP_FRAME_HEIGHT,480);video.set(cv::CAP_PROP_BUFFERSIZE,1);
            if(reverseCheck)return raceCheckReverse(video,*preprocessing,stopping);
            if(parkingPreview)return racePreviewParking(video,*preprocessing,*parkingGround,stopping);
            if(parkingDrive||missionDrive)return raceDriveParking(video,*preprocessing,*parkingGround,config,raceConfig,reverseVerified,stopping,missionDrive,missionLab,noControl);
            return raceCalibrateGround(video,*preprocessing,groundPath,stopping);
        }
        std::optional<RaceGroundCalibration> ground;
        if(bypassEnabled) {
            ground.emplace();ground->load(groundPath);
            if(config.minSpeed>.1f)throw std::invalid_argument("Bypass trial requires minSpeed <=0.10");
        }
        const bool navigationEnabled=bypassEnabled||virtualBypass||navigationConfig.obstacleMode=="preview";
        RaceObstaclePreview obstaclePreview(navigationConfig);
        RaceVirtualBoundary virtualBoundary(navigationConfig);
        RaceVirtualSteeringFeedback virtualFeedback;
        std::optional<RaceGroundObstacle> groundObstacle;if(ground)groundObstacle.emplace(*ground);
        RaceBypassControl bypassControl;RaceTaskPilot bypassPilot(config);
        if(std::min({raceConfig.limitedSpeed,raceConfig.forkSpeed,raceConfig.approachSpeed})<config.minSpeed)
            throw std::invalid_argument("Race speed caps must be at least trackview minSpeed");
        RaceManager raceManager(raceConfig);
        std::unique_ptr<RaceReplay> detectionReplay;
        std::unique_ptr<RacePerception> perception;
        if(replay) detectionReplay=std::make_unique<RaceReplay>(detectionPath);
        else perception=createRacePerception(raceConfig);
        pilot=RaceTaskPilot(config);
        std::cout<<"[Lane] MASTER_NATIVE_640 | calibrated ground projection; single-edge reconstruction in master Pilot\n";
        log=std::make_unique<Diagnostics>(config);
        std::unique_ptr<RaceTrace> forkTrace;
        if(noControl&&!log->outputDirectory().empty())
            forkTrace=std::make_unique<RaceTrace>(log->outputDirectory(),raceConfig);
        std::ofstream raceLog,detectionLog,navigationLog,bypassLog,virtualLog;
        if(!log->outputDirectory().empty()) {
            raceLog.open(log->outputDirectory()/"race.csv");
            detectionLog.open(log->outputDirectory()/"detections.jsonl");
            std::ofstream(log->outputDirectory()/"race-config.json") << nlohmann::json(raceConfig).dump(2);
            if(navigationEnabled) {
                if(replay||virtualBypass)navigationLog.open(log->outputDirectory()/"navigation.jsonl");
                std::ofstream(log->outputDirectory()/"navigation-config.json")<<nlohmann::json(navigationConfig).dump(2);
            }
            if(ground) {
                std::ofstream(log->outputDirectory()/"ground-calibration.json")<<nlohmann::json(*ground).dump(2);
                bypassLog.open(log->outputDirectory()/"bypass-control.csv");
                bypassLog<<"frame,navigation_frame,navigation_age_s,decision,reason,follow,cone_authorized,race_state,stop,command_speed,pwm,route_center,motion_valid,motion_epoch,motion_inliers,rear_passed,steering_source,lookahead_x_m,lookahead_z_m,requested_pwm,wheelbase_m\n";
            }
            if(virtualBypass) {
                virtualLog.open(log->outputDirectory()/"virtual-boundary-control.jsonl");
                std::ofstream(log->outputDirectory()/"bypass-mode.json")<<nlohmann::json({{"algorithm","virtual_boundary_master640"},
                    {"no_control",noControl},{"optical_flow_required",false},{"vehicle_width_fraction",.18/.45},
                    {"clearance_fraction",.005/.45},{"near_row",410},{"visual_hold_command_distance_m",.36},
                    {"visual_return_command_distance_m",.12},{"command_distance_is_measured",false},{"rear_clearance_verified",false}}).dump(2);
            }
            raceLog << "frame,state,reason,ai_sequence,ai_age_s,fresh,stop,cap,branch_selected,command_speed,geometry_split_rows,red_cap_components,geometry_reason\n";
        }
        std::unique_ptr<RaceNavigationWorker> navigationWorker;
        if(navigationEnabled&&!replay&&!virtualBypass)navigationWorker=std::make_unique<RaceNavigationWorker>(navigationConfig,raceConfig,
            config.binary,config.rowCutUp,config.rowCutBottom,
            log->outputDirectory().empty()?fs::path{}:log->outputDirectory()/"navigation.jsonl",RaceNavigationWorker::Process{},ground,config.roadSurface=="black");
        RaceSampleWriter sampleWriter;
        RaceForkEvidence forkEvidence;
        uint64_t raceFrame=0,loggedSequence=0;
        std::string previousRaceState;
        // 先加载相机标定参数，再打开车辆控制设备。
        auto predeal=config.binary<0 ? std::make_shared<Predeal>() : std::make_shared<Predeal>(config.binary);
        predeal->blackRoad=config.roadSurface=="black";
        if(fs::is_regular_file(groundPath)){RaceGroundCalibration nativeGround;nativeGround.load(groundPath);predeal->setGroundCalibration(nativeGround);}
        std::cout<<"[Road] surface="<<config.roadSurface<<"; normalized road=WHITE, background=BLACK\n";
        if(predeal->blackRoad)std::cout<<"[Road] bounded interior glare/paint repair ON; border-connected exclusions retained\n";
        cv::VideoCapture capture;
        if(!replay) capture.open(source,cv::CAP_V4L2);
        if(!capture.isOpened()) capture.open(source);
        if(!capture.isOpened()) throw std::runtime_error("Cannot open video source: "+source);
        capture.set(cv::CAP_PROP_FRAME_WIDTH,640); capture.set(cv::CAP_PROP_FRAME_HEIGHT,480);
        capture.set(cv::CAP_PROP_BUFFERSIZE,1); // 摄像头后端可能忽略此缓冲区设置。
        std::atomic_bool joyLost{false}; // 生命周期必须覆盖手柄回调的整个使用过程。
        std::shared_ptr<JoyStick> js;
        if(!noControl) {
            car=CarControl::create();
            if(!car) throw std::runtime_error("Cannot initialize vehicle serial port");
            car->setCommandTimeout(config.commandTimeoutMs);
            js=JoyStick::create([&joyLost] { joyLost=true; });
            if(!js) std::cerr<<"[Warn] No gamepad; use image window keyboard.\n";
        }
        Track track; track.rowCutUp=config.rowCutUp; track.rowCutBottom=config.rowCutBottom;
        Mode mode=replay?Mode::Auto:Mode::Manual;
        bool panic=false;
        float cruise=config.cruiseSpeed, jsSpeed=0, speed=0;
        uint16_t jsServo=PWMSERVOMID;
        auto start=Clock::now(), last=start;
        const double videoFps=capture.get(cv::CAP_PROP_FPS);
        const float replayDt=(videoFps>=5 && videoFps<=120)?float(1/videoFps):config.referenceDt;
        TrackviewDisplay display;
        RacePreview aiPreview;
        std::ofstream displayTiming;
        if(!log->displayTimingPath().empty()) {
            displayTiming.open(log->displayTimingPath());
            displayTiming << "render,display_ms,key,render_ms,wait_key_ms,raw_key,time_s\n";
        }
        std::cout<<"trackview_race | "<<(noControl?"NO CONTROL":"MANUAL START")
            <<" | default limit="<<cruise<<" m/s | max="<<config.maxSpeed
            <<" | turn="<<pilot.curveSpeed()<<" quality="<<pilot.degradedSpeed()<<"\n";
        if(forkTrace) std::cout<<"[Trace] AUTO FORK CAPTURE | NO CONTROL ONLY | up to 12 episodes\n";
        if(bypassEnabled)std::cout<<"[Bypass] CALIBRATED STRAIGHT CONE TRIAL | configured max="<<config.maxSpeed<<" m/s | loss of visual motion latches stop | no timed blind pass\n";
        else if(virtualBypass)std::cout<<"[Bypass] VIRTUAL BOUNDARY + MASTER640 | matched current AI image; no optical flow | visual exit heuristic, rear clearance unverified\n";
        else if(navigationEnabled)std::cout<<"[Navigation] BACKGROUND ROUTE SHADOW | PIXELS | does not command wheels | physical clearance not calibrated\n";
        if(!bypassEnabled&&!virtualBypass&&!noControl)std::cout<<"[Mode] 普通巡线与路障停车，不执行自动绕行。绕行入口：bash ./run_car.sh bypass competition\n";
        std::atomic_bool trialStop{false};
        std::atomic_int trialSerialStop{-1};
        RaceTrialDeadline trial(trialSeconds,[&] {
            trialStop=true; stopping=true;
            // 先锁定零速，再打印和等待诊断结束；不依赖 GUI 线程返回。
            if(car) {
                try { car->latchStop(); trialSerialStop=1; }
                catch(...) { trialSerialStop=0; std::cerr<<"[Trial] Serial stop failed; use physical stop."<<std::endl; }
            }
            std::cout<<"[Trial] Timed stop requested; restart program for another trial."<<std::endl;
        });
        if(trial.enabled()) std::cout<<"[Trial] ONE SHOT | <=0.10 m/s | "<<trialSeconds
            <<"s after AUTO; manual throttle disabled."<<std::endl;
        auto ms=[](auto end,auto begin) { return std::chrono::duration<double,std::milli>(end-begin).count(); };
        std::jthread driving([&] {
        std::string stage="startup";
        auto stageStarted=Clock::now();
        double captureMs=0, preprocessMs=0, trackMs=0;
        uint16_t lastBypassServo=PWMSERVOMID;
        uint64_t navigationSequence=0,loggedNavigationSequence=0;
        cv::Mat navigationBinary;
        Track navigationTrack;navigationTrack.rowCutUp=config.rowCutUp;navigationTrack.rowCutBottom=config.rowCutBottom;
        std::string previousNavigationState;
        double lastNavigationPrint=-1e30;
        int automaticBypassSamples=0;
        double lastAutomaticBypassSample=-1e30;
        try {
        while(!stopping.load()) {
            stage="before_capture"; stageStarted=Clock::now();
            if(car && car->hasFault()) throw std::runtime_error("Serial fault or command watchdog timeout");
            cv::Mat image;
            stage="capture"; stageStarted=Clock::now();
            if(!capture.read(image) || image.empty()) {
                if(car) car->resetVelocity();
                log->event(replay?"replay finished":"camera frame unavailable",pilot,lastOverlay);
                if(!replay) throw std::runtime_error("Camera frame unavailable");
                break;
            }
            const auto captured=Clock::now(); captureMs=ms(captured,stageStarted);
            stage="preprocess"; stageStarted=captured;
            // Native 640x480 capture and master Camera correction; no decision-image resizing.
            if(replay)detectionReplay->setReferenceWidth(image.cols);
            if(image.size()!=cv::Size(IMAGE_COL,IMAGE_ROW))
                raceNormalizeFrame(image,replay);
            if(!replayCorrected)predeal->correct(image);
            cv::Mat bin=predeal->binarize(image);
            ++raceFrame;
            if(perception) perception->submit(image,raceFrame,
                std::chrono::duration<double>(captured.time_since_epoch()).count());
            const auto preprocessed=Clock::now(); preprocessMs=ms(preprocessed,stageStarted);
            stage="track"; stageStarted=preprocessed;
            track.handle(bin);
            const auto now=Clock::now(); trackMs=ms(now,stageStarted);
            stage="control"; stageStarted=now;
            float dt=replay?replayDt:float(std::chrono::duration<double>(now-last).count()); last=now;
            if(dt<=0) dt=config.referenceDt;
            log->add(image,bin,track);
            const double observationTime=replay?raceFrame*double(replayDt):raceTime();
            auto observation=perception?perception->snapshot():detectionReplay->take(raceFrame,observationTime);
            if(replay&&observation.sequence==raceFrame) {
                observation.image=image.clone();
                if(raceConfig.verifyCrosswalkGeometry) observation.crosswalk=detectCrosswalkGeometry(observation.image);
            }
            RaceObstaclePreviewPlan navigation;
            RaceSnapshot displayObservation=observation;
            if(navigationEnabled) {
                if(navigationWorker) {
                    if(observation.failed) {
                        navigation.sequence=observation.sequence;navigation.captured=observation.captured;
                        navigation.state="OBS_AI_FAILED";navigation.reason=observation.error;
                    } else {
                        navigationWorker->submit(observation,observationTime);
                        auto completed=navigationWorker->view(observationTime);navigation=std::move(completed.plan);
                        if(completed.ai.sequence)displayObservation=std::move(completed.ai);
                    }
                } else {
                    // 回放按虚拟帧时间同步处理，保留可重复的离线验证。
                    if(observation.sequence!=navigationSequence&&!observation.image.empty()) {
                        navigationBinary=predeal->binarize(observation.image);
                        navigationTrack.handle(navigationBinary);navigationSequence=observation.sequence;
                    }
                    navigation=virtualBypass?virtualBoundary.update(observation,observationTime,navigationBinary,navigationTrack,raceConfig,mode==Mode::Auto,speed):
                        groundObstacle?groundObstacle->update(observation,observationTime,navigationBinary,navigationTrack,raceConfig,bypassControl.engaged(),mode==Mode::Auto):
                        obstaclePreview.update(observation,observationTime,navigationBinary,navigationTrack,raceConfig);
                }
                // 检测抖动时限制终端输出频率；navigation.jsonl 仍保留每次新推理结果。
                // 停车原因（尤其路宽/尺度）必须和状态一起显示，否则只看到"要直道"却不知为何。
                const std::string navigationKey=navigation.state+" | "+navigation.reason;
                if(previousNavigationState!=navigationKey&&observationTime-lastNavigationPrint>=1.) {
                    std::cout<<"[Navigation] "<<navigation.state<<" | "<<navigation.reason
                        <<(virtualBypass?" | VIRTUAL BOUNDARY TRIAL":bypassEnabled?" | CALIBRATED TRIAL":" | PREVIEW ONLY / PIXELS");
                    if(navigation.metric&&navigation.laneWidthRaw>0) {
                        std::cout<<" | raw lane "<<std::lround(navigation.laneWidthRaw*100)<<"cm";
                        // 未锁定时打印"需要的倍数"，这样越界一行就能看出差多少，不必再猜。
                        if(navigation.laneScaleWanted>navigation.laneScale*1.001f)
                            std::cout<<" need x"<<std::fixed<<std::setprecision(2)<<navigation.laneScaleWanted<<std::setprecision(6);
                        else
                            std::cout<<" x"<<std::fixed<<std::setprecision(2)<<navigation.laneScale<<std::setprecision(6);
                    }
                    std::cout<<'\n';
                    previousNavigationState=navigationKey;
                    lastNavigationPrint=observationTime;
                }
                if(navigationLog&&observation.sequence>loggedNavigationSequence) {
                    navigationLog<<raceObstacleJson(navigation,navigationConfig).dump()<<'\n';
                    loggedNavigationSequence=observation.sequence;
                }
            }
            if(!headless) aiPreview.publish(displayObservation,observationTime,navigation);
            const uint64_t motionEpoch=car?car->motionEpoch():0;
            auto toggle=[&]() {
                if(virtualBypass&&mode==Mode::Auto)return; // a=start; SPACE is the explicit stop.
                if(mode==Mode::Auto) { mode=Mode::Manual; jsSpeed=0; pilot.resetControl(); }
                else if(bypassEnabled&&!raceBypassArmReady(navigation,observationTime,raceConfig)) {
                    std::cout<<"[Wait] 绕行启动需要已确认锥桶和鲜活可执行路线；当前 "<<navigation.state<<" | "<<navigation.reason<<'\n';
                }
                else if(virtualBypass&&navigation.hasTarget&&(!navigation.fresh||navigation.stop||!navigation.candidate)) {
                    std::cout<<"[Wait] Virtual boundary "<<navigation.state<<" | "<<navigation.reason<<'\n';
                }
                else if((pilot.ready||((bypassEnabled||virtualBypass)&&bypassPilot.ready)) && raceManager.aiReady(observation,observationTime) && !joyLost && (!js || !js->isStopped()) && std::fabs(jsSpeed)<=0.05f) {
                    if(car && !car->resumeMotion(motionEpoch)) return;
                    mode=Mode::Auto; panic=false; jsSpeed=0; pilot.resetControl();
                } else {
                    if(!raceManager.aiReady(observation,observationTime)) std::cout<<"[Wait] Fresh AI result required.\n";
                    else if(!pilot.ready) std::cout<<"[Wait] Lane "<<pilot.pathIssue
                        <<"; width rows="<<pilot.learnedWidthRows()
                        <<" missing width="<<pilot.missingWidthPoints
                        <<"; wait for stable trusted path.\n";
                    else std::cout<<"[Wait] Release gamepad throttle or reconnect gamepad.\n";
                }
            };
            std::string stopReason;
            for(const int key:display.takeKeys()) {
                if(key==' ') {
                    panic=true; mode=Mode::Manual; jsSpeed=0; pilot.resetControl();
                    if(car) car->pauseMotion();
                    stopReason="keyboard stop";
                } else if(key=='a' || key=='A') toggle();
                else if(virtualBypass&&(key=='g'||key=='G')&&mode==Mode::Manual) {
                    virtualBoundary.reset();virtualFeedback.reset();bypassPilot=RaceTaskPilot(config);navigation={};lastBypassServo=PWMSERVOMID;
                    std::cout<<"[Bypass] Reset at current pose; wait for a fresh route, then press a.\n";
                }
                else if(key=='+' || key=='=') cruise=std::min(config.maxSpeed,cruise+0.05f);
                else if(key=='-' || key=='_') cruise=std::max(config.minSpeed,cruise-0.05f);
                else if(key=='[' || key==']') {
                    pilot.setCurveSpeed(pilot.curveSpeed()+(key==']'?0.05f:-0.05f));
                    std::cout<<"[Set] curveSpeed="<<pilot.curveSpeed()<<" m/s\n";
                } else if(key==',' || key=='.') {
                    pilot.setDegradedSpeed(pilot.degradedSpeed()+(key=='.'?0.05f:-0.05f));
                    std::cout<<"[Set] degradedSpeed="<<pilot.degradedSpeed()<<" m/s\n";
                }
                else if(key=='w' || key=='W') pilot.cycleWeight();
            }
            // X11 远程绘图仅用于显示状态，不作为控制循环的心跳。
            // 指令超时和视觉丢线判断独立于远程界面的显示延迟。
            const auto displayAge=headless ? 0LL : display.ageMs();
            if(js) {
                float s=0; uint16_t sv=PWMSERVOMID;
                const bool newCommand=js->takeCarControl(s,sv);
                if(newCommand) { jsSpeed=s; jsServo=sv; }
                if(js->takeSampleOnce()) toggle();
                js->takeBuzzer();
                if(js->isStopped() || joyLost) {
                    if(!panic) stopReason="gamepad disconnected; stop latched";
                    panic=true; mode=Mode::Manual; jsSpeed=0; pilot.resetControl();
                }
                else if(mode==Mode::Auto && newCommand && std::fabs(jsSpeed)>0.05f) {
                    mode=Mode::Manual; pilot.resetControl();
                }
            }
            if(mode==Mode::Auto&&trial.arm()) std::cout<<"[Trial] Timer armed."<<std::endl;
            const auto bypass=virtualBypass?virtualBoundary.decision(navigation,mode==Mode::Auto):
                bypassEnabled?bypassControl.update(navigation,observation,raceConfig,observationTime,mode==Mode::Auto):RaceBypassDecision{};
            if(navigationWorker&&bypassEnabled)navigationWorker->setControlState(mode==Mode::Auto,bypassControl.engaged());
            auto racePlan=raceManager.update(observation,observationTime,dt,mode==Mode::Auto,bin,track,image,bypass.authorizeCone);
            const bool laneTask=std::string(racePlan.state)=="LANE"||std::string(racePlan.state)=="LIMIT_ZONE";
            if(bypassEnabled||virtualBypass) {
                std::vector<PointX> left,right;
                if((navigation.metric||virtualBypass)&&navigation.candidate&&!navigation.stop&&navigation.fresh) {
                    for(const auto &p:navigation.left)left.emplace_back(p.x,int(std::lround(p.y)));
                    for(const auto &p:navigation.right)right.emplace_back(p.x,int(std::lround(p.y)));
                }
                bypassPilot.update(left,right,dt,speed);
                if(bypass.involved) {
                    if(bypass.stop||!laneTask||!bypassPilot.ready)racePlan.stop=true;
                    if(laneTask) {racePlan.state=bypass.state;racePlan.reason=bypass.follow&&!bypassPilot.ready?"BYPASS_CONTROL_PATH_WAIT":bypass.reason;}
                }
            }
            if(forkTrace) forkTrace->observe(raceFrame,observationTime,image,bin,track,observation,racePlan);
            const auto &pathLeft=racePlan.selectedBranch?racePlan.path.left:track.pointsEdgeLeft;
            const auto &pathRight=racePlan.selectedBranch?racePlan.path.right:track.pointsEdgeRight;
            pilot.update(pathLeft,pathRight,dt,speed);
            const bool captureFork=!log->outputDirectory().empty()&&
                forkEvidence.due(racePlan.state,racePlan.reason,observationTime,observation.sequence);
            // Retain the unannotated control-frame image separately from the asynchronous AI image.
            cv::Mat forkCamera;if(captureFork)forkCamera=image.clone();
            track.drawImage(image);
            if(mode==Mode::Auto && pilot.lost && !raceDirectionSeeking(racePlan) && !(bypassEnabled&&bypassControl.engaged())&&!(virtualBypass&&bypass.involved)) {
                mode=Mode::Manual; panic=true; jsSpeed=0; pilot.resetControl();
                stopReason=std::string("trusted lane lost; stop latched, press a after recovery; path=")+pilot.pathIssue;
            }
            const float previousCommandSpeed=speed;
            speed=0; uint16_t servo=PWMSERVOMID;
            RaceBypassSteering routeSteering;
            if(!panic && !stopping) {
                if(mode==Mode::Auto) {
                    if(racePlan.stop) {pilot.resetControl();bypassPilot.resetControl();}
                    else if(bypass.follow) {
                        if(virtualBypass) {
                            speed=bypassPilot.speedFor(std::min(cruise,racePlan.speedCap),dt);
                            servo=virtualFeedback.update(navigation,bypassPilot.error,bypassPilot.servo,lastBypassServo,dt,previousCommandSpeed);
                        } else {
                        routeSteering=raceGroundSteering(navigation,navigationConfig,previousCommandSpeed,lastBypassServo,dt);
                        if(routeSteering.valid) {
                            speed=bypassPilot.speedFor(std::min(cruise,racePlan.speedCap),dt);servo=routeSteering.pwm;
                        } else {racePlan.stop=true;racePlan.reason="BYPASS_STEERING_PATH_WAIT";bypassPilot.resetControl();}
                        }
                    }
                    else if(raceDirectionSeeking(racePlan)) {
                        speed=raceDirectionSpeed(racePlan,pilot,cruise,dt,previousCommandSpeed);
                        servo=raceDirectionServo(racePlan,PWMSERVOMID);
                    }
                    else {speed=pilot.speedFor(std::min(cruise,racePlan.speedCap),dt);servo=raceDirectionServo(racePlan,pilot.servo);}
                }
                else if(!trial.enabled()&&!bypassEnabled&&!virtualBypass) { speed=std::clamp(jsSpeed,-config.maxSpeed,config.maxSpeed); servo=jsServo; }
            }
            if(!virtualBypass||!bypass.follow||speed==0)virtualFeedback.reset();
            if(virtualBypass&&bypass.involved&&speed==0)servo=lastBypassServo;
            lastBypassServo=(virtualBypass&&bypass.involved)||(bypass.follow&&speed>0)?servo:PWMSERVOMID;
            if(car) {
                if(car->hasFault()) throw std::runtime_error("Serial command timeout during processing");
                car->carControl(speed,servo);
                if(racePlan.buzzer&&!panic&&mode==Mode::Auto) car->buzzerSound(Buzzer::ding);
            }
            // 在实际零速指令之后异步保存首次故障画面，避免人工按s时只留下十几秒后的静止图。
            if((bypassEnabled||virtualBypass)&&bypass.stop&&(virtualBypass?bypass.involved:bypassControl.engaged())&&previousCommandSpeed>.001f&&
               automaticBypassSamples<8&&observationTime-lastAutomaticBypassSample>=2&&
               displayObservation.sequence==navigation.sequence&&!log->outputDirectory().empty()) {
                RaceSampleWriter::Job job{log->outputDirectory()/"bypass-stop-evidence",displayObservation,observationTime,
                    raceConfig,automaticBypassSamples+1,true,navigation,navigationConfig};
                if(sampleWriter.submit(std::move(job))) {
                    ++automaticBypassSamples;lastAutomaticBypassSample=observationTime;
                    std::cout<<"[Sample] Queued bypass-stop-evidence/ai-sample-"<<automaticBypassSamples<<'\n';
                }
            }
            const auto controlled=Clock::now();
            if(captureFork) {
                RaceSampleWriter::Job job{log->outputDirectory()/"fork-evidence",observation,observationTime,
                    raceConfig,forkEvidence.nextIndex(),false,{},navigationConfig};
                job.context={{"race_frame",raceFrame},{"ai_frame",observation.sequence},
                    {"ai_captured_s",observation.captured},{"control_time_s",observationTime},
                    {"state",racePlan.state},{"reason",racePlan.reason},{"selected_branch",racePlan.selectedBranch},
                    {"guide_direction",racePlan.guideDirection},{"guide_pwm",racePlan.guidePwm},
                    {"geometry_split_rows",racePlan.geometrySplitRows},{"geometry_reason",racePlan.geometryReason},
                    {"road_surface",config.roadSurface},{"speed_command",speed},{"pwm",servo},
                    {"lane_image_binding","current control frame; camera.png is asynchronous AI frame"}};
                job.laneImage=std::move(forkCamera);job.laneBinary=bin.clone();
                if(sampleWriter.submit(std::move(job)))
                    forkEvidence.queued(racePlan.state,racePlan.reason,observationTime,observation.sequence);
            }
            if(previousRaceState!=racePlan.state) {
                std::cout<<"[Race] "<<racePlan.state<<" | "<<racePlan.reason<<'\n';
                previousRaceState=racePlan.state;
            }
            if(raceLog) raceLog<<raceFrame<<','<<racePlan.state<<','<<racePlan.reason<<','
                <<observation.sequence<<','<<racePlan.detectionAge<<','<<racePlan.fresh<<','
                <<racePlan.stop<<','<<racePlan.speedCap<<','<<racePlan.selectedBranch<<','<<speed<<','
                <<racePlan.geometrySplitRows<<','<<racePlan.colorRejectedComponents<<','<<racePlan.geometryReason<<'\n';
            if(bypassLog) bypassLog<<raceFrame<<','<<navigation.sequence<<','<<observationTime-navigation.captured<<','
                <<bypass.state<<','<<bypass.reason<<','<<bypass.follow<<','<<bypass.authorizeCone<<','<<racePlan.state<<','
                <<racePlan.stop<<','<<speed<<','<<servo<<','<<bypassPilot.center<<','<<navigation.motionValid<<','
                <<navigation.motionEpoch<<','<<navigation.motionInliers<<','<<navigation.passed<<','
                <<(bypass.follow?"GROUND_LOOKAHEAD":"LANE_OR_STOP")<<','<<routeSteering.targetX<<','<<routeSteering.targetZ<<','
                <<routeSteering.requested<<','<<navigationConfig.bypassWheelbase<<'\n';
            if(virtualLog)virtualLog<<nlohmann::json({{"frame",raceFrame},{"ai_frame",observation.sequence},
                {"ai_age_s",observationTime-observation.captured},{"state",bypass.state},{"reason",bypass.reason},
                {"race_state",racePlan.state},{"race_reason",racePlan.reason},{"stop",racePlan.stop},
                {"follow",bypass.follow},{"cone_authorized",bypass.authorizeCone},{"command_speed",speed},{"pwm",servo},
                {"virtual_center",bypassPilot.center},{"virtual_path_ready",bypassPilot.ready},
                {"virtual_error_px",bypassPilot.error},{"virtual_base_pwm",bypassPilot.servo},{"virtual_correction_pwm",virtualFeedback.correction()},
                {"virtual_path_issue",bypassPilot.pathIssue},{"algorithm","virtual_boundary_master640"},
                {"visual_returned",navigation.virtualVisualReturn},{"rear_passed",false},{"optical_flow_required",false}}).dump()<<'\n';
            if(detectionLog&&observation.sequence>loggedSequence) {
                nlohmann::json entries=nlohmann::json::array();
                for(const auto &d:observation.detections) {
                    const auto review=reviewRaceDetection(d,observation.image,raceConfig,&observation.crosswalk);
                    entries.push_back({{"type",d.type},{"score",d.score},{"x",d.x},{"y",d.y},
                        {"width",d.width},{"height",d.height},{"accepted",review.accepted},
                        {"checked_stop",review.checkedStop},{"checked_cross",review.checkedCross},{"acceptance_reason",review.reason}});
                }
                detectionLog<<nlohmann::json({{"frame",observation.sequence},{"captured",observation.captured},
                    {"detections",entries},{"crosswalk_geometry",raceCrosswalkJson(observation,raceConfig)},
                    {"direction_checks",raceDirectionJson(observation)}}).dump()<<'\n';
                loggedSequence=observation.sequence;
            }
            if(raceFrame%15==0) { raceLog.flush(); detectionLog.flush();navigationLog.flush();bypassLog.flush();virtualLog.flush(); }
            const double controlMs=ms(controlled,stageStarted);
            stage="record_publish"; stageStarted=controlled;
            drawHud(image,pilot,mode,panic,cruise,speed,servo,1/dt,displayAge>=2000,racePlan);
            lastOverlay=image;
            log->record(std::chrono::duration<double>(now-start).count(),dt,mode,panic,pilot,cruise,speed,servo,displayAge);
            if(!stopReason.empty()) log->event(stopReason,pilot,image);
            if(!headless&&display.frameDue()) display.publish(image,bin);
            log->recordTiming(captureMs,preprocessMs,trackMs,controlMs,ms(Clock::now(),controlled));
        }
        if(car) car->resetVelocity();
        log->event(trialStop.load()?"timed low-speed trial stop":stopping.load()?"signal or UI stop":"normal exit",pilot,lastOverlay);
        } catch(const std::exception &e) {
            if(car) { try { car->latchStop(); } catch(...) {} }
            const std::string detail=std::string(e.what())+"; stage="+stage+
                " stage_ms="+std::to_string(ms(Clock::now(),stageStarted))+
                " capture_ms="+std::to_string(captureMs)+
                " preprocess_ms="+std::to_string(preprocessMs)+
                " track_ms="+std::to_string(trackMs);
            if(stopping.load() && !display.failed.load()) {
                log->event(trialStop.load()?"timed low-speed trial stop":"operator requested exit",pilot,lastOverlay);
            } else {
                log->event(detail,pilot,lastOverlay);
                std::cerr<<"[Error] "<<detail<<'\n'; display.failed=true;
            }
        } catch(...) {
            if(car) { try { car->latchStop(); } catch(...) {} }
            log->event("unexpected driving thread failure",pilot,lastOverlay);
            display.failed=true;
        }
        display.finished=true;
        });
        StopOnExit stopBeforeJoin{car};
        int render=0;
        int savedAiSamples=0;
        RaceSnapshot shownAi;
        RaceObstaclePreviewPlan shownNavigation;
        double shownAiTime=0;
        const auto displayInterval=std::chrono::milliseconds(1000/displayFps);
        auto lastRender=Clock::now()-displayInterval;
        bool showBinary=false;
        int flushedRender=-1;
        while(!stopping.load() && !display.finished.load()) {
            // 图形界面由主线程处理，线程之间只传递最新的只读图像快照。
            if(car && car->hasFault() && !stopping.load()) { display.failed=true; break; }
            if(headless) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                continue;
            }
            const auto rendered=Clock::now();
            cv::Mat image,bin;
            if(rendered-lastRender>=displayInterval) {
                if(!display.snapshot(image,bin)) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(5));
                    display.progress(); continue;
                }
                auto snapshot=aiPreview.view();
                shownAi=std::move(snapshot.ai);shownNavigation=std::move(snapshot.navigation);
                shownAiTime=replay?snapshot.time:raceTime();
                if(legacyWindows) {
                    cv::imshow("track_view",image); cv::imshow("binarized",bin);
                    cv::imshow("AI detections",racePreviewImage(shownAi,shownAiTime,raceConfig));
                } else cv::imshow("track_view",raceCompactView(image,bin,shownAi,shownAiTime,raceConfig,showBinary,
                    navigationEnabled?&shownNavigation:nullptr));
                lastRender=Clock::now(); ++render;
            }
            const auto waitingKey=Clock::now();
            const int rawKey=cv::waitKeyEx(10);
            const int key=raceUiKey(rawKey);
            const auto receivedKey=Clock::now();
            const bool quit=raceUiQuit(key);
            // 收到退出/停车键后，先禁止运动，再写诊断和等待图像保存。
            if(quit) {
                stopping=true;
                if(car) { try { car->latchStop(); } catch(...) {} }
            } else if(key==' '&&car) car->pauseMotion();
            display.progress();
            if(displayTiming && render>0) {
                displayTiming << render << ',' << std::chrono::duration<double,std::milli>(Clock::now()-rendered).count()
                    << ',' << key << ',' << ms(waitingKey,rendered) << ',' << ms(receivedKey,waitingKey)
                    << ',' << rawKey << ',' << std::chrono::duration<double>(receivedKey-start).count() << '\n';
                if(key>=0||render!=flushedRender) { displayTiming.flush(); flushedRender=render; }
            }
            if(quit) {
                std::cout<<"[Exit] Quit key received; control stop requested, saving diagnostics..."<<std::endl;
                break;
            }
            if(key=='v'||key=='V') { showBinary=!showBinary; lastRender=Clock::now()-displayInterval; continue; }
            if(key=='s'||key=='S') {
                const int index=savedAiSamples+1;
                const double sampleTime=replay?shownAiTime:raceTime();
                RaceSampleWriter::Job job{log->outputDirectory(),shownAi,sampleTime,raceConfig,index,
                    navigationEnabled,shownNavigation,navigationConfig};
                if(sampleWriter.submit(std::move(job))) {
                    savedAiSamples=index;std::cout<<"[Sample] Queued ai-sample-"<<index<<'\n';
                } else std::cout<<"[Sample] Wait for matched AI image or available save queue.\n";
                continue;
            }
            display.post(key);
        }
        stopping=true;
        if(car) { try { car->latchStop(); } catch(...) {} }
        driving.join();
        if(navigationWorker)navigationWorker->finish();
        sampleWriter.finish();
        std::cout<<"[Sample] saved="<<sampleWriter.savedCount()<<" failed="<<sampleWriter.failedCount()
            <<" dropped="<<sampleWriter.droppedCount()<<'\n';
        trial.finish();
        if(trial.enabled()&&!log->outputDirectory().empty()) {
            std::ofstream(log->outputDirectory()/"trial-stop.json")<<nlohmann::json({
                {"limit_s",trialSeconds},{"armed",trial.armed()},{"expired",trial.expired()},
                {"no_control",noControl},{"serial_zero_request_result",trialSerialStop.load()},
                {"serial_result_meaning","-1:not attempted; 0:failed; 1:API completed, not measured wheel stop"}}).dump(2);
        }
        if(forkTrace) {
            forkTrace->finish();
            std::cout<<"[Trace] saved="<<forkTrace->savedCount()<<" failed="<<forkTrace->failedCount()
                <<" dropped="<<forkTrace->droppedCount()<<'\n';
        }
        std::cout<<"[Stop] trackview_race finished.\n";
        return display.failed.load()?1:0;
    } catch(const std::exception &e) {
        if(car) { try { car->resetVelocity(); } catch(...) {} }
        if(log) log->event(e.what(),pilot,lastOverlay);
        std::cerr<<"[Error] "<<e.what()<<'\n'; return 1;
    } catch(...) {
        if(car) { try { car->resetVelocity(); } catch(...) {} }
        if(log) log->event("unexpected exception",pilot,lastOverlay);
        return 1;
    }
}
