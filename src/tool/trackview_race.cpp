/** 比赛集成入口：继承 v6 巡线，叠加异步感知、选路和任务仲裁。
 * 与 track_view.cpp 分开维护，避免改变已验证的 v6 程序。 */
#include "race_manager.hpp"
#include "race_replay.hpp"
#include "race_preview.hpp"
#include "control.hpp"
#include "joystick.hpp"
#include "predeal.hpp"
#include "track.hpp"
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
    void record(double time,float dt,Mode mode,bool panic,const AutoPilot &p,float cruise,float speed,uint16_t pwm,long long displayAge) {
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
    void event(const std::string &reason,const AutoPilot &p,const cv::Mat &overlay={}) noexcept {
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
            widths << "row,width,observations,pending_observations,age_s,refresh_due\n";
            for(const auto &w:p.widthProfile()) widths << w.row << ',' << w.width << ','
                << w.observations << ',' << w.pendingObservations << ',' << w.age << ',' << w.refreshDue << '\n';
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

static void drawHud(cv::Mat &image,const AutoPilot &pilot,Mode mode,bool panic,
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

int main(int argc,char **argv) {
    std::signal(SIGINT,onSignal); std::signal(SIGTERM,onSignal); std::signal(SIGHUP,onSignal);
    std::shared_ptr<CarControl> car;
    std::unique_ptr<Diagnostics> log;
    AutoPilot pilot;
    cv::Mat lastOverlay;
    try {
        std::string source="/dev/video0", configPath="trackview.json";
        std::string raceConfigPath="race.json", detectionPath;
        bool noControl=false, headless=false, replay=false, configRequired=false;
        for(int i=1;i<argc;++i) {
            const std::string arg=argv[i];
            auto value=[&]() { if(++i>=argc) throw std::invalid_argument("Missing argument for "+arg); return std::string(argv[i]); };
            if(arg=="--race-config") raceConfigPath=value();
            else if(arg=="--detections") detectionPath=value();
            else if(arg=="--config") { configPath=value(); configRequired=true; }
            else if(arg=="--replay") { source=value(); replay=noControl=true; }
            else if(arg=="--no-control") noControl=true;
            else if(arg=="--headless") headless=true;
            else if(arg=="--help") {
                std::cout<<"trackview_race [--config trackview.json] [--race-config race.json] [--no-control] [--headless] [--replay video --detections records.jsonl]\n"
                    <<"Default MANUAL; press a in image window. Replay never opens serial/gamepad.\n"; return 0;
            } else if(arg.rfind("--",0)==0) throw std::invalid_argument("Unknown option: "+arg);
            else source=arg;
        }
        if(!noControl && source.rfind("/dev/video",0)!=0)
            throw std::invalid_argument("Video files require --replay (no vehicle commands)");
        TrackviewConfig config; config.load(configPath,configRequired);
        if(!detectionPath.empty()&&!replay)
            throw std::invalid_argument("Recorded detections require --replay; never use them to drive a car");
        if(replay&&detectionPath.empty())
            throw std::invalid_argument("Race replay requires --detections JSONL");
        RaceConfig raceConfig; raceConfig.load(raceConfigPath);
        if(std::min({raceConfig.limitedSpeed,raceConfig.forkSpeed,raceConfig.approachSpeed})<config.minSpeed)
            throw std::invalid_argument("Race speed caps must be at least trackview minSpeed");
        RaceManager raceManager(raceConfig);
        std::unique_ptr<RaceReplay> detectionReplay;
        std::unique_ptr<RacePerception> perception;
        if(replay) detectionReplay=std::make_unique<RaceReplay>(detectionPath);
        else perception=createRacePerception(raceConfig);
        pilot=AutoPilot(config);
        log=std::make_unique<Diagnostics>(config);
        std::ofstream raceLog,detectionLog;
        if(!log->outputDirectory().empty()) {
            raceLog.open(log->outputDirectory()/"race.csv");
            detectionLog.open(log->outputDirectory()/"detections.jsonl");
            std::ofstream(log->outputDirectory()/"race-config.json") << nlohmann::json(raceConfig).dump(2);
            raceLog << "frame,state,reason,ai_sequence,ai_age_s,fresh,stop,cap,branch_selected,command_speed\n";
        }
        uint64_t raceFrame=0,loggedSequence=0;
        std::string previousRaceState;
        // 先加载相机标定参数，再打开车辆控制设备。
        auto predeal=config.binary<0 ? std::make_shared<Predeal>() : std::make_shared<Predeal>(config.binary);
        cv::VideoCapture capture;
        if(!replay) capture.open(source,cv::CAP_V4L2);
        if(!capture.isOpened()) capture.open(source);
        if(!capture.isOpened()) throw std::runtime_error("Cannot open video source: "+source);
        capture.set(cv::CAP_PROP_FRAME_WIDTH,IMAGE_COL); capture.set(cv::CAP_PROP_FRAME_HEIGHT,IMAGE_ROW);
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
            displayTiming << "render,display_ms,key\n";
        }
        std::cout<<"trackview_race experimental 2026-10-16 | "<<(noControl?"NO CONTROL":"MANUAL START")
            <<" | default limit="<<cruise<<" m/s | max="<<config.maxSpeed
            <<" | turn="<<pilot.curveSpeed()<<" quality="<<pilot.degradedSpeed()<<"\n";
        std::jthread driving([&] {
        std::string stage="startup";
        auto stageStarted=Clock::now();
        double captureMs=0, preprocessMs=0, trackMs=0;
        auto ms=[](auto end,auto begin) { return std::chrono::duration<double,std::milli>(end-begin).count(); };
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
            // 标定参数对应 320×240 图像；分辨率不一致时先缩放，再进行去畸变。
            if(image.size()!=cv::Size(IMAGE_COL,IMAGE_ROW))
                cv::resize(image,image,{IMAGE_COL,IMAGE_ROW},0,0,cv::INTER_AREA);
            predeal->correct(image);
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
            if(replay&&observation.sequence==raceFrame) observation.image=image.clone();
            if(!headless) aiPreview.publish(observation,observationTime);
            const uint64_t motionEpoch=car?car->motionEpoch():0;
            auto toggle=[&]() {
                if(mode==Mode::Auto) { mode=Mode::Manual; jsSpeed=0; pilot.resetControl(); }
                else if(pilot.ready && raceManager.aiReady(observation,observationTime) && !joyLost && (!js || !js->isStopped()) && std::fabs(jsSpeed)<=0.05f) {
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
            const auto racePlan=raceManager.update(observation,observationTime,dt,mode==Mode::Auto,bin,track);
            const auto &pathLeft=racePlan.selectedBranch?racePlan.path.left:track.pointsEdgeLeft;
            const auto &pathRight=racePlan.selectedBranch?racePlan.path.right:track.pointsEdgeRight;
            pilot.update(pathLeft,pathRight,dt,speed);
            track.drawImage(image);
            if(mode==Mode::Auto && pilot.lost) {
                mode=Mode::Manual; panic=true; jsSpeed=0; pilot.resetControl();
                stopReason=std::string("trusted lane lost; stop latched, press a after recovery; path=")+pilot.pathIssue;
            }
            speed=0; uint16_t servo=PWMSERVOMID;
            if(!panic && !stopping) {
                if(mode==Mode::Auto) {
                    if(racePlan.stop) pilot.resetControl();
                    else { speed=pilot.speedFor(std::min(cruise,racePlan.speedCap),dt); servo=pilot.servo; }
                }
                else { speed=std::clamp(jsSpeed,-config.maxSpeed,config.maxSpeed); servo=jsServo; }
            }
            if(car) {
                if(car->hasFault()) throw std::runtime_error("Serial command timeout during processing");
                car->carControl(speed,servo);
                if(racePlan.buzzer&&!panic&&mode==Mode::Auto) car->buzzerSound(Buzzer::ding);
            }
            const auto controlled=Clock::now();
            if(previousRaceState!=racePlan.state) {
                std::cout<<"[Race] "<<racePlan.state<<" | "<<racePlan.reason<<'\n';
                previousRaceState=racePlan.state;
            }
            if(raceLog) raceLog<<raceFrame<<','<<racePlan.state<<','<<racePlan.reason<<','
                <<observation.sequence<<','<<racePlan.detectionAge<<','<<racePlan.fresh<<','
                <<racePlan.stop<<','<<racePlan.speedCap<<','<<racePlan.selectedBranch<<','<<speed<<'\n';
            if(detectionLog&&observation.sequence>loggedSequence) {
                nlohmann::json entries=nlohmann::json::array();
                for(const auto &d:observation.detections) entries.push_back({{"type",d.type},{"score",d.score},
                    {"x",d.x},{"y",d.y},{"width",d.width},{"height",d.height}});
                detectionLog<<nlohmann::json({{"frame",observation.sequence},{"captured",observation.captured},
                    {"detections",entries},{"direction_checks",raceDirectionJson(observation)}}).dump()<<'\n';
                loggedSequence=observation.sequence;
            }
            if(raceFrame%15==0) { raceLog.flush(); detectionLog.flush(); }
            const double controlMs=ms(controlled,stageStarted);
            stage="record_publish"; stageStarted=controlled;
            drawHud(image,pilot,mode,panic,cruise,speed,servo,1/dt,displayAge>=2000,racePlan);
            lastOverlay=image;
            log->record(std::chrono::duration<double>(now-start).count(),dt,mode,panic,pilot,cruise,speed,servo,displayAge);
            if(!stopReason.empty()) log->event(stopReason,pilot,image);
            if(!headless) display.publish(image,bin);
            log->recordTiming(captureMs,preprocessMs,trackMs,controlMs,ms(Clock::now(),controlled));
        }
        if(car) car->resetVelocity();
        log->event(stopping.load()?"signal or UI stop":"normal exit",pilot,lastOverlay);
        } catch(const std::exception &e) {
            if(car) { try { car->latchStop(); } catch(...) {} }
            const std::string detail=std::string(e.what())+"; stage="+stage+
                " stage_ms="+std::to_string(ms(Clock::now(),stageStarted))+
                " capture_ms="+std::to_string(captureMs)+
                " preprocess_ms="+std::to_string(preprocessMs)+
                " track_ms="+std::to_string(trackMs);
            if(stopping.load() && !display.failed.load()) {
                log->event("operator requested exit",pilot,lastOverlay);
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
        double shownAiTime=0;
        auto lastRender=Clock::now()-std::chrono::milliseconds(100);
        while(!stopping.load() && !display.finished.load()) {
            // 图形界面由主线程处理，线程之间只传递最新的只读图像快照。
            if(car && car->hasFault()) { display.failed=true; break; }
            if(headless) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                continue;
            }
            const auto rendered=Clock::now();
            cv::Mat image,bin;
            if(rendered-lastRender>=std::chrono::milliseconds(100)) {
                if(!display.snapshot(image,bin)) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(5));
                    display.progress(); continue;
                }
                cv::imshow("track_view",image); cv::imshow("binarized",bin);
                auto snapshot=aiPreview.snapshot();
                shownAi=std::move(snapshot.first);
                shownAiTime=replay?snapshot.second:raceTime();
                cv::imshow("AI detections",racePreviewImage(shownAi,shownAiTime,raceConfig));
                lastRender=Clock::now(); ++render;
            }
            const int key=cv::waitKey(10);
            display.progress();
            if(displayTiming && render>0) {
                displayTiming << render << ',' << std::chrono::duration<double,std::milli>(Clock::now()-rendered).count()
                    << ',' << key << '\n';
                displayTiming.flush();
            }
            if(key==27) break;
            if(key=='s'||key=='S') {
                if(!noControl) std::cout<<"[Sample] Use --no-control to save AI images.\n";
                else {
                    try {
                        // 保存当前屏幕对应的 AI 帧；采样不占用巡线线程。
                        if(saveRaceSample(log->outputDirectory(),shownAi,shownAiTime,raceConfig,++savedAiSamples))
                            std::cout<<"[Sample] Saved ai-sample-"<<savedAiSamples<<'\n';
                        else std::cout<<"[Sample] Wait for AI image; diagnostics must be enabled.\n";
                    } catch(const std::exception &e) { std::cerr<<"[Sample] "<<e.what()<<'\n'; }
                }
                continue;
            }
            display.post(key);
            if(key==' ' && car) car->pauseMotion(); // 立即禁止运动，防止工作线程竞争写入旧的行驶指令。
        }
        stopping=true;
        if(car) { try { car->latchStop(); } catch(...) {} }
        driving.join();
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
