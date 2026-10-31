#pragma once
#include "race_perception.hpp"
#include "task_track.hpp"
#include "race_appearance.hpp"
#include "race_detection_review.hpp"
#include <algorithm>
#include <array>
#include <limits>
#include <string_view>

struct RacePath {
    std::vector<PointX> left,right;
    int splitRows=0;
    const char *geometryReason="INVALID_ROAD_INPUT";
};

// 交通牌属于场景中的物体，牌面上的深色图案不能充当道路分隔带。
// 只采用新鲜且达到阈值的有效检测框；行人、锥桶、道闸、斑马线不在此列。
inline std::vector<cv::Rect> raceSignRegions(const RaceSnapshot &snap,const RaceConfig &cfg) {
    std::vector<cv::Rect> regions;
    for(const auto &d:snap.detections) {
        const bool sign=d.type==LABEL_BUSY||d.type==LABEL_LIMIT||d.type==LABEL_UNLIMIT||
            d.type==LABEL_STOP||d.type==LABEL_FLOW||d.type==LABEL_PARK||
            d.type==LABEL_FORK||d.type==LABEL_LEFT||d.type==LABEL_CHOICE||
            d.type==LABEL_YLEFT||d.type==LABEL_YRIGHT;
        if(!sign||!reviewRaceDetection(d,snap.image,cfg,&snap.crosswalk).accepted) continue;
        // 原始图像为 640×480；留出少量检测抖动和异步帧位移的余量。
        constexpr int margin=24;
        regions.push_back(cv::Rect(d.x-margin,d.y-margin,d.width+2*margin,d.height+2*margin)&
                          cv::Rect(0,0,640,480));
    }
    return regions;
}

// 仅保留与近处路面相接的白色区段，并把细小黑点和已检出的牌面合并回路面。
inline std::vector<std::pair<int,int>> raceRoadBlocks(const cv::Mat &bin,int row,int low,int high,
        const RaceConfig &cfg,const std::vector<cv::Rect> &signRegions,int minimumBlockWidth=0) {
    std::vector<std::pair<int,int>> blocks;
    for(int col=0;col<bin.cols;) {
        if(bin.at<uint8_t>(row,col)<=127) { ++col; continue; }
        const int first=col;
        while(col<bin.cols&&bin.at<uint8_t>(row,col)>127) ++col;
        if(col-first<(minimumBlockWidth?minimumBlockWidth:cfg.minBranchWidth)||std::min(col-1,high)<std::max(first,low)) continue;
        if(!blocks.empty()) {
            const int gapFirst=blocks.back().second+1,gapLast=first-1;
            const int middle=(gapFirst+gapLast)/2;
            const bool covered=std::any_of(signRegions.begin(),signRegions.end(),[&](const cv::Rect &r) {
                const int overlap=std::min(gapLast,r.x+r.width-1)-std::max(gapFirst,r.x)+1;
                return r.contains({middle,row})&&overlap>=(gapLast-gapFirst+2)/2;
            });
            if(gapLast-gapFirst+1<cfg.minBranchGap||covered) {
                blocks.back().second=col-1; continue;
            }
        }
        blocks.emplace_back(first,col-1);
    }
    return blocks;
}

// 分叉必须是同一组通路持续延伸至搜索区远端，不能把不相邻的几行相加。
// 再用搜索区上方直到图像顶边的可见像素复核分隔带，避免裁掉圆环内部的白色。
// 顶部复核只参与判别，不把这些行送入 v6 的转向路径。
// 车头图案会在更远处重新合成一片白色；牌面的圆环会分成两个黑边，
// 都不满足道路分隔带连续的条件。判断使用本帧像素，AI 漏检时也不沿用旧框。
// 此函数只返回可见几何。方向引导由 RaceManager 单独标记，不伪装成已看见支路。
inline RacePath raceBranchPath(const cv::Mat &bin,const Track &track,int direction,const RaceConfig &cfg,
                               const std::vector<cv::Rect> &signRegions={},bool committed=false) {
    RacePath out;
    if(bin.empty()||bin.type()!=CV_8UC1||bin.rows!=480||bin.cols!=640||
       track.pointsEdgeLeft.empty()||track.pointsEdgeRight.empty()) return out;
    int low=track.pointsEdgeLeft.front().y,high=track.pointsEdgeRight.front().y;
    const int start=track.pointsEdgeLeft.front().x;
    if(start<0||start>=bin.rows||track.rowCutUp<0||track.rowCutUp>=start||
       low<0||high>=bin.cols||low>=high) return out;
    struct Row { int number; std::vector<std::pair<int,int>> blocks; };
    std::vector<Row> rows;
    std::vector<std::pair<int,int>> previousGaps;
    int missing=0,continuous=0,forkStart=-1;
    bool rejected=false,reachedFarEnd=false;
    const char *rejection="NO_CONTINUOUS_SPLIT";
    for(int row=start;row>=0;row=(row==0?-1:std::max(0,row-4))) {
        auto blocks=raceRoadBlocks(bin,row,low,high,cfg,signRegions);
        if(blocks.empty()) {
            // 图像断层不能拼成连续分叉，也不能宣称已看清远端。
            rejected=true; continuous=0; rejection="ROAD_ROWS_MISSING";
            if(++missing>3) break;
            continue;
        }
        missing=0;
        std::vector<std::pair<int,int>> gaps;
        for(size_t i=1;i<blocks.size();++i) {
            gaps.emplace_back(blocks[i-1].second+1,blocks[i].first-1);
        }
        if(gaps.empty()) {
            continuous=0; forkStart=-1; previousGaps.clear(); rejected=false; rejection="NO_CONTINUOUS_SPLIT";
        } else {
            if(!previousGaps.empty()) {
                // 同一个黑色分隔带不能突然变成圆环的两条边，也不能跳到别处的黑点。
                if(gaps.size()!=previousGaps.size()) {rejected=true;rejection="DIVIDER_COUNT_CHANGED";}
                else for(size_t i=0;i<gaps.size();++i) {
                    const int middle=(previousGaps[i].first+previousGaps[i].second)/2;
                    if(middle<gaps[i].first||middle>gaps[i].second) {rejected=true;rejection="DIVIDER_DISCONTINUOUS";}
                }
            }
            if(!rejected&&row>track.rowCutUp) {
                if(continuous==0) forkStart=row;
                ++continuous;
            }
            previousGaps=std::move(gaps);
        }
        // 几何检查保留所有连通支路，避免先选了牌面内部的白色而丢失两侧路面。
        low=blocks.front().first; high=blocks.back().second;
        if(row>track.rowCutUp) rows.push_back({row,std::move(blocks)});
        reachedFarEnd=row==0;
    }
    if(reachedFarEnd&&!rejected) out.splitRows=continuous;
    out.geometryReason=!reachedFarEnd?"FAR_END_NOT_VISIBLE":rejected?rejection:
        out.splitRows>=cfg.minBranchRows?"CONNECTED_SPLIT":"NO_CONTINUOUS_SPLIT";

    low=track.pointsEdgeLeft.front().y; high=track.pointsEdgeRight.front().y;
    missing=0;
    for(const auto &r:rows) {
        std::vector<std::pair<int,int>> blocks;
        for(const auto &b:r.blocks)
            if(std::min(b.second,high)>=std::max(b.first,low)) blocks.push_back(b);
        if(blocks.empty()) { if(++missing>3) break; continue; }
        missing=0;
        auto best=std::min_element(blocks.begin(),blocks.end(),[&](auto a,auto b) {
            return std::abs(a.first+a.second-low-high)<std::abs(b.first+b.second-low-high);
        });
        const bool fork=out.splitRows>=cfg.minBranchRows&&r.number<=forkStart;
        // Once a sign commits the turn, keep the connected target side even
        // when the distant divider no longer passes the full split test.
        if((fork||committed)&&direction&&direction!=2) best=direction<0?blocks.begin():blocks.end()-1;
        low=(!direction)?blocks.front().first:best->first;
        high=(!direction)?blocks.back().second:best->second;
        out.left.emplace_back(r.number,low); out.right.emplace_back(r.number,high);
    }
    return out;
}

struct RacePlan {
    float speedCap=10.f;
    bool stop=false,buzzer=false,fresh=false;
    bool selectedBranch=false;
    bool gateWaiting=false,gateOpened=false;
    int guideDirection=0;
    float guidePwm=0;
    float laneBlend=1; // 0: sign guidance; 1: selected-road pilot, including countersteer.
    int colorRejectedComponents=0,geometrySplitRows=0;
    const char *geometryReason="INVALID_ROAD_INPUT";
    double detectionAge=-1;
    const char *state="AI_WAIT",*reason="AI_WAIT";
    RacePath path;
};

class RaceManager {
    RaceConfig cfg;
    std::array<int,15> counts{};
    std::array<bool,15> confirmed{};
    std::array<PredictResult,15> boxes{};
    uint64_t lastSequence=0;
    double lastCapture=-1,now=0,stateSince=0,cooldownUntil=0;
    double gateClearSince=-1,hazardClearSince=-1;
    double directionUntil=0,splitTime=0,exitTime=0,lastCrossSeen=0;
    bool lastActive=false,limited=false,crossBottomSeen=false,gateWaiting=false,hazardWaiting=false;
    bool hazardPersonWaiting=false;
    bool hazardConeWaiting=false;
    int direction=0;bool routeArmed=false;
    double seekSince=-1;
    float targetVisibleTime=0,handoffTime=0,alignedTime=0,handoffPwm=300;
    float targetCenter=std::numeric_limits<float>::quiet_NaN();
    enum class State { Lane,ForkArmed,ForkSelect,CrossApproach,SignalWait,ReviewStop,Fault };
    State state=State::Lane;
    const char *reviewReason="REVIEW_REQUIRED";
    PredictResult gateBox{};
    void enter(State s) { state=s; stateSince=now; splitTime=exitTime=0;seekSince=-1;
        targetVisibleTime=handoffTime=alignedTime=0;targetCenter=std::numeric_limits<float>::quiet_NaN(); }
    static float guidancePwm(double elapsed) {return 300+40*std::clamp(float(elapsed/.35),0.f,1.f);}
    static bool corridorCenters(const RacePath &path,float &nearCenter,float &farCenter,float &heading) {
        if(path.left.size()<20||path.right.size()!=path.left.size())return false;
        nearCenter=farCenter=heading=0;constexpr size_t samples=8;
        for(size_t i=0;i<samples;++i) {
            const auto center=[&](size_t j){return (path.left[j].y+path.right[j].y)*.5f;};
            nearCenter+=center(i)/samples;farCenter+=center(path.left.size()-1-i)/samples;
            heading+=center(i+samples)/samples;
        }
        heading-=nearCenter;return true;
    }
    bool seen(int label) const { return confirmed[label]; }
    bool near(int label,int row) const { return seen(label)&&boxes[label].y+boxes[label].height>=row; }
    bool gatePassageClear(const RaceSnapshot &snap,const cv::Mat &bin,const cv::Mat &image)const {
        if(counts[LABEL_GATE]||bin.type()!=CV_8UC1||bin.size()!=cv::Size(640,480)||
           image.type()!=CV_8UC3||image.size()!=bin.size()||snap.image.type()!=CV_8UC3||snap.image.size()!=bin.size())return false;
        const int width=std::clamp(gateBox.width/2,60,160);
        const cv::Rect region=cv::Rect(gateBox.x+gateBox.width/2-width/2,gateBox.y,width,gateBox.height)&cv::Rect(0,0,640,480);
        if(region.width<40||region.height<8||cv::countNonZero(bin(region))<region.area()*.85)return false;
        for(const auto &d:snap.detections) {
            if(d.type!=LABEL_PERSON&&d.type!=LABEL_CONE&&d.type!=LABEL_BUSY)continue;
            if(reviewRaceDetection(d,snap.image,cfg,&snap.crosswalk).accepted&&
               (cv::Rect(d.x,d.y,d.width,d.height)&region).area()>0)return false;
        }
        // On a white road the yellow part of an undetected bar can itself
        // binarize as road. Do not call a still-yellow passage unobstructed.
        cv::Mat hsv,yellow;cv::cvtColor(image(region),hsv,cv::COLOR_BGR2HSV);
        cv::inRange(hsv,cv::Scalar(15,100,90),cv::Scalar(40,255,255),yellow);
        return cv::countNonZero(yellow)<region.area()*.08;
    }
    bool updateObservation(const RaceSnapshot &snap,bool fresh) {
        if(!fresh||snap.sequence<=lastSequence||snap.captured<=lastCapture) return false;
        if(lastCapture<0||snap.captured-lastCapture>cfg.confirmGap) counts.fill(0);
        lastSequence=snap.sequence; lastCapture=snap.captured;
        std::array<bool,15> present{};
        for(const auto &d:snap.detections) {
            const auto review=reviewRaceDetection(d,snap.image,cfg,&snap.crosswalk);
            if(!review.accepted) continue;
            auto position=d;
            // 低分 cross 的通过位置以验证过的条纹为准，不能被偏大的模型框提前触发。
            if(review.checkedCross) {
                const auto &r=snap.crosswalk.box;
                position.x=r.x;position.y=r.y;position.width=r.width;position.height=r.height;
            }
            if(!present[d.type]||d.score>boxes[d.type].score) boxes[d.type]=position;
            present[d.type]=true;
        }
        // 模型在远处没有候选时，独立的同帧条纹证据也经过三帧确认。
        // 内部框只供位置判定，分数为零，不作为模型结果对外输出。
        if(cfg.verifyCrosswalkGeometry&&snap.crosswalk.accepted&&!present[LABEL_CROSS]&&
           snap.image.type()==CV_8UC3&&snap.image.size()==cv::Size(640,480)) {
            const auto &r=snap.crosswalk.box;
            boxes[LABEL_CROSS]={LABEL_CROSS,"cross",0.f,r.x,r.y,r.width,r.height};
            present[LABEL_CROSS]=true;
        }
        for(int i=0;i<15;++i) { counts[i]=present[i]?std::min(cfg.confirmFrames,counts[i]+1):0; confirmed[i]=counts[i]>=cfg.confirmFrames; }
        return true;
    }
public:
    explicit RaceManager(const RaceConfig &config):cfg(config) { cfg.validate(); }
    bool aiReady(const RaceSnapshot &s,double time) const {
        return !s.failed&&s.sequence>0&&std::isfinite(s.captured)&&time>=s.captured&&time-s.captured<=cfg.maxDetectionAge;
    }
    RacePlan update(const RaceSnapshot &snap,double time,float dt,bool active,const cv::Mat &bin,const Track &track,
                    const cv::Mat &currentImage={},bool calibratedConeBypass=false,bool taughtParkingManeuver=false,int routeDirection=0,bool continueGeometry=false,bool confirmedRouteSign=false,float parkingAreaSpeed=0) {
        now=time;
        RacePlan p;
        p.gateWaiting=gateWaiting;
        const float forkSpeed=parkingAreaSpeed>0?parkingAreaSpeed:cfg.forkSpeed;
        p.detectionAge=snap.sequence?time-snap.captured:-1;
        p.fresh=aiReady(snap,time);
        const bool newFrame=updateObservation(snap,p.fresh);
        const auto signRegions=p.fresh?raceSignRegions(snap,cfg):std::vector<cv::Rect>{};
        auto geometry=raceBranchPath(bin,track,0,cfg,signRegions);
        // 只在原几何规则将要判为分叉时检查颜色，避免正常巡线增加连通域分析。
        const auto geometryBin=geometry.splitRows>=cfg.minBranchRows?
            raceAppearanceGeometry(bin,currentImage,p.colorRejectedComponents):bin;
        if(p.colorRejectedComponents) geometry=raceBranchPath(geometryBin,track,0,cfg,signRegions);
        p.geometrySplitRows=geometry.splitRows;
        p.geometryReason=geometry.geometryReason;
        if(active!=lastActive) {
            // 人工介入后不继续先前的定时动作；限速区域标志仍保留，不能靠按 a 取消限速。
            if(state!=State::ReviewStop&&state!=State::Fault) { enter(State::Lane); direction=0;routeArmed=false; }
            counts.fill(0); confirmed.fill(false); lastActive=active;
        }
        if(!active) { p.state="MANUAL"; p.reason="MANUAL"; return p; }
        if(!p.fresh) {
            p.stop=true; p.state="AI_WAIT"; p.reason=snap.failed?"AI_FAILED":"AI_STALE";
            counts.fill(0); confirmed.fill(false); splitTime=exitTime=0;
            // 无新观测时不累计恢复或穿越时间，也不能沿用旧方向继续完成动作。
            if(state==State::ForkSelect) { enter(State::Fault); reviewReason="FORK_AI_STALE"; }
            gateClearSince=hazardClearSince=-1; return p;
        }
        if(!std::isfinite(dt)||dt<=0||dt>0.25f) { p.stop=true; p.state="FRAME_WAIT"; p.reason="FRAME_GAP"; return p; }
        if(newFrame) {
            if(seen(LABEL_LIMIT)) limited=true;
            else if(seen(LABEL_UNLIMIT)) limited=false;
        }
        if(limited) p.speedCap=cfg.limitedSpeed;

        // 施工区和泊车不能仅凭标签就宣称已完成；基础版明确停车等待专项规划验证。
        if(state!=State::ReviewStop&&state!=State::Fault&&newFrame&&
           ((!taughtParkingManeuver&&near(LABEL_PARK,cfg.triggerRow))||near(LABEL_BUSY,cfg.triggerRow))) {
            reviewReason=seen(LABEL_PARK)?"PARK_PLANNER_PENDING":"CONSTRUCTION_PLANNER_PENDING";
            enter(State::ReviewStop);
        }
        if(state==State::ReviewStop||state==State::Fault) {
            p.stop=true; p.state="REVIEW_STOP"; p.reason=reviewReason; return p;
        }
        if(state!=State::CrossApproach&&state!=State::SignalWait&&newFrame&&near(LABEL_STOP,cfg.triggerRow)) {
            crossBottomSeen=false; lastCrossSeen=now;
            enter(cfg.stopAfterCross?State::CrossApproach:State::SignalWait);
        }
        // 道闸优先于路线选择；抬杆或连续看清原横杆处的空路面才能放行。
        if(newFrame&&counts[LABEL_GATE]>0&&boxes[LABEL_GATE].y+boxes[LABEL_GATE].height>=cfg.triggerRow&&
           boxes[LABEL_GATE].width>=boxes[LABEL_GATE].height&&!gateWaiting) {
            gateBox=boxes[LABEL_GATE]; gateWaiting=true; gateClearSince=-1;
        }
        if(gateWaiting) {
            if(newFrame) {
                const auto &g=boxes[LABEL_GATE];
                const float center=g.x+g.width*.5f;
                // A raised bar may be above either end of its former horizontal
                // footprint, not within 160px of the old box centre.
                const bool sameArea=center>=gateBox.x-48&&center<=gateBox.x+gateBox.width+48&&
                    std::abs(g.y+g.height-gateBox.y-gateBox.height)<120;
                const bool open=seen(LABEL_GATE)&&sameArea&&g.height>=60&&g.width<float(g.height)*cfg.gateOpenAspect;
                if(open||gatePassageClear(snap,bin,currentImage)) { if(gateClearSince<0) gateClearSince=now; }
                else gateClearSince=-1;
                if(gateClearSince>=0&&now-gateClearSince>=cfg.clearTime) {gateWaiting=false;p.gateOpened=true;}
            }
        }
        p.gateWaiting=gateWaiting;
        if(calibratedConeBypass){hazardConeWaiting=false;if(!hazardPersonWaiting)hazardWaiting=false;}
        // 近桶停车不等三次确认：真实记录曾在接近时漏检，等重新凑满三帧已太迟。
        // 已看见的近桶在没有已验证绕行路线时保持停车，不能靠0.6秒漏检消除义务。
        const bool nearCone=newFrame&&std::any_of(snap.detections.begin(),snap.detections.end(),[&](const PredictResult &d){
            return d.type==LABEL_CONE&&d.y+d.height>=std::min(cfg.obstacleRow,290)&&
                reviewRaceDetection(d,snap.image,cfg,&snap.crosswalk).accepted;
        });
        if(!calibratedConeBypass&&nearCone){hazardConeWaiting=true;hazardWaiting=true;hazardClearSince=-1;}
        if(newFrame&&(near(LABEL_PERSON,cfg.obstacleRow)||(!calibratedConeBypass&&near(LABEL_CONE,cfg.obstacleRow)))) {
            hazardWaiting=true; hazardClearSince=-1;
            if(near(LABEL_PERSON,cfg.obstacleRow))hazardPersonWaiting=true;
        }
        if(hazardWaiting) {
            if(newFrame) {
                // 单次漏检不能放行，连续的新鲜无障碍观测才能恢复。
                const bool any=counts[LABEL_PERSON]>0||hazardConeWaiting||(!calibratedConeBypass&&counts[LABEL_CONE]>0);
                if(any) hazardClearSince=-1;
                else if(hazardClearSince<0) hazardClearSince=now;
                if(hazardClearSince>=0&&now-hazardClearSince>=cfg.clearTime) {hazardWaiting=false;hazardPersonWaiting=false;}
            }
        }
        if(gateWaiting||hazardWaiting) {
            if(seekSince>=0)seekSince+=dt;
            if(state==State::ForkArmed||state==State::ForkSelect){stateSince+=dt;directionUntil+=dt;}
            // 高优先级停车叠加在当前任务之上，不丢失信号灯等待等低优先级义务。
            // 停车期间不累计穿越/选路动作时间。
            stateSince+=dt; splitTime=exitTime=0;
            if(state==State::CrossApproach) lastCrossSeen=now;
            p.stop=true;
            p.state=gateWaiting?"GATE_WAIT":"OBSTACLE_WAIT";
            p.reason=gateWaiting?"GATE_NOT_VERIFIED_OPEN":hazardConeWaiting?"CONE_ROUTE_OR_RESTART_REQUIRED":"OBSTACLE_PLANNER_PENDING";
            return p;
        }
        if(state==State::CrossApproach) {
            p.speedCap=std::min(p.speedCap,cfg.approachSpeed);
            p.state="CROSS_APPROACH"; p.reason="STOP_AFTER_CROSS";
            if(newFrame) {
                if(counts[LABEL_CROSS]>0) {
                    lastCrossSeen=now;
                    if(seen(LABEL_CROSS)&&near(LABEL_CROSS,cfg.crossPassedRow)) crossBottomSeen=true;
                } else if(crossBottomSeen&&now-lastCrossSeen>=cfg.crossClearTime) enter(State::SignalWait);
            }
            if(now-stateSince>6&&state==State::CrossApproach) {
                enter(State::Fault); reviewReason="CROSS_PASS_NOT_VERIFIED";
            }
        }
        if(state==State::SignalWait) {
            p.stop=true; p.state="SIGNAL_WAIT"; p.reason="WAIT_FRESH_FLOW";
            if(newFrame&&seen(LABEL_FLOW)&&counts[LABEL_STOP]==0&&now-stateSince>=cfg.stopHoldTime) {
                enter(State::Lane); cooldownUntil=now+cfg.cooldown; p.stop=false;
                // 放行当帧同步更新显示，避免日志仍写等待、速度却已开始恢复。
                p.state=limited?"LIMIT_ZONE":"LANE";
                p.reason=limited?"LIMIT_SIGN":"CRUISE";
            }
            return p;
        }
        const bool leftSign=seen(LABEL_YLEFT)||seen(LABEL_LEFT),rightSign=seen(LABEL_YRIGHT);
        if(routeArmed&&routeDirection&&routeDirection!=direction&&(state==State::ForkArmed||state==State::ForkSelect)) {
            enter(State::Lane);routeArmed=false;
        }
        if(state==State::Lane&&now>=cooldownUntil&&newFrame) {
            if(routeDirection!=0&&(geometry.splitRows>=cfg.minBranchRows||
                (confirmedRouteSign&&routeDirection==-1)||
                (routeDirection==-1&&leftSign&&!rightSign)||(routeDirection==1&&rightSign&&!leftSign))) {
                direction=routeDirection;routeArmed=true;directionUntil=now+cfg.directionMemory;enter(State::ForkArmed);
            }
            else if(routeDirection!=0) {}
            else if(seen(LABEL_FLOW)&&geometry.splitRows>=cfg.minBranchRows) {
                direction=2;routeArmed=false;directionUntil=now+cfg.directionMemory;enter(State::ForkArmed);
            }
            else if(leftSign&&rightSign) { direction=0; }
            else if(leftSign||rightSign) {
                direction=leftSign?-1:1;routeArmed=false;
                directionUntil=now+cfg.directionMemory; enter(State::ForkArmed);
            }
        }
        if(state==State::ForkArmed) {
            if(direction==-1||direction==1) {
                p.guideDirection=direction;
                // Elevated field signs disappear with box bottoms at rows
                // 31..103, never reaching the old ground-object row145 gate.
                // Confirmed sign direction itself starts the road search.
                if(seekSince<0)seekSince=now;
                p.guidePwm=guidancePwm(now-seekSince);p.laneBlend=0;
            }
            p.speedCap=std::min(p.speedCap,forkSpeed);
            p.state="FORK_ARMED"; p.reason=direction<0?"WAIT_LEFT_SPLIT":"WAIT_RIGHT_SPLIT";
            if(!routeArmed&&newFrame&&leftSign&&rightSign) {
                enter(State::Fault); reviewReason="CONFLICTING_DIRECTIONS";
            } else if(!routeArmed&&newFrame&&((direction<0&&rightSign)||(direction==1&&leftSign))) {
                enter(State::Fault); reviewReason="DIRECTION_CHANGED_BEFORE_SPLIT";
            } else {
                // Fresh same-direction observations renew memory, not the total approach budget.
                if(newFrame&&((routeArmed&&routeDirection==direction)||(direction<0?leftSign:rightSign)))directionUntil=now+cfg.directionMemory;
                if(now-stateSince>cfg.directionMemory+cfg.forkMaxTime) {
                    enter(State::Fault);reviewReason="FORK_NOT_CONFIRMED_TIMEOUT";
                } else if(seekSince<0&&now>directionUntil) {
                    enter(State::Fault);reviewReason="DIRECTION_MEMORY_EXPIRED";
                } else if(direction==2&&geometry.splitRows>=cfg.minBranchRows) {
                    splitTime+=dt;
                    if(splitTime>=cfg.forkConfirmTime) { enter(State::ForkSelect); p.buzzer=true; }
                } else {
                    splitTime=0;
                    if((direction==-1||direction==1)&&seekSince>=0) {
                        const double elapsed=now-seekSince;
                        p.state=direction<0?"FORK_SEEK_LEFT":"FORK_SEEK_RIGHT";
                        p.reason="STRONG_SIGN_GUIDANCE_FIND_TARGET_ROAD";
                        p.guidePwm=guidancePwm(elapsed);p.laneBlend=0;
                        const auto candidate=raceBranchPath(geometryBin,track,direction,cfg,signRegions,true);
                        float nearCenter=0,farCenter=0,heading=0,unionNear=0,unionFar=0,unionHeading=0;
                        const bool visible=corridorCenters(candidate,nearCenter,farCenter,heading);
                        const bool splitTarget=geometry.splitRows>=cfg.minBranchRows&&
                            corridorCenters(geometry,unionNear,unionFar,unionHeading)&&direction*(farCenter-unionFar)>24;
                        const bool sameTarget=visible&&targetVisibleTime>0&&std::isfinite(targetCenter)&&
                            std::abs(farCenter-targetCenter)<80;
                        const bool target=visible&&(splitTarget||sameTarget||
                            (direction*(farCenter-nearCenter)>40&&direction*(farCenter-320)>20));
                        // One visible patch must not cancel the strong turn.
                        // Confirm a continuous connected target, never the opposite corridor.
                        if(target&&(!std::isfinite(targetCenter)||std::abs(farCenter-targetCenter)<80))targetVisibleTime+=dt;
                        else targetVisibleTime=0;
                        targetCenter=target?farCenter:std::numeric_limits<float>::quiet_NaN();
                        if(targetVisibleTime>=std::max(.18f,cfg.forkConfirmTime)) {
                            const float guide=p.guidePwm;enter(State::ForkSelect);handoffPwm=guide;p.buzzer=true;
                        } else if(elapsed>=cfg.forkMaxTime) {enter(State::Fault);reviewReason="SIGNED_ROAD_REACQUIRE_TIMEOUT";}
                    }
                }
            }
        }
        if(state==State::ForkSelect) {
            p.guideDirection=direction==2?0:direction;p.guidePwm=0;
            p.speedCap=std::min(p.speedCap,forkSpeed);
            p.state=direction<0?"FORK_LEFT":direction==2?"FORK_STRAIGHT":"FORK_RIGHT"; p.reason=routeArmed?"MISSION_CONNECTED_BRANCH":"CONNECTED_BRANCH";
            p.path=raceBranchPath(geometryBin,track,direction,cfg,signRegions,true); p.selectedBranch=true;
            bool aligned=direction==2;
            if(direction!=2) {
                float nearCenter=0,farCenter=0,heading=0;
                const bool visible=corridorCenters(p.path,nearCenter,farCenter,heading);
                const bool continuous=visible&&(!std::isfinite(targetCenter)||std::abs(farCenter-targetCenter)<96);
                p.guidePwm=handoffPwm;p.laneBlend=std::clamp(handoffTime/.65f,0.f,1.f);
                if(continuous){handoffTime+=dt;targetCenter=farCenter;}
                else {
                    // Do not feed an abruptly substituted opposite road to the
                    // pilot. Continue the signed turn instead of latching stop.
                    enter(State::ForkArmed);seekSince=now;
                    p.selectedBranch=false;p.path={};p.laneBlend=0;p.guidePwm=guidancePwm(0);
                    p.state=direction<0?"FORK_SEEK_LEFT":"FORK_SEEK_RIGHT";
                    p.reason="TARGET_ROAD_LOST_KEEP_SIGN_GUIDANCE";return p;
                }
                if(continuous&&p.laneBlend>=1&&std::abs(nearCenter-320)<80&&std::abs(heading)<32)alignedTime+=dt;
                else alignedTime=0;
                aligned=alignedTime>=.18f;
                if(continuous)p.reason=p.laneBlend<1?"TARGET_ROAD_GRADUAL_HANDOFF":"COMMITTED_ROAD_CENTER_TRACKING";
            }
            if(p.path.left.size()<8) { enter(State::Fault); reviewReason="BRANCH_PATH_LOST"; }
            else if(now-stateSince>cfg.forkMaxTime) { enter(State::Fault); reviewReason="FORK_TIMEOUT"; }
            else if(now-stateSince>=cfg.forkMinTime&&geometry.splitRows==0&&aligned) {
                exitTime+=dt;
                if(exitTime>=cfg.forkExitTime) {
                    enter(State::Lane); cooldownUntil=now+cfg.cooldown;
                    p.selectedBranch=false; p.path={};
                    p.guideDirection=0;
                }
            } else exitTime=0;
        }
        if(state==State::Lane) {
            p.state=limited?"LIMIT_ZONE":"LANE"; p.reason=limited?"LIMIT_SIGN":"CRUISE";
            if(geometry.splitRows>=cfg.minBranchRows&&now>=cooldownUntil&&!continueGeometry) {
                p.stop=true; p.state="FORK_WAIT"; p.reason="NO_CONFIRMED_DIRECTION";
            }
        }
        const std::string_view failureReason=reviewReason;
        if(state==State::Fault&&continueGeometry&&(failureReason=="BRANCH_PATH_LOST"||failureReason=="FORK_TIMEOUT"||
           failureReason=="SIGNED_ROAD_REACQUIRE_TIMEOUT"||failureReason=="FORK_NOT_CONFIRMED_TIMEOUT")) {
            // Painted geometry warnings must not latch the competition run.
            // Seek the committed side again, then reacquire current road edges.
            enter(State::ForkArmed);seekSince=now;
            p.stop=false;p.selectedBranch=false;p.path={};p.guideDirection=direction==2?0:direction;
            p.guidePwm=p.guideDirection?guidancePwm(0):0;p.laneBlend=p.guideDirection?0:1;
            p.state="FORK_RECOVER";p.reason="GEOMETRY_WARNING_CONTINUE";
        }
        else if(state==State::Fault) { p.stop=true; p.state="REVIEW_STOP"; p.reason=reviewReason; }
        return p;
    }
};
