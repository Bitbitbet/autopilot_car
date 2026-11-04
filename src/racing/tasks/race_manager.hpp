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
    int splitRows=0,choiceRows=0;
    const char *geometryReason="INVALID_ROAD_INPUT";
    bool singleRoadAhead=false;
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
        if(row>track.rowCutUp) {
            if(blocks.size()>1)++out.choiceRows;
            rows.push_back({row,std::move(blocks)});
        }
        reachedFarEnd=row==0;
    }
    if(reachedFarEnd&&!rejected) out.splitRows=continuous;
    out.geometryReason=!reachedFarEnd?"FAR_END_NOT_VISIBLE":rejected?rejection:
        out.splitRows>=cfg.minBranchRows?"CONNECTED_SPLIT":"NO_CONTINUOUS_SPLIT";

    // The bonnet may divide the bottom of a black road into two strips that
    // immediately join ahead. A true fork opens ahead of the vehicle.
    // Require a connected centre lane with a visible edge in the lookahead.
    if(direction==0&&out.splitRows==0) {
        bool bottomSplit=true;
        int forwardChoices=0,edgeRun=0,previousRow=start;
        for(const auto &r:rows) {
            const int ahead=start-r.number;
            if(r.blocks.size()==1||ahead>=64)bottomSplit=false;
            if(!bottomSplit&&r.blocks.size()>1)++forwardChoices;
            const auto [left,right]=r.blocks.front();
            if(r.number!=previousRow-4)edgeRun=0;
            previousRow=r.number;
            if(ahead>=64&&ahead<=192&&r.blocks.size()==1&&left<320&&right>320&&
               (left>2||right<637)) {
                if(++edgeRun>=6)out.singleRoadAhead=true;
            } else edgeRun=0;
        }
        if(forwardChoices>=cfg.minBranchRows)out.singleRoadAhead=false;
    }
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

// Parking entrances are side openings, not necessarily Y splits. Estimate the
// unexpanded lane width from the closest visible rows, then offset the connected
// left edge into the road. These are image-space estimates, not metric clearance.
struct RaceEntranceEdge {
    RacePath path;
    int openingRow=-1;
    bool opening=false;
    float nearWidth=0;
};
inline RaceEntranceEdge raceParkingEntranceEdge(const RacePath &road) {
    RaceEntranceEdge out;
    if(road.left.size()<24||road.right.size()!=road.left.size())return out;
    const int start=road.left.front().x;
    float n=0,sy=0,syy=0,sl=0,syl=0,sw=0,syw=0;
    for(size_t i=0;i<road.left.size();++i){
        const auto l=road.left[i],r=road.right[i];const float y=start-l.x;
        if(y>60)break;
        if(l.y<=2||r.y>=637||r.y-l.y<60)continue;
        ++n;sy+=y;syy+=y*y;sl+=l.y;syl+=y*l.y;sw+=r.y-l.y;syw+=y*(r.y-l.y);
    }
    const float denominator=n*syy-sy*sy;
    if(n<8||denominator<1)return out;
    const float leftSlope=(n*syl-sy*sl)/denominator;
    const float widthSlope=std::clamp((n*syw-sy*sw)/denominator,-2.f,.5f);
    const float leftBase=(sl-leftSlope*sy)/n,widthBase=(sw-widthSlope*sy)/n;
    out.nearWidth=widthBase;
    int count=0,first=-1;
    for(size_t i=0;i<road.left.size();++i){
        const auto l=road.left[i],r=road.right[i];const float y=start-l.x;
        const float width=std::clamp(widthBase+widthSlope*y,60.f,600.f);
        const float expansion=(r.y-l.y)-width;
        // A bend of both edges, a distant opening or a one-row paint break is
        // not permission to steer into the island beside the current lane.
        const bool opens=y>=64&&y<=180&&l.y>2&&
            leftBase+leftSlope*y-l.y>std::max(32.f,width*.18f)&&
            expansion>std::max(40.f,width*.22f);
        if(opens){if(!count)first=l.x;++count;}
        else count=0;
        if(count>=5&&first>=start-144){out.opening=true;out.openingRow=first;}
        // Keep the real right edge on normal-width rows. At the broad mouth,
        // synthesize only the inward edge so the old straight road cannot pull
        // the target to the midpoint of the whole junction.
        const int inner=expansion>width*.22f?std::min(r.y,int(std::lround(l.y+width))):r.y;
        out.path.left.push_back(l);out.path.right.emplace_back(l.x,inner);
    }
    out.path.geometryReason="LEFT_EDGE_OFFSET_ESTIMATE";
    return out;
}

// Previous applied command and current-road pilot feedback. Command distance
// is an estimate, not odometry or proof of physical clearance of a sign.
struct RaceForkFeedback {
    bool available=false,roadReady=false;
    float offset=0,heading=0,speed=0;
    uint16_t appliedPwm=1500;
};
// Applied-command progress is only a field-tunable approach estimate, not
// odometry. PARK size alone must neither start full lock nor wait for edges forever.
struct RaceEntranceProgress {
    float approach=0,signClear=0,pulseDistance=0,pulseElapsed=0,trackingTime=0;
    void approachStep(float dt,bool near,bool newFrame,bool parkVisible,float speed){
        if(near)approach+=std::max(0.f,speed)*dt;
        if(newFrame)signClear=parkVisible?0:signClear+dt;
    }
    bool fallbackReady(bool near)const{return near&&approach>=.35f&&(signClear>=.20f||approach>=.55f);}
    void pulseStep(float dt,float speed){pulseElapsed+=dt;pulseDistance+=std::max(0.f,speed)*dt;}
    float pwm()const{
        const float phase=std::max(pulseDistance/.20f,pulseElapsed*.5f);
        return phase<.5f?360+40*std::clamp(phase/.2f,0.f,1.f):
            400*(1-std::clamp((phase-.5f)/1.25f,0.f,1.f));
    }
};
// During a left turn a stationary left sign moves across the camera to the
// right edge (mirrored for right). A lone box at the edge is not a pass.
struct RaceForkSignPass {
    cv::Rect box;
    double lastSeen=-1;
    int label=-1,initialHeight=0,centeredFrames=0;
    float steeredDistance=0;
    bool crossedCenter=false,sawLocalSide=false;

    bool update(int direction,const PredictResult *sign,double captured,bool newFrame,float dt,
                const RaceForkFeedback &road,bool singleRoad,float maxGap) {
        if(newFrame&&sign) {
            const cv::Rect next{sign->x,sign->y,sign->width,sign->height};
            const float center=next.x+next.width*.5f;
            if(label!=sign->type||captured-lastSeen>maxGap||
               (box&next).area()<.10f*std::min(box.area(),next.area())||
               direction*(center-(box.x+box.width*.5f))>box.width*.50f) {
                *this={};
                if(direction*(center-320)<96)return false;
                label=sign->type;initialHeight=next.height;
            }
            box=next;lastSeen=captured;
            crossedCenter|=std::abs(center-320)<=64;
        }
        if(label<0||captured-lastSeen>1.4) {*this={};return false;}
        const bool atExitEdge=crossedCenter&&direction*(box.x+box.width*.5f-320)<=-160&&
            (direction<0?box.br().x>=576:box.x<=64)&&box.height>=initialHeight*1.15f;
        if(sign&&!atExitEdge&&road.speed>0&&direction*(int(road.appliedPwm)-1500)>=100)
            steeredDistance+=road.speed*dt;
        const float lateral=direction*road.offset;
        if(!atExitEdge||!road.roadReady||!std::isfinite(lateral)||!std::isfinite(road.heading)||
           lateral<=-64||std::abs(road.heading)>=60) {
            sawLocalSide=false;centeredFrames=0;return false;
        }
        if(newFrame&&road.speed>0&&lateral>=64&&lateral<160)sawLocalSide=true;
        if(sign||!sawLocalSide||!singleRoad||std::abs(lateral)>=64||steeredDistance<.08f) {
            centeredFrames=0;return false;
        }
        if(newFrame)centeredFrames=std::min(3,centeredFrames+1);
        return centeredFrames>=3;
    }
};
struct RaceYForkProgress {
    RaceForkSignPass passingSign;
    float elapsed=0,pulsePhase=0,commandDistance=0,clearTime=0,trackingTime=0;
    float lostDistance=0,lostMovingTime=0,confirmedTime=0;
    float exitTurnTravel=0,exitRoadTime=0;
    int exitRoadFrames=0;double exitLastCapture=-1;
    int confirmedFrames=0;
    float confirmedCenter=std::numeric_limits<float>::quiet_NaN();
    float lastUsableCenter=std::numeric_limits<float>::quiet_NaN();
    bool waitingForRoad=false;
    int approachSideFrames=0;
    bool sawApproachSide=false;
    void resetApproachSide(){approachSideFrames=0;sawApproachSide=false;}
    // Observe the local lane moving from the signed side towards the car.
    // A steering command or a centred image alone cannot establish the turn.
    void observeApproachSide(int direction,const RaceForkFeedback &road,bool newFrame) {
        const float lateral=direction*road.offset;
        if(!road.roadReady||!std::isfinite(lateral)||!std::isfinite(road.heading)||
           lateral<=-64||std::abs(road.heading)>=60) {resetApproachSide();return;}
        if(!newFrame||sawApproachSide)return;
        if(pulsePhase>=.4f&&road.speed>0&&lateral>=64&&lateral<160) {
            if(++approachSideFrames>=2)sawApproachSide=true;
        } else approachSideFrames=0;
    }
    void clearConfirmation() { confirmedFrames=0;confirmedTime=0;confirmedCenter=std::numeric_limits<float>::quiet_NaN(); }
    // Near-road feedback describes the previous command interval. Once the sign
    // pulse ends, only sustained usable geometry may replenish this allowance.
    void observeTarget(bool usable,float center,bool newFrame,float dt,float appliedSpeed,bool signedReference) {
        if(!usable) {
            clearConfirmation();
            if(appliedSpeed>0) {lostDistance+=appliedSpeed*dt;lostMovingTime+=dt;}
            if(lostDistance>=.12f||lostMovingTime>=.90f)waitingForRoad=true;
        } else if(newFrame) {
            if(std::isfinite(confirmedCenter)&&std::abs(center-confirmedCenter)>=80)clearConfirmation();
            confirmedCenter=center;++confirmedFrames;confirmedTime+=dt;
            if(confirmedFrames>=3&&confirmedTime>=.18f) {
                // Only a measured far end can identify this arm after a later loss.
                if(signedReference)lastUsableCenter=center;
                waitingForRoad=false;lostDistance=lostMovingTime=0;
            }
        }
    }
    void update(float dt,bool newObservation,bool signPresent,const RaceForkFeedback &road) {
        elapsed+=dt;
        const float speed=road.available&&std::isfinite(road.speed)?std::max(0.f,road.speed):.20f;
        // Preserve the successful .20 m/s pulse distance at slower speeds,
        // but always end the pulse within 2.8 s, including a stalled vehicle.
        pulsePhase=std::max(pulsePhase+dt*std::min(speed/.20f,1.f),elapsed*.5f);
        if(road.available)commandDistance+=speed*dt;
        if(signPresent){clearTime=trackingTime=0;return;}
        if(!newObservation)return;
        clearTime+=dt;
        // A curved road need not be parallel to the camera. Require sustained
        // local tracking, not a perfectly straight far corridor.
        if(pulsePhase>=1.4f&&road.available&&road.roadReady&&road.speed>0&&
           std::isfinite(road.offset)&&std::isfinite(road.heading)&&
           std::abs(road.offset)<64&&std::abs(road.heading)<60)trackingTime+=dt;
        else trackingTime=0;
    }
    bool complete()const{return pulsePhase>=1.4f&&commandDistance>=.30f&&clearTime>=.60f&&trackingTime>=.60f;}
    bool observeExitRoad(const RaceForkFeedback &road,bool newObservation,float dt,bool selectedRoad,bool singleRoad,int direction,double captured) {
        // A view of a straight aisle before the car actually steers left cannot
        // complete the exit. Distance here is applied-command evidence only.
        if(!(dt>0&&dt<=.25f)||!std::isfinite(captured)){exitRoadFrames=0;exitRoadTime=0;return false;}
        if(road.available&&road.speed>0&&direction*(int(road.appliedPwm)-1500)>=80)
            exitTurnTravel+=road.speed*dt;
        const bool tracking=selectedRoad&&singleRoad&&road.available&&road.roadReady&&
            std::abs(road.offset)<64&&std::abs(road.heading)<60&&exitTurnTravel>=.08f;
        if(!tracking){exitRoadFrames=0;exitRoadTime=0;return false;}
        if(!newObservation)return false;
        if(captured<=exitLastCapture||captured-exitLastCapture>.30){exitRoadFrames=0;exitRoadTime=0;}
        exitRoadTime+=exitRoadFrames?float(captured-exitLastCapture):0;
        exitLastCapture=captured;++exitRoadFrames;
        // At the travel bound the vehicle may stand still and verify the
        // acquired road. Forward speed cannot be a prerequisite to release it.
        return commandDistance>=.30f&&pulsePhase>=1.4f&&exitRoadFrames>=3&&exitRoadTime>=.24f&&
            std::abs(road.offset)<48&&std::abs(road.heading)<45;
    }
};

struct RacePlan {
    float speedCap=10.f;
    bool stop=false,buzzer=false,fresh=false;
    cv::Rect acceptedStopBox{}; // Same-frame reviewed detection, independent of approach distance.
    cv::Rect acceptedCrossBox{};
    bool acceptedCrossAtPassRow=false;
    bool selectedBranch=false;
    bool alignRoad=false;
    bool parkingTurnCommit=false;
    int parkingOpeningRow=-1;
    bool parkingEdgeEstimated=false;
    bool parkingEntryFallback=false;
    float parkingApproachCommandM=0;
    bool gateWaiting=false,gateOpened=false;
    cv::Rect gateClearRegion{};
    float gateRoadFraction=-1,gateYellowFraction=-1;
    int guideDirection=0;
    int completedDirection=0;
    float forkCommandDistance=0,forkTrackingTime=0,forkSignClearTime=0,forkPulsePhase=0;
    float forkLostDistance=0,forkLostMovingTime=0;
    float guidePwm=0;
    float laneBlend=1; // 0: sign guidance; 1: selected-road pilot, including countersteer.
    int colorRejectedComponents=0,geometrySplitRows=0,geometryChoiceRows=0;
    bool singleRoadAhead=false;
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
    double gateSlowUntil=-1;
    double directionUntil=0,splitTime=0,exitTime=0,lastCrossSeen=0;
    bool lastActive=false,limited=false,crossBottomSeen=false,gateWaiting=false,hazardWaiting=false;
    bool hazardPersonWaiting=false;
    bool hazardConeWaiting=false;
    int direction=0;bool routeArmed=false;
    double seekSince=-1;
    RaceYForkProgress yProgress;
    float entranceOpeningTime=0;
    RaceEntranceProgress entranceProgress;
    bool entranceGuided=false;
    bool entranceGatePassed=false;
    bool exitRoute=false;
    float targetVisibleTime=0,handoffTime=0,alignedTime=0,handoffPwm=300;
    float targetCenter=std::numeric_limits<float>::quiet_NaN();
    enum class State { Lane,ForkArmed,ForkSelect,CrossApproach,SignalWait,ReviewStop,Fault };
    State state=State::Lane;
    const char *reviewReason="REVIEW_REQUIRED";
    PredictResult gateBox{};
    bool gateAtSensor=false;
    float gatePassCommandRemaining=0;
    void enter(State s) { state=s; stateSince=now; splitTime=exitTime=0;seekSince=-1;
        targetVisibleTime=handoffTime=alignedTime=0;targetCenter=std::numeric_limits<float>::quiet_NaN(); }
    // Experimental PWM demand, not a calibrated wheel angle. Use the existing
    // actuator range; target-road visibility still governs the handoff.
    float guidancePwm(double elapsed)const {
        if(entranceGuided)return entranceProgress.pwm();
        if(routeArmed&&!exitRoute)return 360+40*std::clamp(float(elapsed/.20),0.f,1.f);
        // A Y fork is shallower than the parking entrance. Use an initial
        // directional pulse, then ease off instead of holding full lock until
        // a restrictive far-road test eventually succeeds.
        if(elapsed<.40)return 360+40*std::clamp(float(elapsed/.12),0.f,1.f);
        return 400*(1-std::clamp(float((elapsed-.40)/1.0),0.f,1.f));
    }
    static bool corridorCenters(const RacePath &path,float &nearCenter,float &farCenter,float &heading) {
        if(path.left.size()<20||path.right.size()!=path.left.size())return false;
        nearCenter=farCenter=heading=0;constexpr size_t samples=8;
        for(size_t i=0;i<samples;++i) {
            const auto center=[&](size_t j){return (path.left[j].y+path.right[j].y)*.5f;};
            nearCenter+=center(i)/samples;farCenter+=center(path.left.size()-1-i)/samples;
        }
        float nearRow=0,aheadRow=0,aheadCenter=0;int count=0;
        for(size_t i=0;i<samples;++i)nearRow+=path.left[i].x/float(samples);
        for(size_t i=0;i<path.left.size();++i)if(nearRow-path.left[i].x>=64&&nearRow-path.left[i].x<=128){
            aheadRow+=path.left[i].x;aheadCenter+=(path.left[i].y+path.right[i].y)*.5f;++count;
        }
        if(count<4)return false;
        heading=(aheadCenter/count-nearCenter)*80/(nearRow-aheadRow/count);return true;
    }
    bool seen(int label) const { return confirmed[label]; }
    bool near(int label,int row) const { return seen(label)&&boxes[label].y+boxes[label].height>=row; }
    cv::Rect gatePassageRegion()const {
        const int width=std::clamp(gateBox.width/2,60,160);
        // Near the sensor the last bar box reaches the bonnet. Once lifted,
        // that old rectangle contains our own car, not a blocked passage.
        // Check the adjacent road above the bonnet, retaining object/yellow
        // rejection and the continuous stopped clearance interval.
        const int height=std::clamp(gateBox.height,16,64);
        const int row=std::min(gateBox.y,370-height);
        return cv::Rect(gateBox.x+gateBox.width/2-width/2,row,width,height)&cv::Rect(0,0,640,480);
    }
    bool gatePassageClear(const RaceSnapshot &snap,const cv::Mat &bin,const cv::Mat &image,RacePlan &p)const {
        const auto &region=p.gateClearRegion;
        if(counts[LABEL_GATE]||bin.type()!=CV_8UC1||bin.size()!=cv::Size(640,480)||
           image.type()!=CV_8UC3||image.size()!=bin.size()||snap.image.type()!=CV_8UC3||snap.image.size()!=bin.size()||
           region.width<40||region.height<8)return false;
        p.gateRoadFraction=float(cv::countNonZero(bin(region)))/region.area();
        if(p.gateRoadFraction<.85f)return false;
        for(const auto &d:snap.detections) {
            if(d.type!=LABEL_PERSON&&d.type!=LABEL_CONE&&d.type!=LABEL_BUSY)continue;
            if(reviewRaceDetection(d,snap.image,cfg,&snap.crosswalk).accepted&&
               (cv::Rect(d.x,d.y,d.width,d.height)&region).area()>0)return false;
        }
        // On a white road the yellow part of an undetected bar can itself
        // binarize as road. Do not call a still-yellow passage unobstructed.
        cv::Mat hsv,yellow;cv::cvtColor(image(region),hsv,cv::COLOR_BGR2HSV);
        cv::inRange(hsv,cv::Scalar(15,100,90),cv::Scalar(40,255,255),yellow);
        p.gateYellowFraction=float(cv::countNonZero(yellow))/region.area();
        return p.gateYellowFraction<.08f;
    }
    bool updateObservation(const RaceSnapshot &snap,bool fresh) {
        if(!fresh||snap.sequence<=lastSequence||snap.captured<=lastCapture) return false;
        if(lastCapture<0||snap.captured-lastCapture>cfg.confirmGap){counts.fill(0);gateClearSince=-1;}
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
                    const cv::Mat &currentImage={},bool calibratedConeBypass=false,bool taughtParkingManeuver=false,int routeDirection=0,bool continueGeometry=false,bool confirmedRouteSign=false,float parkingAreaSpeed=0,const RaceForkFeedback &road={},bool parkingEntrance=false,bool parkingEntryNear=false,bool parkingAisle=false,bool parkingExit=false,bool finalReturn=false) {
        now=time;
        if(parkingExit&&!exitRoute)yProgress={};
        exitRoute=parkingExit;
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
        p.geometryChoiceRows=geometry.choiceRows;p.singleRoadAhead=geometry.singleRoadAhead;
        p.geometryReason=geometry.geometryReason;
        if(active!=lastActive) {
            // 人工介入后不继续先前的定时动作；限速区域标志仍保留，不能靠按 a 取消限速。
            if(state!=State::ReviewStop&&state!=State::Fault) { enter(State::Lane); direction=0;routeArmed=false; }
            counts.fill(0); confirmed.fill(false); lastActive=active;entranceOpeningTime=0;entranceProgress={};entranceGuided=false;entranceGatePassed=false;
        }
        if(!active) { p.state="MANUAL"; p.reason="MANUAL"; return p; }
        if(!p.fresh) {
            yProgress.passingSign={};
            yProgress.clearTime=yProgress.trackingTime=0;
            yProgress.clearConfirmation();
            yProgress.resetApproachSide();
            p.stop=true; p.state="AI_WAIT"; p.reason=snap.failed?"AI_FAILED":"AI_STALE";
            counts.fill(0); confirmed.fill(false); splitTime=exitTime=0;entranceOpeningTime=0;
            // 无新观测时不累计恢复或穿越时间，也不能沿用旧方向继续完成动作。
            // During the parking entrance, a temporary data hold must not
            // turn the recovered watchdog into a permanent route fault.
            if(state==State::ForkSelect&&!parkingEntrance&&!parkingAisle&&!parkingExit) { enter(State::Fault); reviewReason="FORK_AI_STALE"; }
            gateClearSince=hazardClearSince=-1; return p;
        }
        if(!std::isfinite(dt)||dt<=0||dt>0.25f) { yProgress.passingSign={};yProgress.clearConfirmation();yProgress.resetApproachSide();p.stop=true; p.state="FRAME_WAIT"; p.reason="FRAME_GAP"; return p; }
        if(newFrame) {
            if(counts[LABEL_CROSS]>0) {
                const auto &d=boxes[LABEL_CROSS];
                p.acceptedCrossBox={d.x,d.y,d.width,d.height};
                p.acceptedCrossAtPassRow=d.y+d.height>=cfg.crossPassedRow;
            }
            if(counts[LABEL_STOP]>0) {
                const auto &d=boxes[LABEL_STOP];
                p.acceptedStopBox={d.x,d.y,d.width,d.height};
            }
            if(seen(LABEL_LIMIT)) limited=true;
            else if(seen(LABEL_UNLIMIT)) limited=false;
        }
        if(limited) p.speedCap=cfg.limitedSpeed;
        // Slow at the first accepted horizontal bar, not only after it
        // reaches the close-range sensor approach row. Keep steering under
        // the current task so a distant gate cannot interrupt the entry turn.
        if(newFrame&&counts[LABEL_GATE]>0&&boxes[LABEL_GATE].width>=80&&
           boxes[LABEL_GATE].width>=boxes[LABEL_GATE].height)
            gateSlowUntil=now+.40;
        if(now<gateSlowUntil)p.speedCap=std::min(p.speedCap,cfg.gateApproachSpeed);
        // Keep area speed through the just-opened gate even when mission
        // observes exit release. Command distance is not measured clearance.
        if(gatePassCommandRemaining>0){
            p.speedCap=std::min(p.speedCap,cfg.parkingSpeed);
            if(road.available&&std::isfinite(road.speed))
                gatePassCommandRemaining=std::max(0.f,gatePassCommandRemaining-std::max(0.f,road.speed)*dt);
        }

        // 施工区和泊车不能仅凭标签就宣称已完成；基础版明确停车等待专项规划验证。
        if(state!=State::ReviewStop&&state!=State::Fault&&newFrame&&
           ((!taughtParkingManeuver&&near(LABEL_PARK,cfg.triggerRow))||near(LABEL_BUSY,cfg.triggerRow))) {
            reviewReason=seen(LABEL_PARK)?"PARK_PLANNER_PENDING":"CONSTRUCTION_PLANNER_PENDING";
            enter(State::ReviewStop);
        }
        if(state==State::ReviewStop||state==State::Fault) {
            p.stop=true; p.state="REVIEW_STOP"; p.reason=reviewReason; return p;
        }
        if(!finalReturn&&state!=State::CrossApproach&&state!=State::SignalWait&&newFrame&&near(LABEL_STOP,cfg.triggerRow)) {
            crossBottomSeen=false; lastCrossSeen=now;
            enter(cfg.stopAfterCross?State::CrossApproach:State::SignalWait);
        }
        // 道闸优先于路线选择；抬杆或连续看清原横杆处的空路面才能放行。
        // Gate proximity has its own sensor reference; the generic sign row
        // must not postpone a gate stop which is now earlier than that row.
        const int gateApproachRow=std::min(cfg.triggerRow,cfg.gateStopRow-50);
        if(newFrame&&counts[LABEL_GATE]>0&&boxes[LABEL_GATE].y+boxes[LABEL_GATE].height>=gateApproachRow&&
           boxes[LABEL_GATE].width>=boxes[LABEL_GATE].height&&!gateWaiting) {
            gateBox=boxes[LABEL_GATE]; gateWaiting=true; gateClearSince=-1;
            gateAtSensor=gateBox.y+gateBox.height>=cfg.gateStopRow;
        }
        if(gateWaiting) {
            if(newFrame) {
                const auto &g=boxes[LABEL_GATE];
                if(counts[LABEL_GATE]>0&&g.width>=g.height){
                    // The stopped horizontal bar marks the road to inspect after lifting.
                    // A rising arm can still have a wide box. Following it would inspect the wall.
                    if(!gateAtSensor&&g.y+g.height>=gateBox.y+gateBox.height)gateBox=g;
                    gateAtSensor|=g.y+g.height>=cfg.gateStopRow;
                }
                p.gateClearRegion=gatePassageRegion();
                const float center=g.x+g.width*.5f;
                // A raised bar may be above either end of its former horizontal
                // footprint, not within 160px of the old box centre.
                const bool sameArea=center>=gateBox.x-48&&center<=gateBox.x+gateBox.width+48&&
                    std::abs(g.y+g.height-gateBox.y-gateBox.height)<120;
                const bool open=seen(LABEL_GATE)&&sameArea&&g.height>=60&&g.width<float(g.height)*cfg.gateOpenAspect;
                if(open||gatePassageClear(snap,bin,currentImage,p)) { if(gateClearSince<0) gateClearSince=now; }
                else gateClearSince=-1;
                if(gateClearSince>=0&&now-gateClearSince>=std::max(cfg.clearTime,cfg.gateHoldTime)) {
                    gateWaiting=false;gateAtSensor=false;p.gateOpened=true;gatePassCommandRemaining=.40f;
                    gateSlowUntil=-1;
                    // The entrance road has already been acquired. Its gate
                    // release is a stronger phase landmark than requiring the
                    // branched parking aisle to look like a single straight road.
                    const bool guidedAisle=state==State::ForkArmed&&entranceGuided&&
                        entranceProgress.pwm()<=0&&entranceProgress.pulseDistance>=.30f&&
                        road.available&&road.roadReady&&std::isfinite(road.offset)&&std::isfinite(road.heading)&&
                        std::abs(road.offset)<64&&std::abs(road.heading)<60;
                    if(parkingEntrance&&routeArmed&&direction==-1&&(state==State::ForkSelect||guidedAisle))
                        entranceGatePassed=true;
                    p.speedCap=std::min(p.speedCap,cfg.parkingSpeed);
                }
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
            yProgress.passingSign={};
            yProgress.clearConfirmation();
            yProgress.resetApproachSide();
            yProgress.clearTime=yProgress.trackingTime=0;
            if(seekSince>=0)seekSince+=dt;
            if(state==State::ForkArmed||state==State::ForkSelect){stateSince+=dt;directionUntil+=dt;}
            // 高优先级停车叠加在当前任务之上，不丢失信号灯等待等低优先级义务。
            // 停车期间不累计穿越/选路动作时间。
            stateSince+=dt; splitTime=exitTime=0;
            if(state==State::CrossApproach) lastCrossSeen=now;
            p.stop=true;
            p.state=gateWaiting?"GATE_WAIT":"OBSTACLE_WAIT";
            p.reason=gateWaiting?"GATE_NOT_VERIFIED_OPEN":hazardConeWaiting?"CONE_ROUTE_OR_RESTART_REQUIRED":"OBSTACLE_PLANNER_PENDING";
            if(gateWaiting&&!gateAtSensor&&!hazardWaiting&&counts[LABEL_GATE]==0){
                p.state="GATE_APPROACH_HOLD";p.reason="WAIT_FOR_GATE_VIEW";
            }
            if(gateWaiting&&gateClearSince>=0&&!hazardWaiting){p.state="GATE_OPEN_DWELL";p.reason="OPEN_PASSAGE_ONE_SECOND_HOLD";}
            // Approach the side sensor along the current road, rather than
            // stopping as soon as the distant bar reaches the generic row.
            // Missing/raised detections wait for release; never creep blindly.
            if(gateWaiting&&!gateAtSensor&&!hazardWaiting&&state!=State::CrossApproach&&state!=State::SignalWait&&
               counts[LABEL_GATE]>0&&boxes[LABEL_GATE].width>=boxes[LABEL_GATE].height){
                p.stop=false;p.alignRoad=true;p.speedCap=std::min(p.speedCap,cfg.gateApproachSpeed);
                p.state="GATE_APPROACH";p.reason="APPROACH_SENSOR_BEFORE_GATE_WAIT";
            }
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
        // Once the mission owns the parking aisle, old entrance selection and
        // repeated bay/arrow splits must not keep pulling into the left edge.
        // All traffic stops above still take priority.
        if(parkingAisle) {
            if(state==State::ForkArmed||state==State::ForkSelect)enter(State::Lane);
            routeArmed=false;direction=0;entranceGatePassed=false;
            p.alignRoad=true;p.state=limited?"LIMIT_ZONE":"LANE";p.reason="PARKING_AISLE_CURRENT_ROAD";
            return p;
        }
        const bool leftSign=seen(LABEL_YLEFT)||seen(LABEL_LEFT),rightSign=seen(LABEL_YRIGHT);
        if(!parkingEntrance)entranceGatePassed=false;
        if(parkingEntrance&&entranceGatePassed&&(state==State::ForkSelect||state==State::ForkArmed)) {
            entranceGatePassed=false;routeArmed=false;
            p.completedDirection=-1;enter(State::Lane);cooldownUntil=now+cfg.cooldown;
            p.alignRoad=true;p.state=limited?"LIMIT_ZONE":"LANE";
            p.reason="PARKING_ENTRANCE_GATE_RELEASED_FIND_BAY";
            p.speedCap=std::min(p.speedCap,forkSpeed);
            return p;
        }
        // A confirmed mission entrance is a new maneuver, even if the old Y
        // task never declared completion and happens to have the same direction.
        if(confirmedRouteSign&&routeDirection&&(!routeArmed||routeDirection!=direction)&&
           (state==State::Lane||state==State::ForkArmed||state==State::ForkSelect)) {
            enter(State::Lane);routeArmed=false;cooldownUntil=now;
        }
        if(routeArmed&&routeDirection&&routeDirection!=direction&&(state==State::ForkArmed||state==State::ForkSelect)) {
            enter(State::Lane);routeArmed=false;
        }
        // PARK confirms intent only. Sign box size is diagnostic, never the
        // turn trigger: use a continuous near left opening in the current road.
        RaceEntranceEdge entrance;
        if(parkingEntrance){
            entrance=raceParkingEntranceEdge(raceBranchPath(geometryBin,track,-1,cfg,signRegions,true));
            p.parkingOpeningRow=entrance.openingRow;
            if(!routeArmed){
                entranceProgress.approachStep(dt,parkingEntryNear,newFrame,counts[LABEL_PARK]>0,
                    road.available&&std::isfinite(road.speed)?road.speed:0);
                p.parkingApproachCommandM=entranceProgress.approach;
                if(newFrame)entranceOpeningTime=entrance.opening?entranceOpeningTime+dt:0;
                const bool guided=entranceOpeningTime<.14f&&entranceProgress.fallbackReady(parkingEntryNear);
                if(entranceOpeningTime<.14f&&!guided){
                    p.speedCap=std::min(p.speedCap,forkSpeed);
                    p.state="PARKING_ENTRY_APPROACH";p.reason="FOLLOW_ROAD_UNTIL_NEAR_LEFT_OPENING";
                    return p;
                }
                direction=-1;routeArmed=true;entranceGuided=guided;
                if(guided){enter(State::ForkArmed);directionUntil=now+cfg.directionMemory;}
                else {enter(State::ForkSelect);handoffPwm=0;handoffTime=1;}
            }
            else if((!entranceGuided&&state==State::ForkArmed)||(state==State::Fault&&std::string_view(reviewReason)=="FORK_AI_STALE")){
                // Fresh data resumes edge tracking, never a new full-lock pulse.
                enter(State::ForkSelect);handoffPwm=0;handoffTime=1;
            }
            entranceProgress.pulseStep(dt,road.available&&std::isfinite(road.speed)?road.speed:0);
            p.parkingEntryFallback=entranceGuided;p.parkingApproachCommandM=entranceProgress.approach;
            // A raised/missed entrance gate is not a reason to stay in the
            // entrance forever. Confirm sustained local aisle tracking after
            // acquiring the road and finishing the bounded entry pulse.
            if(newFrame) {
                const bool acquired=(state==State::ForkSelect&&handoffTime>=.65f)||
                    (state==State::ForkArmed&&entranceGuided&&entranceProgress.pwm()<=0);
                const bool tracking=acquired&&
                    (!entranceGuided||entranceProgress.pwm()<=0)&&road.available&&road.roadReady&&road.speed>0&&
                    std::isfinite(road.offset)&&std::isfinite(road.heading)&&std::abs(road.offset)<64&&std::abs(road.heading)<60;
                entranceProgress.trackingTime=tracking?entranceProgress.trackingTime+dt:0;
            }
            if(entranceProgress.pulseDistance>=.30f&&entranceProgress.trackingTime>=.60f) {
                routeArmed=false;enter(State::Lane);cooldownUntil=now+cfg.cooldown;
                p.completedDirection=-1;p.alignRoad=true;p.state=limited?"LIMIT_ZONE":"LANE";
                p.reason="PARKING_ENTRANCE_AISLE_TRACKED_FIND_BAY";return p;
            }
        } else {entranceOpeningTime=0;entranceProgress={};entranceGuided=false;}
        p.parkingTurnCommit=parkingEntrance||parkingExit;
        if(state==State::Lane&&now>=cooldownUntil&&newFrame) {
            if(routeDirection!=0&&(geometry.splitRows>=cfg.minBranchRows||
                (confirmedRouteSign&&routeDirection==-1)||
                (routeDirection==-1&&leftSign&&!rightSign)||(routeDirection==1&&rightSign&&!leftSign))) {
                direction=routeDirection;routeArmed=true;directionUntil=now+cfg.directionMemory;enter(State::ForkArmed);
            }
            else if(routeDirection!=0) {}
            else if(leftSign&&rightSign) { direction=0; }
            else if(leftSign||rightSign) {
                direction=leftSign?-1:1;routeArmed=false;
                yProgress={};
                directionUntil=now+cfg.directionMemory; enter(State::ForkArmed);
            }
            else if(seen(LABEL_FLOW)&&geometry.splitRows>=cfg.minBranchRows) {
                direction=2;routeArmed=false;directionUntil=now+cfg.directionMemory;enter(State::ForkArmed);
            }
        }
        const bool ordinaryFork=!routeArmed&&!parkingExit&&(direction==-1||direction==1)&&
            (state==State::ForkArmed||state==State::ForkSelect);
        RacePath signedPath;
        bool usableTarget=false,oppositeRoad=false,nearHandoff=false,passedSignHandoff=false,selectedRoadMerged=false;
        if((!routeArmed||parkingExit)&&(direction==-1||direction==1)&&(state==State::ForkArmed||state==State::ForkSelect)) {
            yProgress.update(dt,newFrame,!parkingExit&&(counts[LABEL_YLEFT]||counts[LABEL_YRIGHT]||counts[LABEL_LEFT]),road);
            p.forkCommandDistance=yProgress.commandDistance;p.forkSignClearTime=yProgress.clearTime;p.forkPulsePhase=yProgress.pulsePhase;
            if(ordinaryFork) {
                if(state==State::ForkArmed&&road.available) {
                    yProgress.observeApproachSide(direction,road,newFrame);
                    const int label=direction>0?LABEL_YRIGHT:counts[LABEL_YLEFT]?LABEL_YLEFT:LABEL_LEFT;
                    const bool conflicting=direction<0?counts[LABEL_YRIGHT]:(counts[LABEL_YLEFT]||counts[LABEL_LEFT]);
                    if(conflicting||snap.sequence!=lastSequence||snap.captured!=lastCapture)yProgress.passingSign={};
                    else passedSignHandoff=yProgress.passingSign.update(direction,counts[label]?&boxes[label]:nullptr,
                        snap.captured,newFrame,dt,road,geometry.singleRoadAhead,cfg.confirmGap);
                    nearHandoff=(yProgress.sawApproachSide||passedSignHandoff)&&geometry.singleRoadAhead&&yProgress.pulsePhase>=1.4f&&
                        yProgress.commandDistance>=.25f&&yProgress.clearTime>0&&std::abs(road.offset)<64;
                }
                signedPath=raceBranchPath(geometryBin,track,direction,cfg,signRegions,true);
                float near=0,far=0,heading=0;
                const bool visible=corridorCenters(signedPath,near,far,heading);
                const bool sameTrack=state==State::ForkSelect&&!yProgress.waitingForRoad&&std::isfinite(targetCenter)&&
                    std::abs(far-targetCenter)<80;
                oppositeRoad=!sameTrack&&direction*(far-near)<-40&&direction*(far-320)<-20;
                const bool sideVisible=geometry.choiceRows>=cfg.minBranchRows||
                    (direction*(far-near)>40&&direction*(far-320)>20);
                const bool matchesPrior=std::isfinite(yProgress.lastUsableCenter)&&std::abs(far-yProgress.lastUsableCenter)<64;
                usableTarget=nearHandoff||(visible&&!oppositeRoad&&road.roadReady&&(sideVisible||matchesPrior||
                    (!yProgress.waitingForRoad&&(sameTrack||std::abs(near-320)<64))));
                // Once the chosen arm has actually been followed, a split only
                // beside the bonnet must not keep the fork speed cap for a lap.
                // Keep loss/reacquisition independent: the selected arm must
                // still be visible and continuous with its confirmed far end.
                selectedRoadMerged=visible&&sameTrack&&matchesPrior&&handoffTime>=.25f&&
                    geometry.singleRoadAhead;
                if(!usableTarget||(!nearHandoff&&!selectedRoadMerged&&geometry.choiceRows>=cfg.minBranchRows))yProgress.trackingTime=0;
                if(road.available&&(state==State::ForkSelect||yProgress.pulsePhase>=1.4f)) {
                    const bool wasWaiting=yProgress.waitingForRoad;
                    yProgress.observeTarget(usableTarget,nearHandoff?320+road.offset:far,newFrame,dt,road.speed,!nearHandoff);
                    p.forkLostDistance=yProgress.lostDistance;p.forkLostMovingTime=yProgress.lostMovingTime;
                    if(yProgress.waitingForRoad) {
                        // Keep evaluating the indicated branch with the pilot
                        // while stopped, even if an unrelated lane is visible.
                        p.path=std::move(signedPath);p.selectedBranch=!nearHandoff&&visible&&!oppositeRoad;
                        p.alignRoad=true;p.guideDirection=direction;p.guidePwm=0;
                        p.speedCap=std::min(p.speedCap,forkSpeed);
                        p.stop=true;p.state="FORK_HOLD";p.reason="SIGNED_FORK_WAIT_TARGET_ROAD";
                        return p;
                    }
                    if(wasWaiting&&!nearHandoff) {
                        enter(State::ForkSelect);targetCenter=far;handoffPwm=0;handoffTime=1;
                    }
                }
            }
            p.forkTrackingTime=yProgress.trackingTime;
            float exitNear=0,exitFar=0,exitHeading=0;
            const bool exitRoadContinuous=parkingExit&&state==State::ForkSelect&&handoffTime>=.65f&&
                corridorCenters(geometry,exitNear,exitFar,exitHeading)&&std::abs(exitFar-targetCenter)<96;
            const bool rejoined=parkingExit?yProgress.observeExitRoad(road,newFrame,dt,exitRoadContinuous,
                geometry.singleRoadAhead&&geometry.splitRows==0,direction,snap.captured):
                yProgress.complete()&&(!ordinaryFork||(usableTarget&&newFrame&&road.speed>0&&
                    (nearHandoff||selectedRoadMerged||geometry.choiceRows<cfg.minBranchRows)));
            if(rejoined) {
                p.completedDirection=direction;enter(State::Lane);cooldownUntil=now+cfg.cooldown;
                p.state=limited?"LIMIT_ZONE":"LANE";p.reason=parkingExit?"PARK_EXIT_CONFIRMED_ROAD_REJOINED":"SIGNED_FORK_CURRENT_ROAD_REJOINED";
                p.alignRoad=true;return p; // release only the fork cap; traffic limits remain
            }
            if(nearHandoff) {
                p.state=direction<0?"FORK_SEEK_LEFT":"FORK_SEEK_RIGHT";
                p.reason=passedSignHandoff&&!yProgress.sawApproachSide?"SIGNED_FORK_PASSED_SIGN_ROAD_HANDOFF":"SIGNED_FORK_NEAR_ROAD_HANDOFF";
                p.alignRoad=true;p.speedCap=std::min(p.speedCap,forkSpeed);return p;
            }
            if(parkingExit&&yProgress.commandDistance>=.80f) {
                p.stop=true;p.state="PARK_EXIT_REJOIN_HOLD";p.reason="EXIT_ROAD_NOT_REACQUIRED_WITHIN_TRAVEL";
                return p;
            }
        }
        if(state==State::ForkArmed) {
            if(direction==-1||direction==1) {
                p.guideDirection=direction;
                // Elevated field signs disappear with box bottoms at rows
                // 31..103, never reaching the old ground-object row145 gate.
                // Confirmed sign direction itself starts the road search.
                if(seekSince<0)seekSince=now;
                p.guidePwm=guidancePwm(routeArmed&&!parkingExit?now-seekSince:yProgress.pulsePhase);p.laneBlend=0;
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
                        if(entranceGuided)p.reason="PARK_APPROACH_DONE_BOUNDED_LEFT_GUIDANCE";
                        p.guidePwm=guidancePwm(routeArmed&&!parkingExit?elapsed:yProgress.pulsePhase);p.laneBlend=0;
                        RacePath routeCandidate;
                        if(!ordinaryFork)routeCandidate=raceBranchPath(geometryBin,track,direction,cfg,signRegions,true);
                        const auto &candidate=ordinaryFork?signedPath:routeCandidate;
                        float nearCenter=0,farCenter=0,heading=0,unionNear=0,unionFar=0,unionHeading=0;
                        const bool visible=corridorCenters(candidate,nearCenter,farCenter,heading);
                        const bool splitTarget=geometry.splitRows>=cfg.minBranchRows&&
                            corridorCenters(geometry,unionNear,unionFar,unionHeading)&&direction*(farCenter-unionFar)>24;
                        const bool sameTarget=visible&&targetVisibleTime>0&&std::isfinite(targetCenter)&&
                            std::abs(farCenter-targetCenter)<80;
                        // After the initial Y turn, a connected single road can
                        // be off-centre/tilted: requiring it to be aligned before
                        // allowing alignment control is a circular dependency.
                        const bool centeredAfterTurn=(!routeArmed||parkingExit)&&!oppositeRoad&&yProgress.pulsePhase>=1.0&&geometry.splitRows==0&&
                            std::abs(nearCenter-320)<130;
                        const bool entrySide=!parkingEntrance||(direction*(nearCenter-320)>-64&&direction*(farCenter-320)>20);
                        const bool target=visible&&entrySide&&(splitTarget||sameTarget||centeredAfterTurn||
                            (direction*(farCenter-nearCenter)>40&&direction*(farCenter-320)>20));
                        // One visible patch must not cancel the strong turn.
                        // Confirm a continuous connected target, never the opposite corridor.
                        if(target&&(!std::isfinite(targetCenter)||std::abs(farCenter-targetCenter)<80))targetVisibleTime+=dt;
                        else targetVisibleTime=0;
                        targetCenter=target?farCenter:std::numeric_limits<float>::quiet_NaN();
                        if(targetVisibleTime>=std::max(.18f,cfg.forkConfirmTime)) {
                            const float guide=p.guidePwm;enter(State::ForkSelect);handoffPwm=guide;p.buzzer=true;
                            if(ordinaryFork||parkingExit)targetCenter=farCenter;
                        } else if(elapsed>=cfg.forkMaxTime) {enter(State::Fault);reviewReason="SIGNED_ROAD_REACQUIRE_TIMEOUT";}
                        if(p.guidePwm<=0){p.guideDirection=0;p.alignRoad=true;p.reason="SIGN_PULSE_COMPLETE_ROAD_RECOVERY";}
                    }
                }
            }
        }
        if(state==State::ForkSelect) {
            // Follow the forward entry curve first. The near-only alignment
            // controller sees the still-straight approach and turns too late.
            p.alignRoad=!parkingEntrance||entranceGuided;
            p.guideDirection=direction==2?0:direction;p.guidePwm=0;
            p.speedCap=std::min(p.speedCap,forkSpeed);
            p.state=direction<0?"FORK_LEFT":direction==2?"FORK_STRAIGHT":"FORK_RIGHT"; p.reason=routeArmed?"MISSION_CONNECTED_BRANCH":"CONNECTED_BRANCH";
            p.path=ordinaryFork?std::move(signedPath):raceBranchPath(geometryBin,track,direction,cfg,signRegions,true); p.selectedBranch=true;
            if(parkingEntrance&&!entranceGuided){p.path=entrance.path;p.parkingEdgeEstimated=true;}
            if(!ordinaryFork&&(!parkingExit||handoffTime>=.65f)&&geometry.splitRows==0&&(!parkingEntrance||(entranceGuided&&entranceProgress.pwm()<=0)))
                p.path=raceBranchPath(geometryBin,track,2,cfg,signRegions,false);
            bool aligned=direction==2;
            if(direction!=2) {
                float nearCenter=0,farCenter=0,heading=0;
                const bool visible=corridorCenters(p.path,nearCenter,farCenter,heading);
                const bool continuous=visible&&!oppositeRoad&&(!std::isfinite(targetCenter)||std::abs(farCenter-targetCenter)<96);
                p.guidePwm=handoffPwm;p.laneBlend=std::clamp(handoffTime/(routeArmed?.65f:.25f),0.f,1.f);
                if(parkingEntrance&&!entranceGuided){p.guidePwm=0;p.laneBlend=1;}
                if(entranceGuided)p.guidePwm=std::min(p.guidePwm,entranceProgress.pwm());
                if(continuous){handoffTime+=dt;targetCenter=farCenter;}
                else {
                    // Target was already acquired. Keep recovering the current
                    // road; do not turn again just because its far end moved.
                    targetCenter=std::numeric_limits<float>::quiet_NaN();alignedTime=0;
                    p.selectedBranch=false;p.path={};p.laneBlend=1;p.guidePwm=0;p.guideDirection=0;p.alignRoad=true;
                    p.reason="TARGET_ROAD_LOST_CURRENT_LANE_RECOVERY";return p;
                }
                if(continuous&&(!ordinaryFork||!road.available||usableTarget)&&p.laneBlend>=1&&std::abs(nearCenter-320)<40&&std::abs(heading)<12)alignedTime+=dt;
                else alignedTime=0;
                aligned=alignedTime>=.18f;
                if(continuous)p.reason=parkingEntrance&&!entranceGuided?"PARKING_LEFT_EDGE_OFFSET_TRACKING":p.laneBlend<1?"TARGET_ROAD_GRADUAL_HANDOFF":"COMMITTED_ROAD_CENTER_TRACKING";
            }
            if(p.path.left.size()<8) { enter(State::Fault); reviewReason="BRANCH_PATH_LOST"; }
            else if(now-stateSince>cfg.forkMaxTime) {
                if(continueGeometry){stateSince=now;handoffPwm=0;handoffTime=1;p.guidePwm=0;p.laneBlend=1;
                    p.reason="ACQUIRED_ROAD_ALIGNMENT_CONTINUE";}
                else {enter(State::Fault);reviewReason="FORK_TIMEOUT";}
            }
            // A signed fork with applied feedback completes only in yProgress,
            // after the car actually receives forward commands on the recovered road.
            else if(!parkingExit&&now-stateSince>=cfg.forkMinTime&&geometry.splitRows==0&&aligned&&
                    (!ordinaryFork||(!road.available&&geometry.choiceRows<cfg.minBranchRows))) {
                exitTime+=dt;
                if(exitTime>=cfg.forkExitTime) {
                    p.completedDirection=direction;
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
            enter(State::ForkArmed);seekSince=now-(routeArmed?0:1.4);
            p.stop=false;p.selectedBranch=false;p.path={};p.guideDirection=direction==2?0:direction;
            yProgress.pulsePhase=std::max(yProgress.pulsePhase,1.4f);
            p.guidePwm=p.guideDirection?guidancePwm(routeArmed&&!parkingExit?now-seekSince:yProgress.pulsePhase):0;p.laneBlend=p.guideDirection?0:1;
            if(parkingEntrance){p.guidePwm=0;p.guideDirection=0;p.laneBlend=1;p.alignRoad=true;}
            if(p.guidePwm<=0){p.guideDirection=0;p.alignRoad=true;}
            p.state="FORK_RECOVER";p.reason="GEOMETRY_WARNING_CONTINUE";
        }
        else if(state==State::Fault) { p.stop=true; p.state="REVIEW_STOP"; p.reason=reviewReason; }
        return p;
    }
};
