#pragma once
#include "race_manager.hpp"
#include "race_obstacle_plan.hpp"

// 只规划推理原图中的可见通路，不把旧检测框叠加到新图上，不改写原始 Track。
// 输出用于预览和取证；当前实跑仍使用 RaceManager 的障碍停车策略。
class RaceObstaclePreview {
    struct Target { cv::Rect box;int count=0,id=0; };
    struct Row { int y,left,right; };
    RaceNavigationConfig cfg;
    std::vector<Target> targets;
    uint64_t lastSequence=0;
    double lastCapture=-1,clearSince=-1;
    int nextId=0,activeId=0,lockedDirection=0;
    cv::Rect lastPrimary;
    bool nearSeen=false;
    RaceObstaclePreviewPlan latest;

    static float match(const cv::Rect &a,const cv::Rect &b) {
        const auto overlap=a&b;
        const float iou=float(overlap.area())/std::max(1,a.area()+b.area()-overlap.area());
        return iou;
    }
    static std::vector<Row> road(const cv::Mat &bin,const Track &track,const RaceSnapshot &s,
                                 const RaceConfig &race,const RaceNavigationConfig &cfg,
                                 const std::vector<cv::Rect> &obstacles,std::string &reason,RaceObstaclePreviewPlan &diagnostics) {
        std::vector<Row> rows;
        if(bin.type()!=CV_8UC1||bin.size()!=cv::Size(640,480)||
           track.pointsEdgeLeft.empty()||track.pointsEdgeRight.empty()) { reason="OBS_LANE_UNAVAILABLE";return rows; }
        const int start=track.pointsEdgeLeft.front().x;
        int low=track.pointsEdgeLeft.front().y,high=track.pointsEdgeRight.front().y;
        if(start<300||start>470||low<0||high>639||low>=high) { reason="OBS_NEAR_LANE_UNAVAILABLE";return rows; }
        auto masks=raceSignRegions(s,race);
        // 遮罩只合并被本帧物体框覆盖的内部断口，不越过两侧道路边界。
        for(const auto &r:obstacles)masks.push_back(r&cv::Rect(0,0,640,480));
        if(race.verifyCrosswalkGeometry&&s.crosswalk.accepted)
            for(const auto &r:s.crosswalk.bars)masks.push_back(r);
        for(int y=start;y>=cfg.minLookaheadRow;y-=cfg.rowStep) {
            const auto blocks=raceRoadBlocks(bin,y,rows.empty()?0:low,rows.empty()?639:high,race,masks,3);
            if(blocks.empty()) { reason="OBS_LANE_GAP";break; }
            if(blocks.size()!=1) { reason="OBS_BRANCH_AMBIGUOUS";rows.clear();return rows; }
            const int rawLeft=blocks[0].first,rawRight=blocks[0].second;
            const bool leftSeen=rawLeft>6,rightSeen=rawRight<632;
            // 未入镜的一侧保持“未知”。仅把画面内的白色通路作为局部预览范围，
            // 不外推道路边界，也不据此解除实跑的障碍停车。
            const int l=std::max(8,rawLeft),r=std::min(630,rawRight);
            if(r-l<race.minBranchWidth) { reason="OBS_LANE_TOO_NARROW";break; }
            if(!rows.empty()&&(std::abs(l-low)>32||std::abs(r-high)>32)) { reason="OBS_LANE_JUMP";break; }
            rows.push_back({y,l,r});low=l;high=r;
            diagnostics.visibleLeft.emplace_back(y,l);diagnostics.visibleRight.emplace_back(y,r);
            diagnostics.leftBoundarySeen.push_back(leftSeen);diagnostics.rightBoundarySeen.push_back(rightSeen);
            if(!leftSeen||!rightSeen) { diagnostics.visibilityLimited=true;++diagnostics.clippedRows; }
        }
        if(rows.size()<size_t(cfg.minRows)) { if(reason.empty())reason="OBS_SHORT_LANE";rows.clear(); }
        if(!rows.empty()) { diagnostics.visibleNearRow=rows.front().y;diagnostics.visibleFarRow=rows.back().y; }
        return rows;
    }
    bool solve(const std::vector<Row> &rows,const std::vector<cv::Rect> &obstacles,
               const cv::Rect &primary,int locked,RaceObstaclePreviewPlan &out) const {
        if(rows.empty())return false;
        constexpr float inf=1e20f;
        // 半像素网格避免 2 行×0.8 斜率被整数截断成 1 像素后实际只能用 0.5。
        // 搜索和输出都保留小数，不通过放宽配置的斜率来生成路线。
        constexpr int scale=2,states=640*scale;
        std::array<float,states> previous,current;previous.fill(inf);
        std::vector<std::array<int16_t,states>> parents(rows.size());
        auto free=[&](const Row &row,float x) {
            const float half=(row.right-row.left)*(cfg.previewVehicleWidthFraction*.5f+cfg.previewClearanceFraction);
            if(x<row.left+half||x>row.right-half)return false;
            for(const auto &o:obstacles) {
                const int pad=std::max(3,o.width/4);
                if(row.y>=o.y-pad&&row.y<=o.br().y+pad&&x>o.x-half&&x<o.br().x+half)return false;
            }
            if(locked&&row.y>=primary.y-3&&row.y<=primary.br().y+3) {
                if(locked<0&&x>=primary.x)return false;
                if(locked>0&&x<=primary.br().x)return false;
            }
            return true;
        };
        for(size_t i=0;i<rows.size();++i) {
            current.fill(inf);parents[i].fill(-1);
            const auto &row=rows[i];const float middle=(row.left+row.right)*.5f;
            const int maxStep=i?int((rows[i-1].y-row.y)*cfg.maxLateralSlope*scale+1e-5f):0;
            for(int node=row.left*scale;node<=row.right*scale;++node) {
                const float x=float(node)/scale;
                if(!free(row,x))continue;
                if(i==0) { if(std::abs(x-320)<=cfg.startTolerance)current[node]=(x-320)*(x-320);continue; }
                for(int pn=std::max(0,node-maxStep);pn<=std::min(states-1,node+maxStep);++pn) {
                    const float px=float(pn)/scale;
                    const float value=previous[pn]+.02f*(x-middle)*(x-middle)+.8f*(x-px)*(x-px);
                    if(value<current[node]) { current[node]=value;parents[i][node]=pn; }
                }
            }
            previous=current;
        }
        int node=int(std::min_element(previous.begin(),previous.end())-previous.begin());
        if(previous[node]>=inf/2)return false;
        out.centers.resize(rows.size());
        for(size_t i=rows.size();i-->0;) {
            const float x=float(node)/scale;
            out.centers[i]={rows[i].y,x};
            if(!free(rows[i],x))return false;
            if(i) { node=parents[i][node];if(node<0)return false; }
        }
        // 只输出已经逐行通过碰撞检查的路径；不做会切入障碍物的事后平滑。
        for(size_t i=0;i<rows.size();++i) {
            const float half=std::max(1.f,(rows[i].right-rows[i].left)*cfg.previewVehicleWidthFraction*.5f);
            out.left.push_back({rows[i].y,out.centers[i].y-half});
            out.right.push_back({rows[i].y,out.centers[i].y+half});
        }
        auto closest=std::min_element(out.centers.begin(),out.centers.end(),[&](const RacePreviewPoint &a,const RacePreviewPoint &b) {
            const int y=std::clamp(primary.y+primary.height/2,rows.back().y,rows.front().y);
            return std::abs(a.x-y)<std::abs(b.x-y);
        });
        out.direction=closest->y<primary.x+primary.width/2?-1:1;
        return true;
    }
public:
    explicit RaceObstaclePreview(const RaceNavigationConfig &config={}):cfg(config) { cfg.validate(); }
    RaceObstaclePreviewPlan update(const RaceSnapshot &s,double now,const cv::Mat &matchedBinary,
                                   const Track &matchedTrack,const RaceConfig &race) {
        RaceObstaclePreviewPlan out;out.sequence=s.sequence;out.captured=s.captured;
        if(cfg.obstacleMode=="off") { out.state="OBS_PREVIEW_OFF";return out; }
        out.fresh=!s.failed&&s.sequence&&std::isfinite(s.captured)&&now>=s.captured&&now-s.captured<=race.maxDetectionAge;
        if(!out.fresh) { targets.clear();clearSince=-1;out.state="OBS_AI_STALE";return out; }
        if(s.sequence==lastSequence)return latest;
        if(s.sequence<lastSequence||s.captured<=lastCapture) { targets.clear();clearSince=-1;out.state="OBS_NONMONOTONIC";return out; }
        if(s.image.type()!=CV_8UC3||s.image.size()!=cv::Size(640,480)) { targets.clear();clearSince=-1;out.state="OBS_NO_MATCHED_IMAGE";return out; }
        const bool gap=lastCapture<0||s.captured-lastCapture>race.confirmGap;
        lastSequence=s.sequence;lastCapture=s.captured;
        std::vector<cv::Rect> candidates;
        for(const auto &d:s.detections) {
            if((d.type==LABEL_CONE||d.type==LABEL_PERSON)&&reviewRaceDetection(d,s.image,race,&s.crosswalk).accepted)
                candidates.emplace_back(d.x,d.y,d.width,d.height);
        }
        if(candidates.size()>32) { out.hasTarget=true;out.state="OBS_TOO_MANY_TARGETS";targets.clear();latest=out;return out; }
        if(candidates.empty()&&!activeId) { targets.clear();out.stop=false;latest=out;return out; }
        std::string laneIssue;
        const auto rows=road(matchedBinary,matchedTrack,s,race,cfg,candidates,laneIssue,out);
        if(rows.empty()) { targets.clear();clearSince=-1;out.state=laneIssue;out.hasTarget=!candidates.empty()||activeId;latest=out;return out; }
        std::vector<cv::Rect> onRoad;
        for(const auto &r:candidates) {
            const int foot=std::min(r.br().y-1,rows.front().y);
            const auto row=std::min_element(rows.begin(),rows.end(),[&](const Row &a,const Row &b){return std::abs(a.y-foot)<std::abs(b.y-foot);});
            if(r.br().y<rows.back().y)continue;
            if(r.x<row->right&&r.br().x>row->left)onRoad.push_back(r);
        }
        if(gap)targets.clear();
        std::vector<Target> updated;
        std::vector<bool> used(targets.size(),false);
        for(const auto &r:onRoad) {
            int index=-1;float best=.15f;
            for(size_t i=0;i<targets.size();++i)if(!used[i]) {
                const float score=match(r,targets[i].box);
                if(score>best) { best=score;index=int(i); }
            }
            const bool activeAlreadyUsed=std::any_of(updated.begin(),updated.end(),[&](const Target &t){return t.id==activeId;});
            if(index<0)updated.push_back({r,1,activeId&&!activeAlreadyUsed&&match(r,lastPrimary)>.15f?activeId:++nextId});
            else { used[index]=true;updated.push_back({r,std::min(race.confirmFrames,targets[index].count+1),targets[index].id}); }
        }
        targets=std::move(updated);out.obstacles=onRoad;
        for(const auto &t:targets)if(t.count>=race.confirmFrames)++out.confirmedTargets;
        const Target *primary=nullptr;
        for(const auto &t:targets)if(t.id==activeId)primary=&t;
        if(!primary&&activeId) {
            out.hasTarget=true;out.primary=lastPrimary;
            if(!targets.empty()) { clearSince=-1;out.state="OBS_TARGET_CHANGED"; }
            else if(!nearSeen) { clearSince=-1;out.state="OBS_LOST_BEFORE_NEAR"; }
            else {
                if(clearSince<0)clearSince=s.captured;
                out.state="OBS_CLEARING";
                if(s.captured-clearSince>=race.clearTime) {
                    activeId=lockedDirection=0;nearSeen=false;out.hasTarget=false;out.stop=false;out.state="OBS_CLEAR_OBSERVED";
                }
            }
            latest=out;return out;
        }
        if(!primary)for(const auto &t:targets)
            if(!primary||t.box.br().y>primary->box.br().y)primary=&t;
        if(!primary) { out.state="OBS_NO_TARGET";out.stop=false;latest=out;return out; }
        out.hasTarget=true;out.primary=primary->box;clearSince=-1;
        if(primary->count<race.confirmFrames) { out.state="OBS_CONFIRMING";latest=out;return out; }
        activeId=primary->id;lastPrimary=primary->box;
        if(primary->box.br().y>=race.crossPassedRow)nearSeen=true;
        if(primary->box.br().y>rows.front().y) { out.state="OBS_NEAR_REGION_UNSEEN";latest=out;return out; }
        if(!solve(rows,onRoad,primary->box,lockedDirection,out)) {
            out.state="OBS_NO_CLEAR_PATH";out.centers.clear();out.left.clear();out.right.clear();latest=out;return out;
        }
        lockedDirection=out.direction;out.candidate=true;out.stop=out.visibilityLimited;
        if(out.visibilityLimited) {
            out.state=out.direction<0?"OBS_VISIBLE_PASS_LEFT_PREVIEW":"OBS_VISIBLE_PASS_RIGHT_PREVIEW";
            out.reason="VISIBLE_CORRIDOR_ONLY_UNCALIBRATED";
        } else out.state=out.direction<0?"OBS_BYPASS_LEFT_PREVIEW":"OBS_BYPASS_RIGHT_PREVIEW";
        latest=out;return out;
    }
};
