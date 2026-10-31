#pragma once
#include "race_bypass_control.hpp"
#include "race_manager.hpp"
#include "trackview_pilot.hpp"

// Design reference: 2025 iCAR FsmBusy virtual-lane / Bezier repair approach.
// Independently implemented against this program's matched-frame and stop APIs.
// It uses image-space width ratios, not accumulated visual odometry. Its visual
// exit is an experimental heuristic and NEVER sets metric rear_passed=true.
class RaceVirtualBoundary {
    struct Target {cv::Rect box;int count;double seen,modelSeen;cv::Mat patch;};
    struct Row {int y,l,r;float width;};
    RaceNavigationConfig cfg;
    std::vector<Target> targets;
    RaceObstaclePreviewPlan latest;
    uint64_t sequence=0;
    double capture=-1,missingTime=0,returnTime=0;
    bool active=false,nearSeen=false,wasAutomatic=false,interrupted=false,returning=false;
    int direction=0;
    cv::Rect primary;
    float heldOffset=0;
    float widthBase=0,widthSlope=0;
    double widthCaptured=-1;
    bool widthHeld=false,widthUncertain=false,reacquiring=false;
    int reacquireVisibleFrames=0;
    static float overlap(cv::Rect a,cv::Rect b) {
        const auto i=a&b;return float(i.area())/std::max(1,a.area()+b.area()-i.area());
    }
    static float smooth(float t) {t=std::clamp(t,0.f,1.f);return t*t*(3-2*t);}
    static constexpr float vehicleFraction=.18f/.45f,marginFraction=.005f/.45f;
    static constexpr double holdCommandDistance=.36,returnCommandDistance=.12;
    static constexpr int nearRow=205;
    static constexpr float entryLeadRows=140,farBlendRows=80,geometryEpsilon=.001f;

    // Recover only an already confirmed object in the CURRENT matched image.
    // Keep the last AI template fixed to avoid recursively learning background.
    // A missing model box alone must neither erase the route nor mean "passed".
    static bool recoverTarget(const Target &t,const cv::Mat &image,double now,cv::Rect &box) {
        if(t.patch.empty()||now-t.modelSeen>1.0)return false;
        cv::Scalar mean,stddev;cv::meanStdDev(t.patch,mean,stddev);
        if(std::max({stddev[0],stddev[1],stddev[2]})<12)return false;
        const int pad=std::max(18,t.box.width/2);
        const auto roi=(t.box+cv::Size(2*pad,2*pad)-cv::Point(pad,pad))&cv::Rect(0,0,image.cols,image.rows);
        double best=.82;bool found=false;
        for(double scale:{.85,1.,1.15,1.35,1.6}) {
            cv::Mat patch,result;cv::resize(t.patch,patch,{},scale,scale);
            if(patch.cols>roi.width||patch.rows>roi.height||patch.cols<6||patch.rows<6)continue;
            cv::matchTemplate(image(roi),patch,result,cv::TM_CCOEFF_NORMED);
            double score;cv::Point at;cv::minMaxLoc(result,nullptr,&score,nullptr,&at);
            const cv::Rect candidate(roi.tl()+at,patch.size());
            if(std::isfinite(score)&&score>best&&overlap(t.box,candidate)>.1f){best=score;box=candidate;found=true;}
        }
        return found;
    }

    std::vector<Row> road(const RaceSnapshot &s,const cv::Mat &bin,const Track &track,
                         const RaceConfig &race,const std::vector<cv::Rect> &boxes,RaceObstaclePreviewPlan &p) {
        widthHeld=widthUncertain=false;
        std::vector<Row> rows;
        if(bin.type()!=CV_8UC1||bin.size()!=cv::Size(320,240)||track.pointsEdgeLeft.empty()||track.pointsEdgeRight.empty())return rows;
        auto masks=raceSignRegions(s,race);
        for(auto b:boxes)masks.push_back((b+cv::Size(24,24)-cv::Point(12,12))&cv::Rect(0,0,320,240));
        int low=0,high=319;
        auto nearCorridorCoversCones=[&](int farRow) {
            return active&&rows.size()>=size_t(cfg.minRows)&&rows.front().y-rows.back().y>=70&&
                std::all_of(boxes.begin(),boxes.end(),[&](const auto &b){return b.y>=farRow;});
        };
        for(int y=std::min(185,track.pointsEdgeLeft.front().x);y>=cfg.minLookaheadRow;y-=cfg.rowStep) {
            const auto blocks=raceRoadBlocks(bin,y,low,high,race,masks,3);
            if(blocks.size()!=1){
                // During a near-cone pass/return, a bend beyond the current
                // tracking horizon must not discard an adequate connected
                // near corridor. Never truncate before a visible cone.
                if(nearCorridorCoversCones(y)){p.visibilityLimited=true;break;}
                p.reason=blocks.empty()?"VIRTUAL_LANE_GAP":"VIRTUAL_BRANCH_AMBIGUOUS";rows.clear();return rows;
            }
            low=blocks.front().first;high=blocks.front().second;
            if(high-low<35){
                if(nearCorridorCoversCones(y)){p.visibilityLimited=true;break;}
                p.reason="VIRTUAL_LANE_NARROW";rows.clear();return rows;
            }
            rows.push_back({y,low,high,float(high-low)});
        }
        // Width in a flat-road image changes approximately linearly with row.
        // Fit only genuinely visible, unoccluded edges. At clipped near rows,
        // use the larger predicted width for the projected vehicle envelope;
        // do not assume the cropped visible strip is the complete road width.
        if(rows.empty())return rows;
        auto widthRows=rows;
        auto visibleWidth=[&](const Row &r){return r.l>3&&r.r<316&&std::none_of(masks.begin(),masks.end(),[&](const auto &b){
            return b.contains({r.l,r.y})||b.contains({r.r,r.y});});};
        int visibleCount=int(std::count_if(rows.begin(),rows.end(),visibleWidth));
        // When the car is angled, both edges may only be visible ABOVE the
        // steering horizon. Use those actual pixels for width fitting too.
        for(int y=rows.back().y-cfg.rowStep;y>=5&&visibleCount<12;y-=cfg.rowStep) {
            auto blocks=raceRoadBlocks(bin,y,low,high,race,masks,3);
            if(blocks.size()!=1||blocks.front().second-blocks.front().first<35)break;
            low=blocks.front().first;high=blocks.front().second;widthRows.push_back({y,low,high,float(high-low)});
            if(visibleWidth(widthRows.back()))++visibleCount;
        }
        float n=0,sy=0,sw=0,syy=0,syw=0;
        for(const auto &r:widthRows)if(visibleWidth(r)) {
            ++n;sy+=r.y;sw+=r.width;syy+=r.y*r.y;syw+=r.y*r.width;
        }
        const float den=n*syy-sy*sy;
        float slope=den>=1?(n*syw-sy*sw)/den:0,base=n>0?(sw-slope*sy)/n:0;
        float residual=0;for(const auto &r:widthRows)if(visibleWidth(r))residual+=std::pow(r.width-base-slope*r.y,2);
        // A few rows across a bend can give a steep or negative "perspective"
        // fit and inflate the car wider than the entire visible passage.
        // Learn only a plausible fit; bridge short occlusion from that fit,
        // never from the previous path and never refresh its age by reuse.
        const bool fitted=n>=8&&den>=1&&slope>=0&&slope<=2&&base+slope*185>=90&&
            base+slope*185<=480&&std::sqrt(residual/std::max(1.f,n))<=12;
        if(fitted){widthBase=base;widthSlope=slope;widthCaptured=s.captured;}
        else if(widthCaptured>=0&&s.captured-widthCaptured<=2.0) {
            base=widthBase;slope=widthSlope;widthHeld=true;
        } else {
            // Competition policy: missing scale is not a reason to stop on
            // a currently visible road. Use visible pixels as a steering cue,
            // not as a calibrated full-body containment certificate.
            widthUncertain=true;base=slope=0;
        }
        for(auto &r:rows) {
            const bool l=r.l>3,rr=r.r<316;
            if(!l||!rr){r.width=std::max(r.width,base+slope*r.y);p.visibilityLimited=true;++p.clippedRows;}
            p.visibleLeft.emplace_back(r.y,r.l);p.visibleRight.emplace_back(r.y,r.r);
            p.leftBoundarySeen.push_back(l);p.rightBoundarySeen.push_back(rr);
        }
        if(rows.size()<size_t(cfg.minRows)||rows.front().y<135||rows.front().y-rows.back().y<50) {
            p.reason="VIRTUAL_SHORT_VISIBLE_LANE";rows.clear();return rows;
        }
        p.visibleNearRow=rows.front().y;p.visibleFarRow=rows.back().y;return rows;
    }
public:
    explicit RaceVirtualBoundary(RaceNavigationConfig c={}):cfg(c){cfg.validate();}
    void reset(){*this=RaceVirtualBoundary(cfg);}
    bool engaged()const{return active||interrupted;}
    RaceObstaclePreviewPlan update(const RaceSnapshot &s,double now,const cv::Mat &bin,const Track &track,
                                  const RaceConfig &race,bool automatic,float previousCommandSpeed) {
        if(wasAutomatic&&!automatic&&active)interrupted=true;
        wasAutomatic=automatic;
        RaceObstaclePreviewPlan p;p.sequence=s.sequence;p.captured=s.captured;p.virtualBoundary=true;
        p.reason="VIRTUAL_NO_CONE";p.state="VIRTUAL_IDLE";
        p.fresh=!s.failed&&s.sequence&&std::isfinite(s.captured)&&now>=s.captured&&now-s.captured<=std::min(.18f,race.maxDetectionAge);
        if(!p.fresh){p.hasTarget=active;p.state="VIRTUAL_AI_WAIT";p.reason="FRESH_MATCHED_AI_REQUIRED";return p;}
        if(s.sequence==sequence) {p=latest;p.fresh=true;return p;}
        if(s.sequence<sequence||s.captured<=capture){p.hasTarget=true;p.state="VIRTUAL_FRAME_ORDER";p.reason="MONOTONIC_FRAMES_REQUIRED";return p;}
        const double gap=capture<0?0:s.captured-capture;sequence=s.sequence;capture=s.captured;
        auto finish=[&](const std::string &state,const std::string &reason){p.state=state;p.reason=reason;latest=p;return p;};
        if(interrupted){p.hasTarget=true;return finish("VIRTUAL_FAULT","OPERATOR_INTERRUPTED_RESTART_REQUIRED");}
        if(s.image.type()!=CV_8UC3||s.image.size()!=cv::Size(320,240)){p.hasTarget=active;return finish("VIRTUAL_IMAGE_WAIT","MATCHED_IMAGE_REQUIRED");}
        std::vector<cv::Rect> boxes;
        for(const auto &d:s.detections)if(reviewRaceDetection(d,s.image,race,&s.crosswalk).accepted) {
            if(d.type==LABEL_PERSON){p.hasTarget=true;return finish("VIRTUAL_PERSON_STOP","PERSON_PRIORITY");}
            if(d.type==LABEL_CONE)boxes.emplace_back(d.x,d.y,d.width,d.height);
        }
        // A single missed inference previously cleared every confirmation and
        // fed empty edges into v6. Track confirmed cones visually for <=1s;
        // all recovered boxes still go through fresh road/envelope validation.
        std::vector<cv::Rect> recovered;
        for(const auto &t:targets)if(t.count>=race.confirmFrames&&
            std::none_of(boxes.begin(),boxes.end(),[&](auto b){return overlap(t.box,b)>.1f;})) {
            cv::Rect b;if(recoverTarget(t,s.image,s.captured,b)){boxes.push_back(b);recovered.push_back(b);}
        }
        p.obstacles=boxes;p.hasTarget=active||!boxes.empty();
        if(!p.hasTarget){
            // Prime the short-lived width reference while both lane edges are
            // visible, before the cone manoeuvre turns one edge out of view.
            RaceObstaclePreviewPlan reference;road(s,bin,track,race,{},reference);
            targets.erase(std::remove_if(targets.begin(),targets.end(),[&](const auto &t){return s.captured-t.seen>race.confirmGap;}),targets.end());
            p.stop=false;return finish("VIRTUAL_IDLE","NORMAL_LANE");
        }
        if(boxes.size()>8)return finish("VIRTUAL_TOO_MANY_CONES","TARGET_LIMIT");
        auto rows=road(s,bin,track,race,boxes,p);
        p.virtualWidthUncertain=widthUncertain;
        if(rows.empty())return finish("VIRTUAL_PATH_WAIT",p.reason);
        std::vector<cv::Rect> onRoad;
        for(auto b:boxes) {
            const int foot=std::clamp(b.br().y,rows.back().y,rows.front().y);
            const auto r=std::min_element(rows.begin(),rows.end(),[&](auto a,auto c){return std::abs(a.y-foot)<std::abs(c.y-foot);});
            if(b.br().y>=rows.back().y&&b.x<r->r&&b.br().x>r->l)onRoad.push_back(b);
        }
        p.obstacles=onRoad;
        if(!active&&onRoad.empty()){p.hasTarget=false;p.stop=false;return finish("VIRTUAL_IDLE","OFFROAD_CONES");}
        std::vector<Target> next;std::vector<bool> used(targets.size());
        for(auto b:onRoad) {
            int best=-1;float score=.1f;
            for(size_t i=0;i<targets.size();++i)if(!used[i]&&overlap(b,targets[i].box)>score){score=overlap(b,targets[i].box);best=int(i);}
            const bool visual=std::find(recovered.begin(),recovered.end(),b)!=recovered.end();
            if(best<0)next.push_back({b,1,s.captured,s.captured,s.image(b).clone()});
            else {
                used[best]=true;auto t=targets[best];
                t.count=s.captured-t.seen>race.confirmGap?1:std::min(race.confirmFrames,t.count+(visual?0:1));
                t.box=b;t.seen=s.captured;
                if(!visual){t.modelSeen=s.captured;t.patch=s.image(b).clone();}
                next.push_back(std::move(t));
            }
        }
        const size_t currentTargets=next.size();
        for(size_t i=0;i<targets.size();++i)if(!used[i]&&s.captured-targets[i].seen<=race.confirmGap)next.push_back(targets[i]);
        targets=std::move(next);
        for(size_t i=0;i<currentTargets;++i)if(targets[i].count>=race.confirmFrames)++p.confirmedTargets;
        float returnScale=1;
        if(active&&nearSeen&&onRoad.empty()&&(widthHeld||widthUncertain||reacquiring)) {
            // On a bend or pitch transition, stop holding an outward offset
            // until a scale timer expires. Follow the CURRENT visible road
            // and blend the remaining offset away while continuing to move.
            if(!reacquiring){reacquiring=true;returnTime=0;reacquireVisibleFrames=0;}
            if(automatic&&previousCommandSpeed>0&&gap>0&&gap<=.25)returnTime+=gap*previousCommandSpeed;
            const float blend=1-smooth(float(returnTime/returnCommandDistance));
            p.direction=direction;p.primary=primary;
            for(const auto &r:rows) {
                const float center=std::clamp((r.l+r.r)*.5f+std::clamp(heldOffset,-.10f,.10f)*(r.r-r.l)*blend,20.f,299.f);
                const float band=std::min({40.f,center-4,315-center});
                p.centers.push_back({r.y,center});p.left.push_back({r.y,center-band});p.right.push_back({r.y,center+band});
            }
            p.candidate=true;p.stop=false;p.visibilityLimited=true;p.virtualLineCrossing=true;
            reacquireVisibleFrames=(!widthHeld&&!widthUncertain)?reacquireVisibleFrames+1:0;
            if(returnTime>=returnCommandDistance&&reacquireVisibleFrames>=5) {
                active=nearSeen=reacquiring=false;direction=0;heldOffset=0;
                p.hasTarget=false;p.virtualVisualReturn=true;
                return finish("VIRTUAL_VISUAL_EXIT","CURRENT_ROAD_REACQUIRED_NOT_REAR_VERIFIED");
            }
            return finish("VIRTUAL_REACQUIRE","CURRENT_VISIBLE_ROAD_RECOVERY");
        }
        if(!onRoad.empty()) {
            if(reacquiring&&std::none_of(onRoad.begin(),onRoad.end(),[&](const auto &b){return overlap(primary,b)>.1f;})) {
                // A new cone must not inherit the old cone's near/pass phase
                // or locked bypass side when lane recovery is interrupted.
                active=nearSeen=false;direction=0;heldOffset=0;
            }
            reacquiring=false;reacquireVisibleFrames=0;
            missingTime=returnTime=0;returning=false;
            if(p.confirmedTargets<int(currentTargets))return finish("VIRTUAL_CONFIRMING","CONFIRM_ALL_CONES");
            const auto chosen=*std::max_element(onRoad.begin(),onRoad.end(),[](auto a,auto b){return a.area()<b.area();});
            if(active&&!nearSeen&&overlap(primary,chosen)<.02f)return finish("VIRTUAL_TARGET_CHANGED","REACQUIRE_OR_RESTART");
            primary=chosen;p.primary=primary;
            const int foot=std::clamp(primary.br().y,rows.back().y,rows.front().y);
            const auto r=std::min_element(rows.begin(),rows.end(),[&](auto a,auto b){return std::abs(a.y-foot)<std::abs(b.y-foot);});
            const int selected=primary.x-r->l<=r->r-primary.br().x?1:-1;
            if(!active)direction=selected;
            if(automatic){active=true;nearSeen|=primary.br().y>=nearRow;}
        } else {
            p.primary=primary;
            if(!nearSeen)return finish("VIRTUAL_LOST_BEFORE_NEAR","NO_BLIND_EARLY_RELEASE");
            // Current fresh road edges remain mandatory throughout the hold and
            // return. These command-distance phases are visual-exit heuristics,
            // explicitly NOT measured travel or proof of rear clearance.
            if(automatic&&previousCommandSpeed>0&&gap>0&&gap<=.25) {
                if(!returning)missingTime+=gap*previousCommandSpeed;else returnTime+=gap*previousCommandSpeed;
            }
            if(missingTime>=holdCommandDistance)returning=true;
            returnScale=returning?1-smooth(float(returnTime/returnCommandDistance)):1;
            p.virtualVisualReturn=returning&&returnTime>=returnCommandDistance;
            if(p.virtualVisualReturn){active=false;nearSeen=false;direction=0;heldOffset=0;p.hasTarget=false;p.stop=false;return finish("VIRTUAL_VISUAL_EXIT","VISUAL_EXIT_HEURISTIC_NOT_REAR_VERIFIED");}
        }
        p.direction=direction;float peak=0;
        bool softRoad=widthUncertain;
        for(const auto &r:rows) {
            const float half=r.width*(vehicleFraction*.5f+marginFraction),pad=std::max(6.f,r.width*.025f);
            float lo=std::max({r.l+half,float(r.l+15),19.f}),hi=std::min({r.r-half,float(r.r-15),300.f});
            for(auto b:onRoad)if(r.y>=b.y-pad&&r.y<=b.br().y+pad) {
                if(direction>0)lo=std::max(lo,b.br().x+pad+half);
                else hi=std::min(hi,b.x-pad-half);
            }
            if(lo>hi+geometryEpsilon)softRoad=true;
        }
        std::vector<float> preferred,lower,upper;
        for(const auto &r:rows) {
            const float middle=(r.l+r.r)*.5f,half=r.width*(vehicleFraction*.5f+marginFraction),pad=std::max(6.f,r.width*.025f);
            float center=middle;
            if(onRoad.empty())center+=heldOffset*(r.r-r.l)*returnScale;
            for(auto b:onRoad) {
                const float near=b.br().y+pad,far=b.y-pad;
                const float weight=r.y>near?smooth((near+entryLeadRows-r.y)/entryLeadRows):
                    r.y<far?smooth((r.y-(far-farBlendRows))/farBlendRows):1;
                // Aim at the middle of the available gap, rather than touching
                // the minimum cone-clearance boundary. The longer near ramp
                // gives v6 meaningful right/left evidence before the cone is
                // close; ordinary lane-following gains remain unchanged.
                const float minimum=direction>0?b.br().x+pad+half:b.x-pad-half;
                const float outer=direction>0?r.r-half:r.l+half;
                const float wanted=(minimum+outer)*.5f;
                if(direction>0)center=std::max(center,middle+weight*std::max(0.f,wanted-middle));
                else center=std::min(center,middle+weight*std::min(0.f,wanted-middle));
            }
            // Prefer full-body containment; painted boundary crossing is a
            // soft cost when that preference eliminates an otherwise usable
            // cone-free centre path. Cone envelope remains a constraint.
            float lo=std::max({r.l+half,float(r.l+15),19.f}),hi=std::min({r.r-half,float(r.r-15),300.f});
            for(auto b:onRoad)if(r.y>=b.y-pad&&r.y<=b.br().y+pad) {
                if(direction>0)lo=std::max(lo,b.br().x+pad+half);
                else hi=std::min(hi,b.x-pad-half);
            }
            if(softRoad) {
                lo=std::max(float(r.l+15),19.f);hi=std::min(float(r.r-15),300.f);
                for(auto b:onRoad)if(r.y>=b.y-pad&&r.y<=b.br().y+pad) {
                    if(direction>0)lo=std::max(lo,b.br().x+pad+half);
                    else hi=std::min(hi,b.x-pad-half);
                }
                p.visibilityLimited=true;
                p.virtualLineCrossing=true;
            }
            if(lo>hi+geometryEpsilon)return finish("VIRTUAL_NO_CLEAR_PATH","CONE_ENVELOPE_BLOCKS_VISIBLE_CENTERS");
            hi=std::max(lo,hi);
            // Propagate reachable intervals rather than rejecting one jagged
            // preferred point. Only change the path inside current free space.
            if(!lower.empty()) {
                const float step=cfg.maxLateralSlope*(rows[lower.size()-1].y-r.y)+.5f;
                lo=std::max(lo,lower.back()-step);hi=std::min(hi,upper.back()+step);
            }
            if(lo>hi)return finish("VIRTUAL_NO_CLEAR_PATH","NO_CONNECTED_SMOOTH_CORRIDOR");
            lower.push_back(lo);upper.push_back(hi);preferred.push_back(center);
        }
        std::vector<float> solved(rows.size());solved.back()=std::clamp(preferred.back(),lower.back(),upper.back());
        for(int i=int(rows.size())-2;i>=0;--i) {
            const float step=cfg.maxLateralSlope*(rows[i].y-rows[i+1].y)+.5f;
            const float lo=std::max(lower[i],solved[i+1]-step),hi=std::min(upper[i],solved[i+1]+step);
            solved[i]=std::clamp(preferred[i],lo,std::max(lo,hi));
        }
        for(size_t i=0;i<rows.size();++i) {
            const auto &r=rows[i];const float center=solved[i],middle=(r.l+r.r)*.5f;
            const float left=direction>0?2*center-r.r:float(r.l),right=direction<0?2*center-r.l:float(r.r);
            if(right-left<30){p.centers.clear();p.left.clear();p.right.clear();return finish("VIRTUAL_NO_CLEAR_PATH","VIRTUAL_CORRIDOR_TOO_NARROW");}
            // The v6 controller receives a symmetric virtual band, fully inside
            // the repaired corridor. No 0/319 clipped edge is passed off as a
            // usable observed lane edge or allowed to bias its inferred centre.
            const float band=std::min({(right-left)*.5f,center-4,315-center});
            if(band<15){p.centers.clear();p.left.clear();p.right.clear();return finish("VIRTUAL_NO_CLEAR_PATH","CONTROL_BAND_CLIPPED");}
            p.centers.push_back({r.y,center});p.left.push_back({r.y,center-band});p.right.push_back({r.y,center+band});
            if(std::abs((center-middle)/(r.r-r.l))>std::abs(peak))peak=(center-middle)/(r.r-r.l);
        }
        if(!onRoad.empty())heldOffset=peak;
        p.candidate=true;p.stop=false;
        return finish(onRoad.empty()?(returning?"VIRTUAL_RETURN":"VIRTUAL_HOLD_SIDE"):direction>0?"VIRTUAL_RIGHT":"VIRTUAL_LEFT",
            p.virtualLineCrossing?"VISIBLE_ROAD_LINE_CROSSING_ALLOWED":widthHeld?"CURRENT_EDGES_RECENT_WIDTH":onRoad.empty()?"FRESH_LANE_VISUAL_EXIT_HEURISTIC":recovered.empty()?"BEZIER_VIRTUAL_BOUNDARY_V6_TRACKING":"MATCHED_IMAGE_CONE_RECOVERY");
    }
    RaceBypassDecision decision(const RaceObstaclePreviewPlan &p,bool automatic)const {
        RaceBypassDecision d;if(!automatic)return d;
        d.involved=p.hasTarget||active||interrupted;
        d.stop=d.involved&&(!p.fresh||p.stop||!p.candidate);
        d.follow=d.involved&&!d.stop;
        d.authorizeCone=d.follow||p.virtualVisualReturn||(p.fresh&&!p.hasTarget&&p.reason=="OFFROAD_CONES");
        // State text is owned by the caller's plan through the control iteration.
        d.state=p.state.c_str();d.reason=p.reason.c_str();return d;
    }
};

// Bounded integral correction for persistent IMAGE tracking error. This is
// inferred understeer, not measured wheel angle or measured vehicle motion.
class RaceVirtualSteeringFeedback {
    uint64_t sequence=0;
    double captured=-1,stalled=0;
    float extra=0,lastError=0,feed=0;
public:
    void reset(){sequence=0;captured=-1;stalled=0;extra=lastError=feed=0;}
    float correction()const{return extra+feed;}
    uint16_t update(const RaceObstaclePreviewPlan &p,float error,uint16_t base,uint16_t previous,
                    float dt,float previousSpeed) {
        if(!p.fresh||p.stop||!p.candidate||p.state=="VIRTUAL_RETURN"||p.state=="VIRTUAL_REACQUIRE"||!std::isfinite(error)||
           !std::isfinite(dt)||dt<=0||dt>.25f||previousSpeed<=.01f) {
            reset();return raceBypassServo(base,previous,dt);
        }
        // Small entry-only lead requested for the offset starting poses. It
        // fades with alignment and disappears as soon as the cone is behind.
        feed=0;
        if(!p.obstacles.empty()&&p.direction*error>8&&(int(base)-int(PWMSERVOMID))*p.direction>0)
            feed=p.direction*std::min(40.f,2*(p.direction*error-8))*std::clamp((p.primary.br().y-60)/100.f,.25f,1.f);
        if(p.sequence!=sequence) {
            const double elapsed=captured<0?0:p.captured-captured;
            if(elapsed<=0||elapsed>.25||p.sequence<sequence||error*lastError<=0||
               std::abs(error)<8||(int(base)-int(PWMSERVOMID))*error<0) {extra=0;stalled=0;}
            else if(std::abs(error)<std::abs(lastError)-.5f) {
                stalled=0;extra=std::copysign(std::max(0.f,std::abs(extra)-float(100*elapsed)),error);
            } else {
                stalled+=elapsed;
                if(stalled>.25)extra=std::copysign(std::min(120.f,std::abs(extra)+float(60*elapsed)*std::clamp(std::abs(error)/25.f,.5f,1.5f)),error);
            }
            sequence=p.sequence;captured=p.captured;lastError=error;
        }
        extra=std::clamp(extra,-120.f+std::abs(feed),120.f-std::abs(feed));
        const float target=std::clamp(float(base)+extra+feed,float(PWMSERVOMIN),float(PWMSERVOMAX));
        extra=target-base-feed; // no accumulation beyond actuator range
        return raceBypassServo(uint16_t(std::lround(target)),previous,dt);
    }
};
