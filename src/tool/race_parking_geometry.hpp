#pragma once
#include <opencv2/core.hpp>
#include <algorithm>
#include <cmath>
#include <limits>
#include <queue>
#include <stdexcept>
#include <unordered_map>
#include <vector>
#include <functional>

// Fixed empty bay 2. Dimensions measured on 2026-10-29; rear-axle offset and
// steering angle are explicit experimental values, not measured calibrations.
struct RaceParkingDimensions {
    float side=.41f,depth=.38f,aisle=.45f,length=.31f,width=.18f,wheelbase=.20f;
    // Two field runs imply ~33deg effective full-lock response, not the old
    // assumed 45deg. Experimental visual estimate, not wheel-angle calibration.
    float frontToRearAxle=.255f,clearance=.005f,fullLockDegrees=33.f,dwellSeconds=3.f;
    int bay=2;
    void validate() const {
        for(float v:{side,depth,aisle,length,width,wheelbase,frontToRearAxle,clearance,fullLockDegrees,dwellSeconds})
            if(!std::isfinite(v)||v<=0)throw std::invalid_argument("Non-finite parking dimensions");
        if(depth>side||side>.8f||aisle<width+2*clearance||aisle>.8f||length>.6f||width>.3f||
           wheelbase>=length||frontToRearAxle<wheelbase||frontToRearAxle>=length||
           clearance<.003f||clearance>.04f||fullLockDegrees<20||fullLockDegrees>60||dwellSeconds>15||bay!=2)
            throw std::invalid_argument("Invalid fixed parking bay dimensions");
    }
    float stagger() const {validate();return std::sqrt(std::max(0.f,side*side-depth*depth));}
};
struct RaceParkingPose {float x=0,z=0,yaw=0;}; // front bumper ground origin; positive yaw turns left
struct RaceParkingMouthCheck {
    bool valid=false;
    const char *reason="P_MOUTH_SCALE_MISMATCH";
    float scale=0;
    cv::Point2f near,far;
};
inline RaceParkingMouthCheck raceCheckParkingMouth(cv::Point2f near,cv::Point2f far,const RaceParkingDimensions &d={}) {
    RaceParkingMouthCheck out;out.scale=d.side/cv::norm(far-near);
    if(!std::isfinite(out.scale)||out.scale<.8f||out.scale>1.3f)return out;
    out.near=near*out.scale;out.far=far*out.scale;
    if(out.far.y<=out.near.y){out.reason="P_NEAR_FAR_ORDER_REVERSED";return out;}
    if(std::abs(out.far.x-out.near.x)>.10f){out.reason="P_ALIGN_CAR_WITH_AISLE";return out;}
    // Visibility and a collision-free forward route decide feasibility. The old
    // arbitrary 8cm staging requirement is not a measured vehicle constraint.
    if(out.near.y<0){out.reason="P_NEAR_BEHIND_VISIBLE_FRONT";return out;}
    if(out.far.y>1.4f){out.reason="P_FAR_POINT_TOO_DISTANT";return out;}
    out.valid=true;out.reason="P_MOUTH_OK";return out;
}
inline float raceParkingAngle(float a) {return std::remainder(a,float(2*CV_PI));}
inline RaceParkingPose raceParkingAdvance(RaceParkingPose p,float ds,float curvature,const RaceParkingDimensions &d) {
    // Rear axle bicycle integration, then transform back to front-bumper coordinates.
    float x=p.x+d.frontToRearAxle*std::sin(p.yaw),z=p.z-d.frontToRearAxle*std::cos(p.yaw);
    const float a=p.yaw+ds*curvature;
    if(std::abs(curvature)>1e-6f) {
        x+=(std::cos(a)-std::cos(p.yaw))/curvature;
        z+=(std::sin(a)-std::sin(p.yaw))/curvature;
    } else {x-=ds*std::sin(p.yaw);z+=ds*std::cos(p.yaw);}
    return {x-d.frontToRearAxle*std::sin(a),z+d.frontToRearAxle*std::cos(a),raceParkingAngle(a)};
}
class RaceParkingSpace {
    RaceParkingDimensions cfg;
    float rearBoundary=-std::numeric_limits<float>::infinity();
public:
    explicit RaceParkingSpace(RaceParkingDimensions d={}):cfg(d){cfg.validate();}
    const RaceParkingDimensions &dimensions()const{return cfg;}
    void keepRearWithinStart(RaceParkingPose start) {
        rearBoundary=std::numeric_limits<float>::infinity();
        for(auto p:body(start))rearBoundary=std::min(rearBoundary,p.y);
    }
    float rearLimit()const{return rearBoundary;}
    std::vector<cv::Point2f> bay()const {
        const float x=-cfg.aisle/2,s=cfg.stagger();
        return {{x,0},{x,cfg.side},{x-cfg.depth,cfg.side+s},{x-cfg.depth,s}};
    }
    bool inBay(cv::Point2f p,float margin=0) const {
        const float t=(-cfg.aisle/2-p.x)/cfg.depth;
        // Distance to the two sloping boundaries; parallel mouth/outer edges use x.
        const float h=(p.y-t*cfg.stagger())*cfg.depth/cfg.side;
        return p.x<=-cfg.aisle/2-margin+1e-6f&&p.x>=-cfg.aisle/2-cfg.depth+margin-1e-6f&&
            h>=margin-1e-6f&&h<=cfg.depth-margin+1e-6f;
    }
    std::vector<cv::Point2f> body(RaceParkingPose p,float margin=-1) const {
        std::vector<cv::Point2f> points;
        const float clearance=margin<0?cfg.clearance:margin;
        const float s=std::sin(p.yaw),c=std::cos(p.yaw);
        for(auto ab:{cv::Point2f(-.5f,0),cv::Point2f(.5f,0),cv::Point2f(.5f,1),cv::Point2f(-.5f,1)}) {
            const float x=ab.x*(cfg.width+2*clearance),z=clearance-ab.y*(cfg.length+2*clearance);
            points.emplace_back(p.x+c*x-s*z,p.z+s*x+c*z);
        }
        return points;
    }
private:
    bool freeWithRearTolerance(RaceParkingPose p,float rearTolerance,float mouthTolerance=0,bool taperedMouth=false)const {
        if(!std::isfinite(p.x)||!std::isfinite(p.z)||!std::isfinite(p.yaw)||p.z<=-.9f||p.z>=1.6f)return false;
        const auto polygon=body(p);const float edge=-cfg.aisle/2;
        auto inObservedBay=[&](cv::Point2f q) {
            // Paint at the mouth is traversable. Give observed geometry a
            // small local noise budget; do not expand the deep bay or road.
            if(taperedMouth) {
                const float depth=(edge-q.x)/cfg.depth;
                const float h=(q.y-depth*cfg.stagger())*cfg.depth/cfg.side;
                const float allowance=mouthTolerance*std::clamp(1-depth,0.f,1.f);
                return q.x>=edge-cfg.depth&&h>=-allowance&&h<=cfg.depth+allowance;
            }
            return inBay(q,q.x>=edge-.06f?-mouthTolerance:0.f);
        };
        for(auto q:polygon)if(q.x>cfg.aisle/2||q.y<rearBoundary-rearTolerance-1e-6f)return false;
        // Clip the complete convex body to the portion outside the aisle. Every
        // vertex of that polygon must lie in the convex bay, including intersections
        // at the mouth; checking four corners alone misses a crossed divider.
        for(size_t i=0;i<polygon.size();++i) {
            const auto a=polygon[i],b=polygon[(i+1)%polygon.size()];
            if(a.x<edge&&!inObservedBay(a))return false;
            if((a.x<edge)!=(b.x<edge)) {
                const float t=(edge-a.x)/(b.x-a.x);
                if(!inObservedBay({edge,a.y+t*(b.y-a.y)}))return false;
            }
        }
        return true;
    }
public:
    // Search and predicted stopping sweeps retain the exact initial rear plane.
    bool free(RaceParkingPose p)const {return freeWithRearTolerance(p,0);}
    // Engineering noise budgets, not calibrated accuracy. Rear plane: 2mm;
    // traversable painted mouth: 10mm within the first 6cm. Planned footprints
    // and deep-bay/road boundaries retain the original geometry.
    static constexpr float observedRearTolerance=.002f;
    static constexpr float observedMouthTolerance=.01f;
    bool freeObserved(RaceParkingPose p)const {return freeWithRearTolerance(p,observedRearTolerance,observedMouthTolerance);}
    // A convex tapered transition over traversable mouth paint. It adds no
    // space at the outer bay edge, does not move the road/rear limits, and is
    // never used as the parked-success boundary or a planning shortcut.
    static constexpr float entryMouthAllowance=.04f;
    bool freeEntryMouth(RaceParkingPose p)const {
        return freeWithRearTolerance(p,observedRearTolerance,entryMouthAllowance,true);
    }
    bool parked(RaceParkingPose p)const {
        for(auto q:body(p))if(!inBay(q))return false;
        return true;
    }
    bool parkedObserved(RaceParkingPose p)const {
        // Keep planning clearance out of a stationary camera-jitter decision.
        // Physical footprint must remain in the bay, within 3mm observation noise.
        for(auto q:body(p,0))if(!inBay(q,-.003f))return false;
        return true;
    }
};
// Spatial samples interpolate measured rear-axle poses; no command/time odometry.
// Unlike popping one historical frame per loop, resolution/FPS do not set distance.
inline std::vector<RaceParkingPose> raceParkingReverseSamples(const std::vector<RaceParkingPose> &entry,const RaceParkingDimensions &d) {
    if(entry.size()<2)return entry;
    auto axle=[&](auto p){return cv::Point2f(p.x+d.frontToRearAxle*std::sin(p.yaw),p.z-d.frontToRearAxle*std::cos(p.yaw));};
    std::vector<RaceParkingPose> result{entry.front()};float remainder=0;
    for(size_t i=1;i<entry.size();++i) {
        const auto a=axle(entry[i-1]),b=axle(entry[i]);const float length=cv::norm(b-a);
        if(length<1e-6f)continue;
        float used=.015f-remainder;
        for(;used<=length;used+=.015f) {
            const float t=used/length,yaw=entry[i-1].yaw+t*raceParkingAngle(entry[i].yaw-entry[i-1].yaw);
            const auto q=a+(b-a)*t;
            result.push_back({q.x-d.frontToRearAxle*std::sin(yaw),q.y+d.frontToRearAxle*std::cos(yaw),yaw});
        }
        remainder=length-(used-.015f);
    }
    if(cv::norm(axle(entry.back())-axle(result.back()))<.003f)result.back()=entry.back();else result.push_back(entry.back());
    std::reverse(result.begin(),result.end());return result;
}
struct RaceParkingSearchTrial {float curvatureScale=1,heuristicWeight=1;int expanded=0,candidates=0,rejected=0;const char *reason="NOT_RUN";float curvaturePreview=.10f;};
struct RaceParkingRoute {bool valid=false;const char *reason="NO_PARKING_PATH";std::vector<RaceParkingPose> entry,exit;int expanded=0,geometricCandidates=0,modelRejected=0;std::vector<RaceParkingSearchTrial> searches;float entryCurvaturePreview=.10f;};
inline RaceParkingRoute racePlanParking(const RaceParkingSpace &space,RaceParkingPose start,float curvatureScale=1.f,
                                      const std::function<bool(const RaceParkingRoute&)> &acceptRoute={},float heuristicWeight=1.f,bool preferEarlyTurn=true) {
    RaceParkingRoute out;
    if(!std::isfinite(curvatureScale)||curvatureScale<.7f||curvatureScale>1||
       !std::isfinite(heuristicWeight)||heuristicWeight<1||heuristicWeight>2){out.reason="INVALID_PARKING_SEARCH_CURVATURE";return out;}
    if(!space.free(start)){out.reason="START_OUTSIDE_AISLE_OR_BAY";return out;}
    const auto &d=space.dimensions();const float step=.025f,k=curvatureScale*std::tan(d.fullLockDegrees*float(CV_PI)/180)/d.wheelbase;
    struct Node {RaceParkingPose p;float cost;int parent;float steering=0;};
    struct Item {float score;int index;bool operator<(const Item &b)const{return score>b.score;}};
    auto key=[](RaceParkingPose p,float steering) {
        const int x=int(std::lround((p.x+1.f)/.02f)),z=int(std::lround((p.z+1.f)/.02f));
        const int a=int(std::lround((raceParkingAngle(p.yaw)+float(CV_PI))/.065f));
        const int wheel=int(std::lround(steering/.13f))+9;
        return (((int64_t(x)<<32)|(int64_t(z)<<16)|a)<<5)|wheel;
    };
    const cv::Point2f goal{-d.aisle/2-d.depth/2,d.side/2+d.stagger()/2};
    auto heuristic=[&](RaceParkingPose p) {
        const float centerX=p.x+d.length*.5f*std::sin(p.yaw),centerZ=p.z-d.length*.5f*std::cos(p.yaw);
        return std::hypot(centerX-goal.x,centerZ-goal.y)+.04f*std::abs(raceParkingAngle(p.yaw-float(CV_PI)/2));
    };
    std::vector<Node> nodes{{start,0,-1}};std::priority_queue<Item> queue;queue.push({heuristicWeight*heuristic(start),0});
    std::unordered_map<int64_t,float> best;best[key(start,0)]=0;
    while(!queue.empty()&&out.expanded<35000) {
        const int index=queue.top().index;queue.pop();const auto node=nodes[index];
        if(node.cost>best[key(node.p,node.steering)]+1e-5f)continue;
        ++out.expanded;
        // A complete body inside the bay is a goal. Do not require another
        // centimetre beyond the already inflated footprint in this tight bay.
        if(space.parked(node.p)) {
            ++out.geometricCandidates;RaceParkingRoute candidate=out;
            for(int i=index;i>=0;i=nodes[i].parent)candidate.entry.push_back(nodes[i].p);
            std::reverse(candidate.entry.begin(),candidate.entry.end());candidate.exit=candidate.entry;std::reverse(candidate.exit.begin(),candidate.exit.end());
            candidate.valid=true;candidate.reason="PREVIEW_GEOMETRY_ONLY";
            if(!acceptRoute||acceptRoute(candidate))return candidate;
            ++out.modelRejected;
            // Rejection of the first goal is not proof that no other trackable
            // approach exists. Continue the same bounded search for alternatives.
            if(out.modelRejected>=24){out.reason="MODEL_CANDIDATE_BUDGET_EXHAUSTED";return out;}
        }
        for(float factor:{-1.f,-.5f,0.f,.5f,1.f}) {
            // In the preferred entry, retain the established left turn instead
            // of an early left wiggle followed by straightening toward the far
            // divider. The fallback search still permits steering reversals.
            if(preferEarlyTurn&&node.p.yaw>.08f&&factor<0)continue;
            auto next=node.p;bool ok=true;float wheel=node.steering;
            const float target=std::atan(d.wheelbase*k*factor);
            const float wheelRate=d.fullLockDegrees*float(CV_PI)/180*600/400;
            // Search the same finite steering slew as control at the parking
            // task's 0.10m/s, rather than instant wheel jumps every 2.5cm.
            for(int sub=0;sub<5;++sub) {
                wheel+=std::clamp(target-wheel,-wheelRate*(step/5)/.10f,wheelRate*(step/5)/.10f);
                next=raceParkingAdvance(next,step/5,std::tan(wheel)/d.wheelbase,d);
                if(!space.free(next)){ok=false;break;}
            }
            if(!ok)continue;
            // Prefer fewer steering reversals, rather than a lattice zigzag
            // that assumes each 2.5cm segment can instantly change wheel angle.
            // Prefer building the left turn while approaching the mouth. A
            // shortest-path-only search can hug the aisle edge, go straight,
            // then ask for a late sharp turn beside the far divider. These are
            // soft costs: a tight start can still use any feasible trajectory.
            const float desiredHeading=std::clamp((next.z+.20f)*2.f,0.f,1.35f);
            const float lateTurn=std::max(0.f,desiredHeading-next.yaw);
            float farPressure=0;
            for(auto q:space.body(next))if(q.x<-d.aisle/2) {
                const float depth=(-d.aisle/2-q.x)/d.depth;
                const float h=(q.y-depth*d.stagger())*d.depth/d.side;
                farPressure=std::max(farPressure,std::max(0.f,(h-(d.depth-.08f))/.08f));
            }
            const float cost=node.cost+step*(1+.04f*std::abs(factor)+(preferEarlyTurn?1.5f*lateTurn+1.5f*farPressure:0.f))+
                .008f*std::abs(wheel-node.steering);auto id=key(next,wheel);
            if(auto old=best.find(id);old!=best.end()&&old->second<=cost)continue;
            best[id]=cost;nodes.push_back({next,cost,index,wheel});queue.push({cost+heuristicWeight*heuristic(next),int(nodes.size()-1)});
        }
    }
    out.reason=out.expanded>=35000?"SEARCH_BUDGET_EXHAUSTED":"NO_FOOTPRINT_CLEAR_PATH";return out;
}

// Dwell time is counted only while fresh observations continuously confirm a stopped,
// fully contained vehicle. Unknown reverse capability never becomes an exit command.
class RaceParkingDwell {
    double stoppedSince=-1;
public:
    bool ready(double now,bool fresh,bool stopped,bool fullyInside,bool reverseVerified,float seconds=3.f) {
        if(!std::isfinite(now)||!std::isfinite(seconds)||seconds<3||!fresh||!stopped||!fullyInside){stoppedSince=-1;return false;}
        if(stoppedSince<0||now<stoppedSince)stoppedSince=now;
        return now-stoppedSince>=seconds&&reverseVerified;
    }
};
