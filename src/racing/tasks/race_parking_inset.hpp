#pragma once
#include "race_parking_geometry.hpp"
#include <optional>

// Use the physical bay opening and all four sides. The aisle is traversable
// during entry but does not count as part of a completed parking footprint.
inline bool raceParkingInsetContained(const RaceParkingSpace &space,RaceParkingPose pose) {
    const float margin=std::max(.005f,space.dimensions().clearance);
    for(auto corner:space.body(pose,0))
        if(!space.inBay(corner,margin))return false;
    return true;
}

// At an oblique divider, padding both vehicle axes can reject a real corner
// that already has the requested perpendicular clearance. The extra case is
// restricted to the strip between the two divider lines (even in the aisle).
inline bool raceParkingInsetClear(const RaceParkingSpace &space,RaceParkingPose pose) {
    const auto &d=space.dimensions();
    const float margin=std::max(.005f,d.clearance);
    if(d.clearance>=margin&&space.free(pose))return true;
    if(!std::isfinite(pose.x)||!std::isfinite(pose.z)||!std::isfinite(pose.yaw)||pose.z<=-.9f||pose.z>=1.6f)
        return false;
    const float mouth=-d.aisle/2,stagger=d.stagger();
    for(auto q:space.body(pose,0)) {
        const float nearDistance=((q.y*d.depth)-(mouth-q.x)*stagger)/d.side;
        if(nearDistance<margin||nearDistance>d.depth-margin||q.x<mouth-d.depth+margin||
           q.x>d.aisle/2-margin||q.y<space.rearLimit()+margin)return false;
    }
    return true;
}

// Called once from a measured standstill. Test a bounded forward arc and
// its braking reserve before asking the driver to preset the steering.
inline std::optional<float> raceParkingInsetWheel(const RaceParkingSpace &space,RaceParkingPose start,
                                                   float currentWheel,float remainingM,bool firstFeasible=false) {
    const auto &d=space.dimensions();
    constexpr float step=.001f,coast=.020f;
    const float budget=std::min(.16f,remainingM);
    if(!std::isfinite(remainingM)||budget<=coast||!std::isfinite(currentWheel)||
       !raceParkingInsetClear(space,start)||raceParkingInsetContained(space,start))return std::nullopt;

    // Score the width of the interval where the original 15mm parking goal
    // is achievable. Every cycle after first containment must stay inside
    // with 5mm padding for the full 2cm reserve (0.5s at correction speed).
    const float mouth=-d.aisle/2,margin=std::max(.005f,d.clearance);
    const float normalX=d.stagger()/d.side,normalZ=d.depth/d.side;
    auto goalWindow=[&](float wheel,float drift) {
        auto pose=start;
        const float curvature=std::tan(wheel)/d.wheelbase;
        int firstInside=-1,goalSteps=0,longest=0;
        for(int i=1;i*step<=budget+1e-6f;++i) {
            pose=raceParkingAdvance(pose,step,curvature,d);
            pose.x-=drift*step*curvature*std::cos(pose.yaw);
            pose.z-=drift*step*curvature*std::sin(pose.yaw);
            // Project the oriented physical rectangle onto the side-divider
            // normal and the aisle axes. No polygon allocations in the search.
            const float s=std::sin(pose.yaw),c=std::cos(pose.yaw);
            const float cx=pose.x+.5f*d.length*s,cz=pose.z-.5f*d.length*c;
            const float halfX=.5f*(d.width*std::abs(c)+d.length*std::abs(s));
            const float halfZ=.5f*(d.width*std::abs(s)+d.length*std::abs(c));
            const float normalRight=std::abs(normalX*c+normalZ*s),normalForward=std::abs(normalZ*c-normalX*s);
            const float halfN=.5f*(d.width*normalRight+d.length*normalForward);
            const float h=(cx-mouth)*normalX+cz*normalZ;
            const float divider=std::min(h,d.depth-h)-halfN;
            const float back=cx-halfX-(mouth-d.depth),front=mouth-(cx+halfX);
            if(back<margin||cx+halfX>d.aisle/2-margin||cz-halfZ<space.rearLimit()+margin)return 0;
            // A corner in the aisle can lie outside the extended divider lines.
            // Use the clipped original envelope only in this ambiguous case.
            if(divider<margin) {
                const float side=h<=d.depth*.5f?-1.f:1.f;
                const float cornerX=cx+side*.5f*(std::copysign(d.width,normalX*c+normalZ*s)*c-
                                                   std::copysign(d.length,normalZ*c-normalX*s)*s);
                if(cornerX<=mouth-margin||!raceParkingInsetClear(space,pose))return 0;
            }
            if(std::min(front,divider)<margin) {
                if(firstInside>=0)return 0;
                continue;
            }
            if(firstInside<0)firstInside=i;
            const float goalMargin=std::max(.015f,d.clearance);
            const bool centered=std::min(front,back)>=goalMargin*(std::abs(c)+std::abs(s))&&
                                divider>=goalMargin*(normalRight+normalForward);
            goalSteps=centered?goalSteps+1:0;
            longest=std::max(longest,goalSteps);
            if((i-firstInside)*step>=coast-1e-6f)return 1+longest;
        }
        return 0;
    };
    int best=0;std::optional<float> chosen;
    for(int deg=-int(d.fullLockDegrees);deg<=int(d.fullLockDegrees);++deg) {
        const float wheel=deg*float(CV_PI)/180;
        int width=goalWindow(wheel,0);
        // The controller presets and holds the requested wheel before moving.
        // Evaluate residual angle error and the same lateral-drift model.
        const float lock=d.fullLockDegrees*float(CV_PI)/180;
        for(float error:{-.05236f,0.f,.05236f}) {
            if(width==0)break;
            const float actual=std::clamp(wheel+error,-lock,lock);
            if(error!=0)width=std::min(width,goalWindow(actual,0));
            if(width>0)width=std::min(width,goalWindow(actual,.09f));
            if(width>0)width=std::min(width,goalWindow(actual,-.09f));
        }
        if(firstFeasible&&width>0)return wheel;
        if(width>best||(width>0&&width==best&&std::abs(wheel-currentWheel)<std::abs(*chosen-currentWheel))) {
            chosen=wheel;best=width;
        }
    }
    return chosen;
}

inline bool raceParkingInsetArcClear(const RaceParkingSpace &space,RaceParkingPose start,float travel,float wheel) {
    const auto &d=space.dimensions();
    const float lock=d.fullLockDegrees*float(CV_PI)/180;
    for(float error:{-.05236f,0.f,.05236f}) {
        const float curvature=std::tan(std::clamp(wheel+error,-lock,lock))/d.wheelbase;
        for(float drift:{-.09f,.09f}) {
            auto pose=start;
            for(float used=0;used<std::abs(travel)-1e-6f;) {
                const float ds=std::copysign(std::min(.002f,std::abs(travel)-used),travel);
                pose=raceParkingAdvance(pose,ds,curvature,d);
                pose.x-=drift*ds*curvature*std::cos(pose.yaw);
                pose.z-=drift*ds*curvature*std::sin(pose.yaw);
                if(!raceParkingInsetClear(space,pose))return false;
                used+=std::abs(ds);
            }
        }
    }
    return true;
}
// The expected resting poses following a short reverse adjustment must all
// admit a complete forward correction. Select the actual wheel after settling.
inline bool raceParkingCanStageInset(const RaceParkingSpace &space,RaceParkingPose start) {
    const auto &d=space.dimensions();
    float outside=0;
    for(auto corner:space.body(start,0))outside=std::max(outside,corner.x+d.aisle/2);
    if(outside<=0||outside>.10f||!raceParkingInsetClear(space,start)||!raceParkingInsetArcClear(space,start,-.035f,0))
        return false;
    for(float ds:{-.010f,-.020f,-.030f})
        if(!raceParkingInsetWheel(space,raceParkingAdvance(start,ds,0,d),0,.16f,true))return false;
    return true;
}
