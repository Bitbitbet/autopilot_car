#pragma once

static void parkingFallbackBoundsChecks() {
    RaceParkingDimensions dims;
    RaceParkingObservation start{{0,-.4f,0},1,1,1,true};
    RaceParkingPoseFallback fallback;
    auto step=[&](double time,bool valid,bool estimated,float speed){
        auto raw=start;raw.sequence=uint64_t(std::lround(time*100));raw.captured=time;
        raw.valid=valid;raw.estimated=estimated;
        return fallback.update(raw,time,speed,0,dims);
    };
    require(!step(1,true,true,0).valid,"Relative-only starting position initializes absolute budget");
    require(step(1.01,true,false,0).valid,"Fresh absolute position cannot initialize budget");
    require(step(1.11,false,false,.3f).estimated,"Brief missed frame cannot bridge");
    require(step(1.21,true,true,-.3f).valid&&step(1.31,false,false,.3f).valid,
        "In-budget relative correction cannot continue");
    require(!step(1.41,true,true,-.3f).valid&&fallback.budgetExhausted()&&fallback.estimateTravel()>.11f,
        "Opposite directions or relative fits reset blind travel");
    require(!step(1.51,true,true,0).valid,"Stopping replenishes exhausted travel");
    require(step(1.61,true,false,0).valid&&!fallback.budgetExhausted()&&fallback.estimateTravel()==0,
        "Original-frame visual return cannot replenish budget");
    require(step(2.22,true,true,0).valid&&!step(2.28,true,true,0).valid&&fallback.budgetExhausted(),
        "Stationary relative observations evade time bound");
    std::cout<<"PASS parking estimate bounded by time and absolute applied travel; relative frames and stops do not reset it\n";

    RaceParkingSpace space(dims);space.keepRearWithinStart(start.pose);
    auto route=racePlanTrackableParking(space,start.pose,.3f);
    require(route.valid,"Missing test parking route");
    {
        RaceParkingControl control(space,.3f);
        auto estimated=start;estimated.estimated=true;
        require(!control.arm(route,estimated,1,true)&&control.arm(route,start,1,true),"Estimate may arm a route");
        fallback.reset();fallback.update(start,1,0,0,dims);
        auto o=start;o.captured=1.1;++o.sequence;o.valid=false;
        auto bridged=fallback.update(o,o.captured,.1f,0,dims);
        require(bridged.valid&&bridged.estimated,"Missing short bridge for discontinuity test");
        control.update(bridged,o.captured,.1f,true,.3f);
        o.valid=true;o.captured=1.2;++o.sequence;o.pose.x+=.3f;
        const auto recovered=fallback.update(o,o.captured,.1f,0,dims);
        require(recovered.valid&&!recovered.estimated&&recovered.pose.x==o.pose.x,"Absolute correction was silently clipped");
        require(control.update(recovered,o.captured,.1f,true,.3f).speed==0&&control.failed(),
            "Large absolute correction bypasses discontinuity stop");
    }
    {
        RaceParkingControl control(space,.3f);auto o=start;
        require(control.arm(route,o,1,true),"Cannot arm stage confirmation test");
        bool entryHeld=false,dwell=false,reverse=false;int relativeDwell=0,relativeRejoin=0;double dwellResumed=0;
        float speed=0,curvature=0;
        for(int i=0;i<3000&&!control.done();++i) {
            o.captured+=.04;++o.sequence;
            const bool atEntryGoal=!dwell&&!entryHeld&&space.parked(o.pose);
            const bool inDwell=dwell&&!reverse&&relativeDwell<30;
            const bool atRejoin=reverse&&relativeRejoin<8&&raceParkingReadyToRejoin(space,o.pose);
            o.estimated=atEntryGoal||inDwell||atRejoin;
            const auto c=control.update(o,o.captured,.04f,true,.3f,speed,curvature);
            require(!control.failed(),"Stage confirmation model failed");
            if(o.estimated){
                require(!control.done()&&std::string(c.state)!="PARK_DWELL"&&std::string(c.state)!="PARK_COMPLETE",
                    "Relative tracking certified a parking stage");
                if(std::string(c.reason)=="WAIT_ABSOLUTE_VISUAL_CONFIRMATION") {
                    require(c.speed==0,"Relative arrival continues past goal");
                    if(atEntryGoal)entryHeld=true;
                    if(inDwell)++relativeDwell;
                    if(atRejoin)++relativeRejoin;
                }
            }
            if(std::string(c.state)=="PARK_DWELL") {
                dwell=true;
                if(relativeDwell==30&&!dwellResumed)dwellResumed=o.captured;
            }
            if(!reverse&&std::string(c.state)=="PARK_EXIT") {
                require(dwellResumed>0&&o.captured-dwellResumed>=.99,"Relative frames count toward stationary dwell");
                reverse=true;
            }
            speed=c.speed;curvature=c.curvature;o.pose=raceParkingAdvance(o.pose,speed*.04f,curvature,dims);
        }
        require(entryHeld&&relativeDwell==30&&relativeRejoin==8&&control.usedPoseEstimate()&&control.done()&&
            !control.terminalRecoveryUsed(),"Historical estimation blocks subsequent visually confirmed completion");
    }
    std::cout<<"PASS estimate cannot arm or certify containment/dwell/rejoin; fresh absolute confirmation resumes; true jumps still fault\n";
}
