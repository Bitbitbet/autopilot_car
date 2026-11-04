#pragma once

static void parkingLineFieldChecks(const fs::path &projectRoot,const fs::path &latestRun) {
    for(const auto &run:{projectRoot/"vehicle_evidence/mission-1793775093534",latestRun}) {
        nlohmann::json saved;std::ifstream(run/"parking-route.json")>>saved;
        RaceParkingRoute route;route.valid=true;route.entryCurvaturePreview=saved.at("entry_curvature_preview_m");
        for(const auto &p:saved.at("entry"))route.entry.push_back({p.at(0),p.at(1),p.at(2)});
        route.exit.assign(route.entry.rbegin(),route.entry.rend());
        RaceParkingSpace space;space.keepRearWithinStart(route.entry.front());
        const auto &d=space.dimensions();
        RaceParkingControl replay(space,.30f);
        float speed=0,curvature=0;unsigned brake=0,recordedStop=0;int paced=0;bool checkedBlind=false;
        std::ifstream log(run/"parking-control.jsonl");std::string line;bool armed=false;
        while(std::getline(log,line)) {
            const auto r=nlohmann::json::parse(line);
            if(!r.at("armed").get<bool>())continue;
            if(r.at("state")=="PARK_DWELL"){recordedStop=r.at("frame");break;}
            const auto p=r.at("visual_pose");
            RaceParkingObservation o{{p[0],p[1],p[2]},r.at("captured"),r.at("frame"),r.at("motion_epoch"),r.at("motion_valid"),r.at("pose_estimated")};
            if(!armed){armed=replay.arm(route,o,o.captured,true);require(armed,"Recorded bay failed to arm");}
            if(!checkedBlind&&o.pose.x<-.5f*d.aisle-.35f*d.depth) {
                auto interrupted=replay;auto blind=o;blind.estimated=true;
                const float dt=r.at("control_dt_s");
                require(interrupted.update(blind,o.captured,dt,true,.30f,speed,curvature).speed==0,
                    "An estimated deep-bay pose authorizes forward motion without measured coast");
                ++o.sequence;o.captured+=.08;
                require(interrupted.update(o,o.captured,.08f,true,.30f,0,0).speed==0,
                    "A single recovered deep-bay pose authorizes forward motion");
                --o.sequence;o.captured=r.at("captured");checkedBlind=true;
            }
            const auto cmd=replay.update(o,o.captured,r.at("control_dt_s"),true,.30f,speed,curvature);
            if(cmd.speed>0&&cmd.speed+.01f<r.at("command_speed").get<float>())++paced;
            if(!brake&&std::string_view(cmd.state)=="PARK_BRAKE") {
                require(cmd.speed==0&&space.free(o.pose),"Entry first braked after reaching the paint");
                brake=o.sequence;
            }
            speed=r.at("command_speed");curvature=-r.at("physical_curvature").get<float>();
        }
        require(checkedBlind&&paced>0&&brake&&recordedStop>brake,"Missing recorded-pose slowdown or advance brake");
        require(raceParkingModelCompletes(space,route,.30f)&&raceParkingModelCompletes(space,route,.30f,.10f),
            "Previously feasible contained parking no longer completes");
        const auto planned=racePlanTrackableParking(space,route.entry.front(),.30f);
        require(planned.valid&&space.parkingGoal(planned.entry.back())&&raceParkingModelCompletes(space,planned,.30f,.08f,true),
            "Recorded start has no centered, response-checked entry route");
        float maxPathChange=planned.entry.size()==route.entry.size()?0.f:1.f;
        if(!maxPathChange)for(size_t i=0;i<route.entry.size();++i)
            maxPathChange=std::max(maxPathChange,std::hypot(route.entry[i].x-planned.entry[i].x,route.entry[i].z-planned.entry[i].z));
        std::cout<<"PASS parking line "<<run.filename().string()<<" paced_frames="<<paced<<" brake="<<brake
            <<" old_zero="<<recordedStop<<" same_reference="<<(maxPathChange<.0001f)
            <<" nodes="<<planned.expanded<<"; recorded poses; nominal full maneuver and delayed/coasting entry model\n";
    }
}
