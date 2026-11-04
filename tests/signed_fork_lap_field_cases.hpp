#pragma once

static void signedForkLapFieldChecks(const fs::path &dir) {
    // The field run kept its first selected fork until PARK superseded it.
    // Its images stop at selection; use its measured feedback/timing with
    // explicitly constructed road alternatives for the later handoff checks.
    std::ifstream input(dir/"parking-control.jsonl");
    require(bool(input),"Missing lap fork evidence");
    const auto arms=signedForkScene(-1,false);
    cv::Mat merged=cv::Mat::zeros(480,640,CV_8UC1);
    cv::rectangle(merged,{96,0,448,480},cv::Scalar(255),cv::FILLED);
    cv::rectangle(merged,{240,416,160,64},cv::Scalar(0),cv::FILLED);
    Track track;track.handle(merged);RaceConfig cfg;
    const auto view=raceBranchPath(merged,track,0,cfg);
    require(view.choiceRows>=cfg.minBranchRows&&view.singleRoadAhead,
        "Synthetic bonnet must separate the bottom strips of a single visible lane");

    SignedForkFixture lane(-1);lane.acquire(signedForkScene(-1,true));
    auto divided=lane,stationary=lane,lost=lane;
    for(int i=0;i<12;++i)lost.step(cv::Mat::zeros(480,640,CV_8UC1),.07f,.30f,false,false);
    require(lost.plan.stop&&!lost.plan.completedDirection,"Missing selected road bypasses the hold");

    int selected=0,recordedFrames=0,simulatedRelease=0;
    float elapsed=0,oldPeak=0;
    std::string line;nlohmann::json previous;
    while(std::getline(input,line)) {
        const auto r=nlohmann::json::parse(line);
        const int frame=r.at("frame");const std::string traffic=r.at("traffic_state");
        if(traffic=="PARKING_ENTRY_APPROACH")break;
        if(traffic=="FORK_LEFT") {
            if(!selected)selected=frame;
            ++recordedFrames;
            const float dt=r.at("control_dt_s"),applied=previous.at("command_speed");
            elapsed+=dt;oldPeak=std::max(oldPeak,r.at("fork_tracking_seconds").get<float>());
            require(r.at("fork_completed_direction")==0&&r.at("command_speed")<=.301f&&r.at("traffic_speed_cap")<=.301f,
                "This recording does not reproduce the stuck fork cap");
            const bool seen=r.at("fork_sign_clear_seconds").get<float>()==0;
            const bool valid=previous.at("lane_ready").get<bool>()&&previous.at("lane_alignment_valid").get<bool>();
            const float offset=previous.at("lane_alignment_offset_px"),heading=previous.at("lane_alignment_heading_px");
            if(!simulatedRelease) {
                const auto p=lane.step(merged,dt,applied,seen,valid,offset,heading);
                if(p.completedDirection) {
                    require(p.completedDirection==-1&&!p.stop&&!p.selectedBranch&&p.speedCap>.5f,
                        "Merged road handoff did not release ordinary fork control and cap");
                    simulatedRelease=frame;
                }
            }
            // The genuine two-arm and stationary cases remain capped despite
            // exactly the same sustained, locally aligned feedback.
            if(recordedFrames<=75) {
                const auto fork=divided.step(arms,dt,applied,seen,valid,offset,heading);
                const auto stopped=stationary.step(merged,dt,0,seen,valid,offset,heading);
                require(!fork.completedDirection&&fork.speedCap<=.251f&&!stopped.completedDirection,
                    "Local alignment alone released a live fork or a stationary vehicle");
            }
        }
        previous=r;
    }
    require(selected&&recordedFrames>400&&elapsed>25&&oldPeak<.60f&&simulatedRelease>selected,
        "Field timing, persistent cap, or merged-road policy was not reproduced");
    std::cout<<"PASS signed fork lap "<<dir.filename().string()<<" selected="<<selected
        <<" capped_frames="<<recordedFrames<<" capped_s="<<elapsed<<" old_peak_tracking_s="<<oldPeak
        <<" conditional_release="<<simulatedRelease<<" (recorded timing/feedback with synthetic merged road; post-turn images unavailable)\n";
}
