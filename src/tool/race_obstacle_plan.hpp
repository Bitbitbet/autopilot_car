#pragma once
#include "race_navigation_config.hpp"
#include "tools.hpp"
#include <cstdint>
#include <string>
#include <vector>

struct RacePreviewPoint { int x=0;float y=0; }; // row, col；保留半像素搜索结果。

struct RaceObstaclePreviewPlan {
    uint64_t sequence=0;
    double captured=0;
    bool fresh=false,hasTarget=false,candidate=false,stop=true;
    bool visibilityLimited=false;
    int clippedRows=0,visibleNearRow=-1,visibleFarRow=-1;
    std::string state="OBS_NO_TARGET",reason="PIXEL_PREVIEW_UNCALIBRATED";
    int direction=0,confirmedTargets=0;
    cv::Rect primary;
    std::vector<cv::Rect> obstacles;
    std::vector<RacePreviewPoint> centers,left,right;
    // 图像内确实可见的通路范围；边界未入镜时它不是完整道路边界。
    std::vector<PointX> visibleLeft,visibleRight;
    std::vector<int> leftBoundarySeen,rightBoundarySeen;
    // 标定后的绕行实验与像素预览严格分开。真正发出的指令见 bypass-control.csv。
    bool metric=false,motionValid=false,passed=false;
    int motionEpoch=0,motionInliers=0;
    int motionFeatures=0,motionMatches=0,motionMisses=0;
    float motionReferenceAge=0;
    bool motionCoordinatesHeld=false;
    std::string motionReason="UNAVAILABLE";
    float motionFitScale=0,motionFitTranslation=0,motionFitAngle=0,motionSpanX=0,motionSpanZ=0;
    bool motionBackupReferenceUsed=false;
    int motionBackupAttempts=0;
    int trackedTargets=0,discardedTargets=0;
    bool historyCommitted=false;
    float laneOffset=0,laneHeading=0,motionResidual=0;
    float motionX=0,motionZ=0,motionAngle=0;
    float groundVehicleWidth=0,groundVehicleLength=0,groundRoadWidth=0,groundClearance=0;
    // 四点标定原始米制下的实测中位路宽、按45cm求得的米制尺度、修正后的实测路宽。
    // 它们只用于解释当前画面为何停车或放行，不替代配置的45cm。
    float laneWidthRaw=0,laneScale=1,laneWidthMeasured=0;
    // 未锁定时"需要的倍数"（=45cm/实测中位宽），用于把越界原因变成可读数字。
    float laneScaleWanted=0;
    int laneRows=0,laneRejectedRows=0;
    float laneWidthObserved=0,laneWidthUsed=0;
    // 可信行在地面上的纵向跨度与近/远边界：跨度<0.2m 会报 GROUND_LANE_SPAN_SHORT，
    // 只有这三个数才能区分"真跨度不足"和"近处被锥桶/车壳遮罩吃掉"。
    float laneSpanZ=0,laneZNear=0,laneZFar=0;
    int predictedTargets=0;
    float routeBlockedZ=-1,coneX=0,coneZ=0,coneRadius=0,leftGap=0,rightGap=0;
    std::vector<std::array<float,2>> groundCenters;
};

inline nlohmann::json raceObstacleJson(const RaceObstaclePreviewPlan &p,const RaceNavigationConfig &cfg) {
    auto points=[](const auto &v) {
        auto out=nlohmann::json::array();for(const auto &a:v)out.push_back({a.x,a.y});return out;
    };
    auto obstacles=nlohmann::json::array();
    for(const auto &r:p.obstacles)obstacles.push_back({r.x,r.y,r.width,r.height});
    return {{"frame",p.sequence},{"captured",p.captured},{"state",p.state},{"reason",p.reason},
        {"fresh",p.fresh},{"has_target",p.hasTarget},{"candidate",p.candidate},{"would_stop",p.stop},
        {"direction",p.direction},{"confirmed_targets",p.confirmedTargets},{"preview_only",!p.metric},
        {"units",p.metric?"ground_meters_with_image_overlay":"image_pixels"},
        {"physical_clearance_verified",false},{"drives_vehicle",false},{"metric_route",p.metric},
        {"motion_valid",p.motionValid},{"motion_epoch",p.motionEpoch},{"motion_inliers",p.motionInliers},
        {"motion_features",p.motionFeatures},{"motion_matches",p.motionMatches},{"motion_misses",p.motionMisses},{"motion_reference_age_s",p.motionReferenceAge},
        {"motion_coordinates_held",p.motionCoordinatesHeld},{"motion_reason",p.motionReason},
        {"motion_fit_scale",p.motionFitScale},{"motion_fit_translation_m",p.motionFitTranslation},
        {"motion_fit_angle_rad",p.motionFitAngle},{"motion_span_x_m",p.motionSpanX},{"motion_span_z_m",p.motionSpanZ},
        {"motion_backup_reference_used",p.motionBackupReferenceUsed},{"motion_backup_attempts",p.motionBackupAttempts},
        {"tracked_targets",p.trackedTargets},{"discarded_targets",p.discardedTargets},{"history_committed",p.historyCommitted},
        {"motion_residual_m",p.motionResidual},{"rear_passed",p.passed},
        {"motion_x_m",p.motionX},{"motion_z_m",p.motionZ},{"motion_angle_rad",p.motionAngle},
        {"lane_offset_m",p.laneOffset},{"lane_heading",p.laneHeading},{"ground_center",p.groundCenters},
        {"visibility_limited",p.visibilityLimited},{"clipped_rows",p.clippedRows},
        {"width_reference",p.metric?"current_car_ground_calibration":"visible_image_corridor"},
        {"vehicle_width_m",p.groundVehicleWidth},{"vehicle_length_m",p.groundVehicleLength},
        {"road_width_m",p.groundRoadWidth},{"clearance_m",p.groundClearance},
        {"lane_width_raw_m",p.laneWidthRaw},{"lane_scale",p.laneScale},{"lane_width_measured_m",p.laneWidthMeasured},
        {"lane_scale_wanted",p.laneScaleWanted},
        {"lane_rows",p.laneRows},{"lane_rejected_rows",p.laneRejectedRows},
        {"lane_width_observed_m",p.laneWidthObserved},{"lane_width_used_m",p.laneWidthUsed},
        {"lane_span_z_m",p.laneSpanZ},{"lane_z_near_m",p.laneZNear},{"lane_z_far_m",p.laneZFar},
        {"predicted_targets",p.predictedTargets},{"route_blocked_z_m",p.routeBlockedZ},
        {"nearest_cone_x_m",p.coneX},{"nearest_cone_z_m",p.coneZ},{"nearest_cone_radius_m",p.coneRadius},
        {"left_gap_m",p.leftGap},{"right_gap_m",p.rightGap},
        {"visible_near_row",p.visibleNearRow},{"visible_far_row",p.visibleFarRow},
        {"visible_left",points(p.visibleLeft)},{"visible_right",points(p.visibleRight)},
        {"left_boundary_seen",p.leftBoundarySeen},{"right_boundary_seen",p.rightBoundarySeen},
        {"assumed_vehicle_width_fraction",cfg.previewVehicleWidthFraction},
        {"assumed_clearance_fraction",cfg.previewClearanceFraction},
        {"obstacles",obstacles},{"center",points(p.centers)},{"left",points(p.left)},{"right",points(p.right)}};
}

inline void drawRaceObstaclePreview(cv::Mat &image,const RaceObstaclePreviewPlan &p) {
    if(image.type()!=CV_8UC3||image.size()!=cv::Size(320,240))return;
    const auto color=p.candidate&&!p.visibilityLimited?cv::Scalar(255,255,0):cv::Scalar(0,165,255);
    for(const auto &r:p.obstacles)cv::rectangle(image,r,color,1);
    for(size_t i=1;i<p.centers.size();++i)
        cv::line(image,{int(std::lround(p.centers[i-1].y)),p.centers[i-1].x},{int(std::lround(p.centers[i].y)),p.centers[i].x},color,2);
    for(const auto *edge:{&p.left,&p.right})for(size_t i=1;i<edge->size();++i)
        cv::line(image,{int(std::lround((*edge)[i-1].y)),(*edge)[i-1].x},{int(std::lround((*edge)[i].y)),(*edge)[i].x},color,1);
    cv::putText(image,p.metric?"CALIBRATED BYPASS TRIAL":p.visibilityLimited?"VISIBLE ONLY / PIXELS":"PATH PREVIEW / PIXELS",{4,232},cv::FONT_HERSHEY_PLAIN,.8,color,1);
}
