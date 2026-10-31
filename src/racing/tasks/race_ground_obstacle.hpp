#pragma once
#include "race_ground_motion.hpp"
#include "race_obstacle.hpp"
#include <algorithm>
#include <optional>

inline bool raceGroundProjectPath(const RaceGroundMap &map,const RaceGroundCalibration &cfg,
                                  const std::vector<cv::Point2f> &path,RaceObstaclePreviewPlan &out) {
    if(path.size()<2)return false;
    std::vector<cv::Point2f> projected;for(const auto &p:path)projected.push_back(map.toImage(p));
    for(const auto &p:path)out.groundCenters.push_back({p.x,p.y});
    for(int row=438;row>=70;--row)for(size_t i=1;i<path.size();++i) {
        const auto a=projected[i-1],b=projected[i];
        if(!std::isfinite(a.y)||!std::isfinite(b.y)||a.y<=b.y||row>a.y||row<b.y)continue;
        const float alpha=(a.y-row)/(a.y-b.y);const cv::Point2f middle(a.x+(b.x-a.x)*alpha,float(row));
        const auto center=map.toGround(middle),left=map.toImage(center-cv::Point2f(cfg.vehicleWidth*.5f,0)),right=map.toImage(center+cv::Point2f(cfg.vehicleWidth*.5f,0));
        // 任意有效四点映射下都按实际图像行求交，不假设相机没有横滚。
        // 虚拟边线关于目标中心对称，避免透视不对称改变既有巡线控制器的目标中心。
        const float half=std::min(middle.x-left.x,right.x-middle.x);
        if(!std::isfinite(half)||half<30||middle.x-half<=8||middle.x+half>=630)break;
        out.centers.push_back({row,middle.x});out.left.push_back({row,middle.x-half});out.right.push_back({row,middle.x+half});break;
    }
    return out.centers.size()>=30&&out.centers.front().x>=150&&out.centers.front().x-out.centers.back().x>=50;
}

// 首版实跑只支持平坦直道上的静态锥桶。转弯、分叉、行人和不确定的地面运动均停车。
class RaceGroundObstacle {
    struct Cone {cv::Point2f world;float radius;int count;double seen;bool near=false;cv::Rect box;};
    struct Lane {float center=0,slope=0,width=0;};
    struct Row {int y=0,low=0,high=0;};
    // 四点标定给出初始映射与尺度；点选误差或车体姿态变化会影响米制换算。
    // 用已知的45cm路宽求一次有限修正（不代表已验证纵向距离或所有姿态变化）：
    // 观测比配置更宽说明标定尺度偏大（车辆被算得比实际小），更窄则可能是真的换窄路。
    // 超出范围时停车，由人确认，不无限放大或缩小米制。
    // 上界从1.10放宽到1.30：现场实测（2026-10-27 19:xx 一轮）需要1.1054，只差0.5%就被拦下，
    // 根因是那组四点的横向跨度比白纸实际边缘宽约9~11%（点偏外侧），属于系统性偏差而非换路面。
    // 下界保持0.80：需要缩小到0.8以下意味着画面里的路面比标定宽25%，那更可能是换道/岔路/别条路。
    static constexpr float kScaleMin=.80f,kScaleMax=1.30f;
    static constexpr float kRowWidthTolerance=.035f;
    RaceGroundCalibration cfg;RaceGroundMap map;RaceGroundMotion motion;
    std::vector<Cone> cones;
    RaceObstaclePreviewPlan latest;
    int epoch=-1,lockedDirection=0;
    uint64_t sequence=0;
    double captured=-1;
    bool scaleLocked=false;
    float lockedWidth=0;                                            // 锁定时的原始（四点标定）中位路宽
    float measuredWidth=0;                                          // 本帧修正后的中位路宽
    float pendingWidth=0,pendingScale=0;                            // 本帧实测中位宽 / 需要的倍数（未锁定时也填）
    std::vector<float> scaleWidths;

    // 逐行取左右路沿；只做图像工作，米制换算交给调用者。
    // 尺度估计与车道拟合必须共用同一组行，否则两处会看到不同的路。
    bool collectRows(const cv::Mat &bin,const Track &track,const RaceSnapshot &s,const RaceConfig &race,
                     std::vector<Row> &rows,std::string &issue,const RaceGroundMotionResult *current=nullptr) const {
        issue="GROUND_LANE_UNAVAILABLE";rows.clear();
        if(bin.type()!=CV_8UC1||track.pointsEdgeLeft.empty()||track.pointsEdgeRight.empty())return false;
        std::vector<cv::Rect> masks=raceSignRegions(s,race);
        for(const auto &d:s.detections)if(d.type==LABEL_CONE&&reviewRaceDetection(d,s.image,race,&s.crosswalk).accepted)
            masks.emplace_back(d.x,d.y,d.width,d.height);
        // 模型漏一帧时，桶仍会遮挡白路。与有限预测共用已确认地图，防止
        // 同一物体的橙色区域被当成新岔路；只用于路沿遮罩，不授予通行。
        const bool currentValid=current&&current->valid;
        if(currentValid||((latest.motionValid||latest.motionCoordinatesHeld)&&s.captured-latest.captured<=.18)) {
            const RaceGroundPose pose=currentValid?current->pose:RaceGroundPose{latest.motionX,latest.motionZ,latest.motionAngle};
            for(const auto &c:cones)if(c.count>=race.confirmFrames) {
                const auto local=pose.local(c.world);
                const bool mappedNear=c.near&&std::abs(local.x)>=cfg.vehicleWidth*.5f+c.radius+cfg.clearance;
                if(s.captured-c.seen>.35&&!mappedNear)continue;
                const bool visible=std::any_of(s.detections.begin(),s.detections.end(),[&](const PredictResult &d){
                    if(d.type!=LABEL_CONE||!reviewRaceDetection(d,s.image,race,&s.crosswalk).accepted)return false;
                    const cv::Rect box(d.x,d.y,d.width,d.height);const int overlap=(box&c.box).area();
                    return float(overlap)/std::max(1,box.area()+c.box.area()-overlap)>.15f;
                });
                if(visible)continue;
                if(local.y<=.10f)continue;
                const auto foot=map.toImage(local),left=map.toImage(local-cv::Point2f(c.radius,0)),right=map.toImage(local+cv::Point2f(c.radius,0));
                if(!std::isfinite(foot.y)||!std::isfinite(left.x)||!std::isfinite(right.x))continue;
                const int pad=currentValid?8:16;
                const auto oldFoot=map.toGround({c.box.x+c.box.width*.5f,float(c.box.y+c.box.height)});
                const float ratio=std::isfinite(oldFoot.y)?std::clamp(oldFoot.y/local.y,.5f,2.f):1.f;
                const int height=int(std::ceil(c.box.height*ratio));
                const auto predicted=cv::Rect(int(std::floor(left.x))-pad,int(std::floor(foot.y))-height-pad,
                    std::max(1,int(std::ceil(right.x-left.x)))+2*pad,height+2*pad)&cv::Rect(0,0,640,480);
                if(predicted.area()>0)masks.push_back(predicted);
            }
        }
        int low=0,high=639;
        for(int y=std::min(438,track.pointsEdgeLeft.front().x);y>=70;y-=4) {
            const auto blocks=raceRoadBlocks(bin,y,low,high,race,masks,3);
            if(blocks.empty())break;
            if(blocks.size()!=1){issue="GROUND_LANE_BRANCH";rows.clear();return false;}
            low=blocks[0].first;high=blocks[0].second;
            if(low<=6||high>=632)continue;
            // 物体从道路边缘遮进来时，剩余白块边缘不是道路边线；跳过该行，不能拿它校验路宽。
            const bool occluded=std::any_of(masks.begin(),masks.end(),[&](const cv::Rect &r){
                const auto expanded=cv::Rect(r.x-2,r.y-2,r.width+4,r.height+4);
                return expanded.contains({low,y})||expanded.contains({high,y});
            });
            if(occluded)continue;
            rows.push_back({y,low,high});
        }
        return !rows.empty();
    }

    // 用实测路宽求米制尺度。宽度始终按四点标定的原始坐标测量（除以当前尺度），
    // 否则会变成"修正后再修正"的反馈。MANUAL 阶段持续跟踪，AUTO 后冻结。
    void updateScale(const std::vector<Row> &rows,bool automatic,std::string &issue) {
        if(scaleLocked&&automatic)return;
        const float current=map.scale();
        std::vector<float> widths;
        for(const auto &r:rows) {
            const auto l=map.toGround({float(r.low),float(r.y)}),rr=map.toGround({float(r.high),float(r.y)});
            if(!std::isfinite(l.x)||!std::isfinite(l.y)||!std::isfinite(rr.x)||!std::isfinite(rr.y))continue;
            const float z=((l.y+rr.y)*.5f)/current,w=(rr.x-l.x)/current,depth=(l.y-rr.y)/current;
            if(z<cfg.nearDistance||z>cfg.farDistance||std::abs(depth)>.06f)continue;
            widths.push_back(w);
        }
        if(widths.size()<12){issue="GROUND_LANE_SCALE_ESTIMATING";return;}
        std::sort(widths.begin(),widths.end());
        const float median=widths[widths.size()/2],spread=widths.back()-widths.front();
        // 越界/不一致时也要把数值带出去，否则现场只看到"超出范围"却不知道差多少。
        pendingWidth=median;pendingScale=cfg.roadWidth/median;
        // 左右边不平行（接缝、弯道、遮挡残留）时这一帧不参与尺度，避免锁到错误值。
        if(spread>kRowWidthTolerance){issue="GROUND_LANE_WIDTH_INCONSISTENT";return;}
        const float wanted=cfg.roadWidth/median;
        if(!std::isfinite(wanted)||wanted<kScaleMin||wanted>kScaleMax){issue="GROUND_LANE_WIDTH_OUT_OF_RANGE";return;}
        scaleWidths.push_back(median);
        if(scaleWidths.size()>5)scaleWidths.erase(scaleWidths.begin());
        if(scaleWidths.size()<3){issue="GROUND_LANE_SCALE_ESTIMATING";return;}
        std::vector<float> recent=scaleWidths;std::sort(recent.begin(),recent.end());
        lockedWidth=recent[recent.size()/2];
        const float scale=cfg.roadWidth/lockedWidth;
        // 尺度系数死区0.02避免逐帧抖动改变米制。
        if(!scaleLocked||std::abs(scale-map.scale())>.02f){map.setMetricScale(scale);motion.setMetricScale(scale);}
        scaleLocked=true;
    }

    // 用锁定后的米制把同一组行换算成路宽与中心，再检验直道与宽度一致性。
    std::optional<Lane> lane(const std::vector<Row> &rows,std::string &issue,RaceObstaclePreviewPlan &out) {
        issue="GROUND_LANE_UNAVAILABLE";measuredWidth=0;
        std::vector<cv::Point2f> middle;std::vector<float> widths,observed;int skipped=0;
        for(const auto &r:rows) {
            const auto l=map.toGround({float(r.low),float(r.y)}),rr=map.toGround({float(r.high),float(r.y)});
            if(!std::isfinite(l.x)||!std::isfinite(l.y)||!std::isfinite(rr.x)||!std::isfinite(rr.y))continue;
            const float z=(l.y+rr.y)*.5f,w=rr.x-l.x;
            if(z<cfg.nearDistance||z>cfg.farDistance)continue;
            observed.push_back(w);
            // 单行路宽与配置值相差过大说明这条边不是稳定路沿（遮挡、接缝、收窄），跳过该行。
            if(std::abs(w-cfg.roadWidth)>kRowWidthTolerance||std::abs(l.y-rr.y)>.06f){++skipped;continue;}
            middle.emplace_back((l.x+rr.x)*.5f,z);widths.push_back(w);
        }
        out.laneRows=int(middle.size());out.laneRejectedRows=skipped;
        if(!observed.empty()) {std::sort(observed.begin(),observed.end());out.laneWidthObserved=observed[observed.size()/2];}
        if(middle.size()<12) {
            // 大部分行都被宽度检查拒绝，说明当前路面已不是锁定时测量的那条直道。
            issue=int(rows.size())>=12&&skipped*2>=int(rows.size())?"GROUND_LANE_WIDTH_MISMATCH":"GROUND_LANE_TOO_FEW_DUAL_ROWS";
            return {};
        }
        float sx=0,sz=0,szz=0,sxz=0,w=0,minz=10,maxz=-10;
        for(size_t i=0;i<middle.size();++i){const auto p=middle[i];sx+=p.x;sz+=p.y;szz+=p.y*p.y;sxz+=p.x*p.y;w+=widths[i];minz=std::min(minz,p.y);maxz=std::max(maxz,p.y);}
        const float n=middle.size(),den=n*szz-sz*sz;
        // 诊断：可信行的地面纵向跨度、近/远边界，用于区分"真跨度不足"与"被近桶遮罩吃掉"。
        out.laneSpanZ=maxz-minz;out.laneZNear=minz;out.laneZFar=maxz;
        if(maxz-minz<.2f||den<1e-5){issue="GROUND_LANE_SPAN_SHORT";return {};}
        const float b=(n*sxz-sx*sz)/den,a=(sx-b*sz)/n;
        float error=0;for(const auto &p:middle)error=std::max(error,std::abs(p.x-a-b*p.y));
        // 直道证据必须跨越足够地面距离；不能将弯道/窄段当作完整45cm道路外推。
        if(error>.012f||std::abs(b)>.45f){issue="GROUND_LANE_NOT_STRAIGHT";return {};}
        // 车身夹角下，等z截面的横向距离不是道路法向宽度。
        measuredWidth=w/n/std::hypot(1.f,b);out.laneWidthMeasured=measuredWidth;
        if(std::abs(measuredWidth-cfg.roadWidth)>kRowWidthTolerance){issue="GROUND_LANE_WIDTH_MISMATCH";return {};}
        // 不把观测的47cm当成额外空间；42~45cm观测则按最窄有效行规划。
        // 与逐行原有3.5cm检查共用容差，避免另一个2cm硬门槛造成边界抖动。
        const float used=std::min(cfg.roadWidth,*std::min_element(widths.begin(),widths.end()));
        out.laneWidthUsed=used;
        return Lane{a,b,used};
    }

public:
    // 与感知解耦的米制规划核心，便于用现场几何和闭环运动单独核对。
    static bool solveRoute(const RaceGroundCalibration &cfg,const RaceGroundMap &map,
                          float center,float heading,float width,
                          const std::vector<std::array<float,3>> &localCones,int direction,
                          RaceObstaclePreviewPlan &out) {
        const Lane lane{center,heading,width};
        // 状态含横向位移和路径斜率，限制曲率变化；不对碰撞检查后的路径再做平滑。
        constexpr float dx=.002f,dz=.025f,range=.55f,inf=1e20f;
        constexpr int nx=551,nv=9,zero=275;
        const int nz=int(cfg.farDistance/dz)+1;
        struct Obstacle {float x,z,r;};std::vector<Obstacle> obstacles;
        for(const auto &c:localCones)obstacles.push_back({c[0],c[1],c[2]});
        const float half=cfg.vehicleWidth*.5f+cfg.clearance;
        out.reason="GROUND_ROUTE_SEARCH_EXHAUSTED";
        float nearest=inf;
        for(const auto &c:obstacles)if(c.z+c.r+cfg.vehicleLength+.04f>=0&&c.z<nearest) {
            nearest=c.z;out.coneX=c.x;out.coneZ=c.z;out.coneRadius=c.r;
            out.leftGap=c.x-c.r-(lane.center+lane.slope*c.z-lane.width*.5f);
            out.rightGap=lane.center+lane.slope*c.z+lane.width*.5f-c.x-c.r;
        }
        auto free=[&](float x,float z,float slope) {
            const float norm=std::hypot(1.f,slope);
            const cv::Point2f forward(slope/norm,1.f/norm),normal(forward.y,-forward.x),front(x,z);
            const float rear=-cfg.vehicleLength-.04f,nose=.04f;
            // 31cm长车身转向后，车尾的横向摆幅不能仍按18cm轴对齐宽度计算。
            for(float along:{rear,nose})for(float across:{-half,half}) {
                const auto corner=front+forward*along+normal*across;
                if(std::abs(corner.x-lane.center-lane.slope*corner.y)>lane.width*.5f-.003f)return false;
            }
            for(const auto &c:obstacles) {
                const cv::Point2f relative(c.x-x,c.z-z);
                const float along=relative.dot(forward),across=relative.dot(normal);
                const float dz=std::max({rear-along,0.f,along-nose}),dx=std::max(std::abs(across)-half,0.f);
                if(std::hypot(dx,dz)<c.r+.003f)return false;
                if(direction&&along>=rear-c.r&&along<=nose+c.r&&
                    ((direction<0&&x>=c.x)||(direction>0&&x<=c.x)))return false;
            }
            return true;
        };
        if(!free(0,0,0)){out.reason="GROUND_CURRENT_FOOTPRINT_BLOCKED";out.routeBlockedZ=0;return false;}
        // 提前进入可用通道的中部，保持到整段车身通过，再平滑回中。
        // 这是搜索代价，不放宽道路/锥桶/车身扫掠的硬碰撞检查。
        auto reference=[&](float z) {
            const float middle=lane.center+lane.slope*z;
            float offset=0;
            for(const auto &c:obstacles) {
                const float entry=std::clamp((z-(c.z-.45f))/.30f,0.f,1.f);
                const float exit=std::clamp((c.z+c.r+cfg.vehicleLength+.34f-z)/.30f,0.f,1.f);
                const float weight=entry*entry*(3-2*entry)*exit*exit*(3-2*exit);
                const float inner=c.x+direction*(c.r+half+.003f);
                const float outer=middle+direction*(lane.width*.5f-half-.003f);
                const float desired=(inner+outer)*.5f-middle;
                const float candidate=desired*weight;
                if(direction*candidate>direction*offset)offset=candidate;
            }
            return middle+offset;
        };
        std::vector<float> prev(nx*nv,inf),next(nx*nv,inf);
        std::vector<std::vector<int16_t>> parents(nz,std::vector<int16_t>(nx*nv,-1));
        prev[zero*nv+4]=0;
        for(int row=1;row<nz;++row) {
            std::fill(next.begin(),next.end(),inf);const float z=row*dz,target=reference(z);
            for(int x=0;x<nx;++x) {
                const float px=x*dx-range;
                for(int velocity=0;velocity<nv;++velocity) {
                    const int shift=velocity-4,oldx=x-shift;
                    if(oldx<0||oldx>=nx)continue;
                    const float slope=shift*dx/dz;
                    if(!free(px,z,slope))continue;
                    // 每25mm横向斜率至多改变0.08，路径曲率约束3.2/m。
                    for(int oldv=std::max(0,velocity-1);oldv<=std::min(nv-1,velocity+1);++oldv) {
                        const int index=oldx*nv+oldv;
                        if(prev[index]>=inf/2)continue;
                        // 连续的过渡段也检查，避免两节点间横穿膨胀障碍。
                        bool segment=true;
                        for(int part=1;part<5;++part) {
                            const float alpha=part/5.f;
                            if(!free((oldx*dx-range)*(1-alpha)+px*alpha,z-dz+dz*alpha,
                                ((oldv-4)*(1-alpha)+shift*alpha)*dx/dz)){segment=false;break;}
                        }
                        if(!segment)continue;
                        const float e=px-target;
                        const float value=prev[index]+40*e*e+.0008f*shift*shift+.002f*(velocity-oldv)*(velocity-oldv);
                        const int node=x*nv+velocity;
                        if(value<next[node]){next[node]=value;parents[row][node]=int16_t(index);}
                    }
                }
            }
            prev.swap(next);
            if(*std::min_element(prev.begin(),prev.end())>=inf/2){out.routeBlockedZ=z;return false;}
        }
        int node=int(std::min_element(prev.begin(),prev.end())-prev.begin());if(prev[node]>=inf/2)return false;
        std::vector<cv::Point2f> path(nz);
        for(int row=nz-1;row>=0;--row) {
            path[row]={float(node/nv)*dx-range,row*dz};
            if(!free(path[row].x,path[row].y,(node%nv-4)*dx/dz))return false;
            if(row){node=parents[row][node];if(node<0)return false;}
        }
        if(!raceGroundProjectPath(map,cfg,path,out)){out.reason="GROUND_ROUTE_IMAGE_PROJECTION_SHORT";return false;}
        out.reason="CALIBRATED_STRAIGHT_CONE_TRIAL";out.routeBlockedZ=-1;return true;
    }
    explicit RaceGroundObstacle(const RaceGroundCalibration &c):cfg(c),map(c),motion(c) {}
    RaceObstaclePreviewPlan update(const RaceSnapshot &s,double now,const cv::Mat &bin,const Track &track,const RaceConfig &race,
                                   bool committed=true,bool automatic=true) {
        RaceObstaclePreviewPlan out;out.metric=true;out.sequence=s.sequence;out.captured=s.captured;
        out.historyCommitted=committed;
        out.groundVehicleWidth=cfg.vehicleWidth;out.groundVehicleLength=cfg.vehicleLength;
        out.groundRoadWidth=cfg.roadWidth;out.groundClearance=cfg.clearance;
        out.reason="CALIBRATED_STRAIGHT_CONE_TRIAL";
        out.fresh=!s.failed&&s.sequence&&std::isfinite(now)&&now>=s.captured&&now-s.captured<=race.maxDetectionAge;
        if(!out.fresh){out.state="BYPASS_AI_STALE";return out;}
        if(s.sequence==sequence)return latest;
        if(s.sequence<sequence||s.captured<=captured){out.state="BYPASS_NONMONOTONIC";return out;}
        const double gap=captured<0?0:s.captured-captured;sequence=s.sequence;captured=s.captured;
        // 先取路沿并确定米制尺度，再换算锥桶和路线；三者必须用同一个坐标系。
        // AUTO尺度已冻结时先算当前真实运动，避免恢复首帧仍用失效前车姿遮罩路沿。
        std::optional<RaceGroundMotionResult> currentMotion;
        if(scaleLocked&&automatic)currentMotion=motion.update(s,committed);
        std::vector<Row> rows;std::string laneIssue;
        const bool rowsOk=collectRows(bin,track,s,race,rows,laneIssue,currentMotion?&*currentMotion:nullptr);
        std::string scaleIssue;
        if(rowsOk)updateScale(rows,automatic,scaleIssue);
        // 尺度先更新，再计算里程与锥桶；尺度变更时重置关键帧，避免混用两套米制。
        const auto odo=motion.update(s,committed);out.motionValid=odo.valid;out.motionEpoch=odo.epoch;
        if(!committed)lockedDirection=0;
        out.motionInliers=odo.inliers;out.motionResidual=odo.residual;
        out.motionFeatures=odo.features;out.motionMatches=odo.matches;out.motionMisses=odo.misses;out.motionReferenceAge=odo.referenceAge;
        out.motionCoordinatesHeld=odo.coordinatesHeld;out.motionReason=odo.reason;
        out.motionFitScale=odo.fitScale;out.motionFitTranslation=odo.fitTranslation;out.motionFitAngle=odo.fitAngle;
        out.motionSpanX=odo.spanX;out.motionSpanZ=odo.spanZ;
        out.motionBackupReferenceUsed=odo.backupReferenceUsed;out.motionBackupAttempts=odo.backupAttempts;
        out.motionX=odo.pose.x;out.motionZ=odo.pose.z;out.motionAngle=odo.pose.angle;
        if(epoch!=odo.epoch){cones.clear();lockedDirection=0;epoch=odo.epoch;}
        out.laneScale=map.scale();out.laneWidthRaw=scaleLocked?lockedWidth:pendingWidth;
        out.laneScaleWanted=scaleLocked?map.scale():pendingScale;
        std::optional<Lane> corridor;
        if(!scaleLocked) {
            laneIssue=rowsOk?(scaleIssue.empty()?"GROUND_LANE_SCALE_ESTIMATING":scaleIssue):laneIssue;
        } else if(rowsOk) {
            std::string issue;corridor=lane(rows,issue,out);
            if(!corridor)laneIssue=issue;
        }
        if(corridor) {
            out.laneOffset=corridor->center;out.laneHeading=corridor->slope;out.laneWidthMeasured=measuredWidth;
        }
        std::vector<bool> used(cones.size(),false);
        bool human=false,invalidCone=false;
        for(const auto &d:s.detections)if(reviewRaceDetection(d,s.image,race,&s.crosswalk).accepted) {
            if(d.type==LABEL_PERSON){human=true;continue;}
            if(d.type!=LABEL_CONE)continue;
            out.obstacles.emplace_back(d.x,d.y,d.width,d.height);
            const auto l=map.toGround({float(d.x),float(d.y+d.height)}),r=map.toGround({float(d.x+d.width),float(d.y+d.height)});
            const cv::Point2f p=(l+r)*.5f;
            if(std::isfinite(p.y)&&p.y>cfg.farDistance)continue; // 尚在规划范围外，仍低速接近
            const bool clipped=d.y+d.height>=236||d.x<=2||d.x+d.width>=318;
            if(clipped) {
                const bool mapped=std::any_of(cones.begin(),cones.end(),[&](const Cone &c){
                    const auto local=odo.pose.local(c.world);if(!c.near||local.y<=.07f)return false;
                    const auto foot=map.toImage(local),left=map.toImage(local-cv::Point2f(c.radius,0)),right=map.toImage(local+cv::Point2f(c.radius,0));
                    const bool depthMatches=d.y+d.height>=236?local.y<=p.y+.04f:std::abs(local.y-p.y)<=.04f;
                    return std::isfinite(foot.y)&&depthMatches&&
                        right.x>d.x&&left.x<d.x+d.width;
                });
                if(mapped)continue; // 仅已建图且位置相符的近锥桶由连续视觉里程维持
                invalidCone=true;continue;
            }
            if(!std::isfinite(p.x)||!std::isfinite(p.y)||p.y<0||p.y>cfg.farDistance||d.y+d.height>=236||
               clipped||std::abs(l.y-r.y)>.06f||r.x-l.x<.015f||r.x-l.x>.18f){invalidCone=true;continue;}
            // 路外锥桶不启动绕行；空间不够确定时宁可停车。
            if(corridor&&std::abs(p.x-corridor->center-corridor->slope*p.y)>corridor->width*.5f+.02f)continue;
            const auto world=odo.pose.world(p);const float radius=std::max(.025f,(r.x-l.x)*.5f)+.01f;
            int index=-1;float distance=.04f;
            for(size_t i=0;i<cones.size();++i)if(!used[i]&&cv::norm(world-cones[i].world)<distance){distance=cv::norm(world-cones[i].world);index=int(i);}
            if(index<0){cones.push_back({world,radius,1,s.captured,p.y<.30f,cv::Rect(d.x,d.y,d.width,d.height)});used.push_back(true);}
            else {auto &c=cones[index];used[index]=true;c.world=world;c.radius=std::max(radius,c.radius);
                c.count=gap<=race.confirmGap?std::min(race.confirmFrames,c.count+1):1;c.seen=s.captured;c.near=c.near||p.y<.30f;
                c.box={d.x,d.y,d.width,d.height};}
        }
        // 待启动时是重新采集目标，并非已经在绕行。漏检和框底抖动不能永久
        // 留下旧目标，让同一个锥桶重新出现后仍被另一条历史记录锁死。
        // AUTO待启动仅允许用同位置的明确新检测替换旧记录；没有新检测不能放行。
        std::vector<Cone> retained;std::vector<bool> retainedUsed;
        for(size_t i=0;i<cones.size();++i) {
            bool replaced=false;
            if(!committed&&!used[i])for(size_t j=0;j<cones.size();++j)if(used[j]) {
                const auto overlap=cones[i].box&cones[j].box;
                const float iou=float(overlap.area())/std::max(1,cones[i].box.area()+cones[j].box.area()-overlap.area());
                if(iou>.3f&&cv::norm(cones[i].world-cones[j].world)<.12f){replaced=true;break;}
            }
            const bool discard=!used[i]&&(cones[i].count<race.confirmFrames||(!committed&&(!automatic||replaced)));
            if(discard){++out.discardedTargets;continue;}
            retained.push_back(cones[i]);retainedUsed.push_back(used[i]);
        }
        cones=std::move(retained);used=std::move(retainedUsed);out.trackedTargets=int(cones.size());
        out.hasTarget=!cones.empty()||invalidCone;
        auto finish=[&](const char *state){out.state=state;latest=out;return out;};
        if(human){out.hasTarget=true;return finish("BYPASS_PERSON_STOP");}
        if(invalidCone)return finish("BYPASS_CONE_OUTSIDE_CALIBRATED_RANGE");
        if(!corridor){out.reason=laneIssue;return finish("BYPASS_STRAIGHT_LANE_REQUIRED");}
        if(cones.empty()){out.stop=false;return finish("BYPASS_NO_TARGET");}
        if(cones.size()>8)return finish("BYPASS_TOO_MANY_CONES");
        if(!odo.valid)return finish("BYPASS_VISUAL_MOTION_REQUIRED");
        bool allPassed=true;
        std::vector<std::array<float,3>> localCones;
        for(size_t i=0;i<cones.size();++i) {
            const auto &c=cones[i];const auto p=odo.pose.local(c.world);
            if(c.count<race.confirmFrames)return finish("BYPASS_CONFIRMING");
            ++out.confirmedTargets;
            const bool passed=p.y+c.radius+cfg.vehicleLength+.04f<0;
            allPassed=allPassed&&passed;
            // 远处丢失不能视作"已绕过"。近处也必须持续有地面视觉里程。
            const double age=s.captured-c.seen;
            const bool predicted=!used[i]&&!passed&&committed&&odo.valid&&age>=0&&age<=.35;
            if(!used[i]&&!passed&&!predicted&&(!c.near||std::abs(p.x)<cfg.vehicleWidth*.5f+c.radius+cfg.clearance))
                return finish("BYPASS_TARGET_LOST_BEFORE_CLEARANCE");
            out.predictedTargets+=predicted;
            localCones.push_back({p.x,p.y,c.radius+(predicted?.005f+.02f*float(age):0.f)});
        }
        out.passed=allPassed;
        int direction=lockedDirection;
        if(!direction) {
            float nearest=1e9f;
            for(const auto &c:localCones)if(c[1]+c[2]+cfg.vehicleLength+.04f>=0&&c[1]<nearest) {
                nearest=c[1];direction=c[0]<corridor->center+corridor->slope*c[1]?1:-1;
            }
        }
        if(!solveRoute(cfg,map,corridor->center,corridor->slope,corridor->width,localCones,direction,out))
            return finish("BYPASS_NO_CLEAR_PATH");
        out.candidate=true;out.stop=false;
        if(!lockedDirection&&!out.groundCenters.empty()) {
            lockedDirection=direction;
        }
        out.direction=lockedDirection;
        if(allPassed)lockedDirection=0;
        return finish(allPassed?"BYPASS_RETURN":lockedDirection>0?"BYPASS_RIGHT":"BYPASS_LEFT");
    }
};
