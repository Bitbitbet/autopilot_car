#pragma once
#include "race_ground.hpp"
#include "race_perception.hpp"
#include "race_detection_review.hpp"
#include <algorithm>
#include <string_view>
#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/video/tracking.hpp>

struct RaceGroundPose {
    float x=0,z=0,angle=0;
    cv::Point2f world(cv::Point2f p) const {
        const float c=std::cos(angle),s=std::sin(angle);return {x+c*p.x-s*p.y,z+s*p.x+c*p.y};
    }
    cv::Point2f local(cv::Point2f p) const {
        const float c=std::cos(angle),s=std::sin(angle);p-=cv::Point2f(x,z);return {c*p.x+s*p.y,-s*p.x+c*p.y};
    }
};
struct RaceGroundMotionResult {
    bool valid=false;int inliers=0,epoch=0,features=0,matches=0,misses=0;float residual=0,referenceAge=0;
    RaceGroundPose pose;
    std::string reason="REFERENCE_INITIALIZING";
    float fitScale=0,fitTranslation=0,fitAngle=0,spanX=0,spanZ=0;
    bool coordinatesHeld=false;
    bool estimatedOrigin=false;
    bool backupReferenceUsed=false;
    int backupAttempts=0;
    bool absoluteProbe=false,absoluteRecovered=false;
    int absoluteMatches=0,absoluteReference=0;
};

struct RaceGroundFitDiagnostics {
    const char *reason="TOO_FEW_MATCHES";
    float scale=0,translation=0,angle=0,spanX=0,spanZ=0;
};

// 只用地面纹理计算相邻图像的刚体运动；速度指令和等待时间都不作为里程。
// 纹理不够、图像跳帧或尺度变化时报告无效，由绕行控制器停车。
class RaceGroundMotion {
    RaceGroundCalibration cfg;RaceGroundMap map;
    cv::Mat previous,previousMask;
    double captured=-1;
    uint64_t sequence=0;
    RaceGroundMotionResult result;
    bool discontinuous=true;
    int misses=0;
    struct Region {cv::Rect box;double captured;};
    std::vector<Region> regions;
    // features 缓存该参考图自己的角点：恢复尝试时只需再做双向光流，
    // 不再每帧对每张参考图重跑 goodFeaturesToTrack。
    struct Reference {cv::Mat image,mask;double captured;RaceGroundPose pose;std::vector<cv::Point2f> features;};
    std::vector<Reference> references;
    size_t backupCursor=0;          // 备用参考轮转起点：每帧最多尝试一张
    float rigidResidualLimit=.005f;
    bool parkingMode=false;

    // Keep a camera image whose pose is in the original parking frame even
    // when relative LK is restarted from a command-estimated origin.
    struct ParkingAnchor {
        Reference view;
        std::vector<cv::KeyPoint> keys;
        cv::Mat descriptors;
        bool prepared=false;
    };
    ParkingAnchor absoluteAnchor;
    std::vector<ParkingAnchor> parkingViews;
    size_t probeView=0;
    cv::Mat recoveryMask;
    cv::Ptr<cv::ORB> orb;
    double lastProbe=-1e20,pendingAt=-1;
    RaceGroundPose pendingPose;

    bool relocalize(ParkingAnchor &anchor,const cv::Mat &gray,const cv::Mat &currentMask,double now,RaceGroundMotionResult &out) {
        out.absoluteProbe=true;lastProbe=now;
        const double priorTime=pendingAt;pendingAt=-1;
        cv::Mat roi;
        const auto &view=anchor.view;
        if(!anchor.prepared) {
            cv::bitwise_and(view.mask,recoveryMask,roi);
            orb->detectAndCompute(view.image,roi,anchor.keys,anchor.descriptors);
            anchor.prepared=true;
        }
        if(anchor.descriptors.rows<16)return false;
        std::vector<cv::KeyPoint> keys;cv::Mat descriptors;
        cv::bitwise_and(currentMask,recoveryMask,roi);
        orb->detectAndCompute(gray,roi,keys,descriptors);
        if(descriptors.rows<16)return false;
        cv::BFMatcher matcher(cv::NORM_HAMMING);
        std::vector<std::vector<cv::DMatch>> forward,back;
        matcher.knnMatch(anchor.descriptors,descriptors,forward,2);
        matcher.knnMatch(descriptors,anchor.descriptors,back,2);
        const auto distinct=[](const auto &pair){return pair.size()==2&&pair[0].distance<=48&&pair[0].distance<.72f*pair[1].distance;};
        std::vector<cv::Point2f> a,b;
        for(const auto &pair:forward)if(distinct(pair)) {
            const auto &m=pair.front();const auto &reverse=back[m.trainIdx];
            if(!distinct(reverse)||reverse.front().trainIdx!=m.queryIdx)continue;
            a.push_back(map.toGround(anchor.keys[m.queryIdx].pt));b.push_back(map.toGround(keys[m.trainIdx].pt));
        }
        out.absoluteMatches=int(a.size());if(a.size()<16)return false;
        auto pose=view.pose;RaceGroundFitDiagnostics info;int inliers=0;float residual=0;
        if(!fit(a,b,pose,inliers,residual,&info,float(now-view.captured),.005f)||
           residual>.004f||std::abs(info.scale-1.f)>.01f)return false;
        // Reject ambiguous matches confined to one divider or one small patch.
        std::vector<cv::Point2f> consistent,hull;
        for(size_t i=0;i<a.size();++i)if(cv::norm(pose.local(view.pose.world(a[i]))-b[i])<=.006f)
            consistent.push_back(a[i]);
        if(consistent.size()<16||consistent.size()<.85f*a.size())return false;
        cv::convexHull(consistent,hull);if(cv::contourArea(hull)<.012)return false;
        // Adjacent bay markings can repeat. An older view may locate the car
        // near its last measured position, not teleport it to another bay.
        const auto &lastMeasured=absoluteAnchor.view.pose;
        if(std::hypot(pose.x-lastMeasured.x,pose.z-lastMeasured.z)>.30f||
           std::abs(std::remainder(pose.angle-lastMeasured.angle,float(2*CV_PI)))>.65f)return false;
        const double dt=now-priorTime;
        if(priorTime<0||dt<=0||dt>.25||std::hypot(pose.x-pendingPose.x,pose.z-pendingPose.z)>.008+.42*dt||
           std::abs(std::remainder(pose.angle-pendingPose.angle,float(2*CV_PI)))>.025+1.4*dt) {
            pendingAt=now;pendingPose=pose;return false;
        }
        out.pose=pose;out.valid=true;out.estimatedOrigin=false;out.absoluteRecovered=true;
        out.reason="PARK_ABSOLUTE_VISUAL_RECOVERED";out.matches=out.absoluteMatches;out.inliers=inliers;
        out.features=int(keys.size());out.residual=residual;out.referenceAge=float(now-view.captured);
        out.fitScale=info.scale;out.fitAngle=info.angle;out.fitTranslation=info.translation;
        out.spanX=info.spanX;out.spanZ=info.spanZ;return true;
    }

    cv::Mat mask(const RaceSnapshot &s) {
        regions.erase(std::remove_if(regions.begin(),regions.end(),[&](const Region &r){return s.captured-r.captured>.65;}),regions.end());
        for(const auto &d:s.detections)if(raceDetectionBoxValid(d))regions.push_back({{d.x,d.y,d.width,d.height},s.captured});
        if(regions.size()>64)regions.erase(regions.begin(),regions.end()-64);
        cv::Mat output=cv::Mat::zeros(480,640,CV_8UC1);
        for(int y=20;y<420;++y)for(int x=10;x<630;++x) {
            const auto p=map.toGround({float(x),float(y)});
            // 同一平面的路旁纹理也可参与里程，不再被65cm规划终点截掉。
            // 限制在近地带；物体框继续剔除，刚体拟合仍检验尺度和残差。
            if(std::isfinite(p.x)&&std::abs(p.x)<.55f&&p.y>=.12f&&p.y<=1.f)output.at<uchar>(y,x)=255;
        }
        for(const auto &r:regions) {
            const auto &b=r.box;
            cv::rectangle(output,(cv::Rect(b.x-24,b.y-24,b.width+48,b.height+56)&cv::Rect(0,0,640,480)),{0},-1);
        }
        // 当前车保存图中车头蓝色外壳占据下方中央，不能作为静止地面特征。
        cv::rectangle(output,{90,380,480,100},{0},-1);
        return output;
    }
public:
    explicit RaceGroundMotion(const RaceGroundCalibration &c):cfg(c),map(c) {}
    // Parking's 320x240 field sample has 42/42 consistent tracks and 1.25%
    // scale variation, but 5.087mm rigid residual. Do not permanently freeze
    // the reference over that 0.087mm excess. Keep this opt-in for parking;
    // RANSAC, scale, span and motion-step checks are still applied.
    void enableParkingNoiseTolerance(){
        rigidResidualLimit=.008f;parkingMode=true;orb=cv::ORB::create(500,1.2f,8,19,0,2,cv::ORB::HARRIS_SCORE,31,10);
        // Near ground beside the bay is often textured while the central
        // calibration rectangle contains just black asphalt. Use the same
        // measured map and dynamic object/body mask as local tracking; cap
        // depth at the far calibration mark for descriptor relocalization.
        recoveryMask=cv::Mat::zeros(480,640,CV_8UC1);
        for(int y=20;y<420;++y)for(int x=10;x<630;++x) {
            const auto p=map.toGround({float(x),float(y)});
            if(std::abs(p.x)<=cfg.roadWidth&&p.y>=.12f&&p.y<=cfg.farDistance)recoveryMask.at<uchar>(y,x)=255;
        }
    }
    // Recover relative visual tracking after a parking-only command estimate.
    // The caller supplies the propagated pose at THIS frame, never the old
    // frozen pose. Absolute coordinates remain explicitly estimated thereafter.
    bool reanchorParkingEstimate(const RaceSnapshot &s,const RaceGroundPose &pose) {
        if(!parkingMode||s.failed||s.sequence!=sequence||s.captured<=captured||
           !std::isfinite(s.captured)||s.image.type()!=CV_8UC3||s.image.size()!=cv::Size(640,480)||
           !std::isfinite(pose.x)||!std::isfinite(pose.z)||!std::isfinite(pose.angle)||
           result.valid||misses<3)return false;
        // previous may share pixels with an absolute keyframe. Never write
        // the estimated-origin image into that original-coordinate reference.
        cv::Mat gray;cv::cvtColor(s.image,gray,cv::COLOR_BGR2GRAY);previous=gray;previousMask=mask(s);
        captured=s.captured;result.pose=pose;result.estimatedOrigin=true;
        references.clear();misses=0;result.misses=0;
        result.reason="PARKING_ESTIMATED_REFERENCE_REANCHORED";
        return true;
    }
    // 与绕行规划共用同一米制尺度；里程增量必须与路线在同一个坐标系里。
    void setMetricScale(float s) {
        if(std::abs(s-map.scale())<1e-6f)return;
        map.setMetricScale(s);previous.release();previousMask.release();captured=-1;
        absoluteAnchor={};parkingViews.clear();probeView=0;lastProbe=-1e20;pendingAt=-1;
        result.pose={};result.valid=false;result.estimatedOrigin=false;++result.epoch;misses=0;discontinuous=true;regions.clear();references.clear();
    }
    // 也供回归测试验证米制刚体拟合，不接受像素/运动指令冒充位移。
    static bool fit(const std::vector<cv::Point2f> &before,const std::vector<cv::Point2f> &after,
                    RaceGroundPose &pose,int &inliers,float &residual,
                    RaceGroundFitDiagnostics *diagnostics=nullptr,float referenceAge=.07f,float residualLimit=.005f,
                    bool checkArea=false) {
        RaceGroundFitDiagnostics info;
        auto fail=[&](const char *reason){info.reason=reason;if(diagnostics)*diagnostics=info;return false;};
        inliers=0;residual=0;
        if(before.size()<8||before.size()!=after.size())return fail("TOO_FEW_MATCHES");
        cv::Mat selected;
        const auto transform=cv::estimateAffinePartial2D(before,after,selected,cv::RANSAC,.008,300,.995,10);
        if(transform.empty())return fail("ROBUST_FIT_EMPTY");
        const double a=transform.at<double>(0,0),b=transform.at<double>(1,0);
        const double scale=std::hypot(a,b);
        info.scale=float(scale);
        // RANSAC只用于剔除错误匹配。必须重新拟合单位尺度的刚体，不能
        // 取含尺度模型的平移再丢掉尺度，否则会系统性少计/多计地面距离。
        cv::Point2d meanBefore(0,0),meanAfter(0,0);int count=0;
        for(size_t i=0;i<before.size();++i)if(selected.at<uchar>(int(i))){meanBefore+=cv::Point2d(before[i]);meanAfter+=cv::Point2d(after[i]);++count;}
        inliers=count;
        if(count<8)return fail("TOO_FEW_INLIERS");
        meanBefore*=1./count;meanAfter*=1./count;double dot=0,cross=0,xx=0,zz=0,xz=0;
        for(size_t i=0;i<before.size();++i)if(selected.at<uchar>(int(i))) {
            const auto p=cv::Point2d(before[i])-meanBefore,q=cv::Point2d(after[i])-meanAfter;
            dot+=p.x*q.x+p.y*q.y;cross+=p.x*q.y-p.y*q.x;
            if(checkArea){xx+=p.x*p.x;zz+=p.y*p.y;xz+=p.x*p.y;}
        }
        if(checkArea) {
            // A diagonal row of features can pass both bounding-box spans.
            // Require spread across the weaker axis as well (2 residual gates RMS).
            const double minorVariance=.5*(xx+zz-std::hypot(xx-zz,2*xz))/count;
            if(minorVariance<4*residualLimit*residualLimit)return fail("GROUND_SUPPORT_COLLINEAR");
        }
        const double theta=std::atan2(cross,dot);
        const float c=std::cos(theta),s=std::sin(theta);
        const double tx=meanAfter.x-c*meanBefore.x+s*meanBefore.y,tz=meanAfter.y-s*meanBefore.x-c*meanBefore.y;
        info.translation=float(std::hypot(tx,tz));info.angle=float(theta);
        // 跨丢帧拟合的位移上界按真实图像间隔增加；不使用速度指令积分。
        const float extra=std::clamp(referenceAge-.08f,0.f,.57f);
        if(std::abs(theta)>.09f+.8f*extra||info.translation>.04f+.4f*extra)return fail("GROUND_STEP_TOO_LARGE");
        float sum=0;inliers=0;cv::Point2f low(100,100),high(-100,-100);
        for(size_t i=0;i<before.size();++i)if(selected.at<uchar>(int(i))) {
            const auto p=before[i];const cv::Point2f predicted(c*p.x-s*p.y+tx,s*p.x+c*p.y+tz);
            sum+=cv::norm(predicted-after[i]);++inliers;
            low.x=std::min(low.x,p.x);low.y=std::min(low.y,p.y);high.x=std::max(high.x,p.x);high.y=std::max(high.y,p.y);
        }
        info.spanX=high.x-low.x;info.spanZ=high.y-low.y;
        if(inliers<8||inliers<int(before.size()*.7))return fail("INLIER_RATIO_LOW");
        if(info.spanX<.08f||info.spanZ<.15f)return fail("GROUND_FEATURE_SPAN_SHORT");
        residual=sum/inliers;if(residual>std::clamp(residualLimit,.005f,.008f))return fail("RIGID_RESIDUAL_HIGH");
        if(!std::isfinite(scale)||scale<.98||scale>1.02)return fail("GROUND_SCALE_CHANGED");
        const auto position=pose.world({float(-c*tx-s*tz),float(s*tx-c*tz)});
        pose.x=position.x;pose.z=position.y;pose.angle-=theta;
        info.reason="GROUND_RIGID_OK";if(diagnostics)*diagnostics=info;return true;
    }
    // Reuse already-tracked points when a wide-field fit shows scale distortion.
    // Project the allowed LK round-trip tolerance (1.4 px) through the local
    // ground map at BOTH ends. Only retry on tracks whose combined metric
    // sensitivity fits inside the unchanged RANSAC residual budget.
    bool fitMetricSupport(const std::vector<cv::Point2f> &before,const std::vector<cv::Point2f> &after,
                          RaceGroundPose &pose,int &matches,int &inliers,float &residual,
                          RaceGroundFitDiagnostics &info,float age,bool nearBand=false) const {
        if(before.size()<12||before.size()!=after.size())return false;
        auto metresPerPixel=[&](cv::Point2f floor){
            const auto pixel=map.toImage(floor);
            const auto dx=map.toGround(pixel+cv::Point2f{1,0})-floor;
            const auto dy=map.toGround(pixel+cv::Point2f{0,1})-floor;
            return std::sqrt(dx.dot(dx)+dy.dot(dy));
        };
        std::vector<cv::Point2f> a,b;
        const auto nearGround=[&](cv::Point2f p){return p.y>=.12f&&p.y<=cfg.farDistance*map.scale()&&
            std::abs(p.x)<=cfg.roadWidth*map.scale();};
        for(size_t i=0;i<before.size();++i) {
            const bool usable=nearBand?nearGround(before[i])&&nearGround(after[i]):
                1.4f*(metresPerPixel(before[i])+metresPerPixel(after[i]))<=rigidResidualLimit;
            if(usable) {
                a.push_back(before[i]);b.push_back(after[i]);
            }
        }
        if(a.size()<12||a.size()==before.size())return false;
        // Independent scale check over all well-conditioned long baselines.
        // A compact RANSAC subset alone must not hide coherent image zoom.
        std::vector<float> scaleRatios;
        for(size_t i=0;i<a.size();++i)for(size_t j=0;j<i;++j) {
            const float length=cv::norm(a[i]-a[j]);
            if(length>=.15f)scaleRatios.push_back(cv::norm(b[i]-b[j])/length);
        }
        if(scaleRatios.size()<12)return false;
        const auto middle=scaleRatios.begin()+scaleRatios.size()/2;
        std::nth_element(scaleRatios.begin(),middle,scaleRatios.end());
        if(*middle<.98f||*middle>1.02f)return false;
        int supported=0;float error=0;RaceGroundFitDiagnostics local;
        if(!fit(a,b,pose,supported,error,&local,age,rigidResidualLimit,true))return false;
        matches=int(a.size());inliers=supported;residual=error;info=local;
        info.reason=nearBand?"GROUND_RIGID_OK_NEAR_CALIBRATED_DEPTH":"GROUND_RIGID_OK_METRIC_SUPPORT";return true;
    }
    // Busy bay edges can supply a mixture of ground and off-plane tracks.
    // Recover only a dominant, spatially spread rigid surface whose forward
    // AND inverse reprojections agree with the tracked image points.
    bool fitParkingConsensus(const std::vector<cv::Point2f> &a,const std::vector<cv::Point2f> &b,
                             RaceGroundPose &pose,int &inliers,float &residual,RaceGroundFitDiagnostics &info,float age)const {
        if(a.size()<32||a.size()!=b.size())return false;
        cv::Mat selected;
        const auto transform=cv::estimateAffinePartial2D(a,b,selected,cv::RANSAC,.005,300,.995,10);
        if(transform.empty())return false;
        const auto scale=std::hypot(transform.at<double>(0,0),transform.at<double>(1,0));
        if(scale<.98||scale>1.02)return false;
        std::vector<cv::Point2f> src,dst,others,nextOthers;
        for(size_t i=0;i<a.size();++i){
            if(selected.at<uchar>(int(i))){src.push_back(a[i]);dst.push_back(b[i]);}
            else {others.push_back(a[i]);nextOthers.push_back(b[i]);}
        }
        if(src.size()<24||src.size()<.55f*a.size())return false;
        if(others.size()>=16&&others.size()>=.25f*a.size()){
            cv::Mat competing;
            const auto other=cv::estimateAffinePartial2D(others,nextOthers,competing,cv::RANSAC,.005,200,.99,5);
            if(!other.empty()&&cv::countNonZero(competing)>=std::max(16,int(std::ceil(.25f*a.size()))))return false;
        }
        auto candidate=pose;RaceGroundFitDiagnostics details;int supported=0;float error=0;
        if(!fit(src,dst,candidate,supported,error,&details,age,.005f,true)||error>.004f)return false;
        std::vector<cv::Point2f> consistent,hull;
        for(size_t i=0;i<src.size();++i){
            const auto forward=candidate.local(pose.world(src[i])),back=pose.local(candidate.world(dst[i]));
            if(cv::norm(forward-dst[i])<=.006f&&cv::norm(map.toImage(forward)-map.toImage(dst[i]))<=2.5f&&
               cv::norm(map.toImage(back)-map.toImage(src[i]))<=2.5f)consistent.push_back(src[i]);
        }
        if(consistent.size()<24||consistent.size()<.55f*a.size())return false;
        cv::convexHull(consistent,hull);if(cv::contourArea(hull)<.025)return false;
        pose=candidate;inliers=int(consistent.size());residual=error;info=details;
        info.reason="GROUND_RIGID_OK_PARKING_CONSENSUS";return true;
    }
    RaceGroundMotionResult update(const RaceSnapshot &s,bool preserveCoordinates=false,bool allowReanchor=true) {
        if(s.sequence==sequence)return result;
        result.valid=false;result.inliers=0;result.residual=0;result.features=result.matches=0;
        result.referenceAge=0;result.misses=misses;
        result.reason="REFERENCE_INITIALIZING";result.coordinatesHeld=false;
        result.backupReferenceUsed=false;result.backupAttempts=0;
        result.absoluteProbe=result.absoluteRecovered=false;result.absoluteMatches=result.absoluteReference=0;
        result.fitScale=result.fitTranslation=result.fitAngle=result.spanX=result.spanZ=0;
        const bool fresh=!s.failed&&s.sequence>sequence&&std::isfinite(s.captured)&&
            s.image.type()==CV_8UC3&&s.image.size()==cv::Size(640,480);
        if(!fresh){result.reason="INVALID_MOTION_FRAME";
            if(preserveCoordinates&&!previous.empty()){result.coordinatesHeld=true;return result;}
            previous.release();references.clear();if(!discontinuous)++result.epoch;result.pose={};discontinuous=true;return result;}
        cv::Mat gray;cv::cvtColor(s.image,gray,cv::COLOR_BGR2GRAY);const auto currentMask=mask(s);
        const double gap=s.captured-captured;
        result.referenceAge=previous.empty()?0:float(gap);
        const auto previousPose=result.pose;
        const double horizon=preserveCoordinates?.65:.28;
        auto attempt=[&](const cv::Mat &reference,const cv::Mat &referenceMask,double age,
                         const RaceGroundPose &origin,RaceGroundMotionResult &candidate,
                         std::vector<cv::Point2f> *cached=nullptr) {
            std::vector<cv::Point2f> features;
            if(cached&&!cached->empty())features=*cached;     // 复用已算好的角点，省一次检测
            else {
                cv::goodFeaturesToTrack(reference,features,160,.01,10,referenceMask,5);
                if(cached)*cached=features;
            }
            candidate.features=int(features.size());candidate.referenceAge=float(age);
            std::vector<cv::Point2f> before,after;
            std::vector<cv::Point2f> forward,back;std::vector<uchar> found,returned;std::vector<float> errors,backErrors;
            if(!features.empty()) {
                cv::calcOpticalFlowPyrLK(reference,gray,features,forward,found,errors,{41,41},2);
                cv::calcOpticalFlowPyrLK(gray,reference,forward,back,returned,backErrors,{41,41},2);
            }
            for(size_t i=0;i<features.size();++i) {
                const auto p=features[i],q=forward[i];
                if(!found[i]||!returned[i]||errors[i]>18||backErrors[i]>18||
                    !std::isfinite(q.x)||!std::isfinite(q.y)||q.x<10||q.x>=630||q.y<20||q.y>=420||
                    cv::norm(back[i]-p)>1.4||cv::norm(q-p)>60)continue;
                if(!currentMask.at<uchar>(int(q.y),int(q.x)))continue;
                const auto a=map.toGround(p),b=map.toGround(q);
                if(!std::isfinite(a.x)||!std::isfinite(a.y)||!std::isfinite(b.x)||!std::isfinite(b.y))continue;
                before.push_back(a);after.push_back(b);
            }
            candidate.matches=int(before.size());auto pose=origin;
            RaceGroundFitDiagnostics info;
            candidate.valid=fit(before,after,pose,candidate.inliers,candidate.residual,&info,float(age),rigidResidualLimit);
            if(!candidate.valid&&parkingMode&&(std::string_view(info.reason)=="RIGID_RESIDUAL_HIGH"||
                      std::string_view(info.reason)=="GROUND_SCALE_CHANGED"||std::string_view(info.reason)=="INLIER_RATIO_LOW")) {
                candidate.valid=fitMetricSupport(before,after,pose,candidate.matches,candidate.inliers,
                                                  candidate.residual,info,float(age));
                // The measured calibration depth is a second fixed support
                // region when the sensitivity-only mask has too few corners.
                // Keep the same independent baseline-scale, 2D spread and fit checks.
                if(!candidate.valid)candidate.valid=fitMetricSupport(before,after,pose,candidate.matches,candidate.inliers,
                                                  candidate.residual,info,float(age),true);
                if(!candidate.valid)
                    candidate.valid=fitParkingConsensus(before,after,pose,candidate.inliers,candidate.residual,info,float(age));
            }
            if(candidate.valid)candidate.pose=pose;
            candidate.reason=info.reason;candidate.fitScale=info.scale;candidate.fitTranslation=info.translation;candidate.fitAngle=info.angle;
            candidate.spanX=info.spanX;candidate.spanZ=info.spanZ;
        };
        // An expired LK reference has no search path back to the original
        // coordinates. Periodically match distinctive ground features directly
        // to the last measured anchor instead. On a probe frame run only one
        // matching pipeline; never add a second full optical-flow pass.
        const bool probe=parkingMode&&preserveCoordinates&&!absoluteAnchor.view.image.empty()&&
            (result.estimatedOrigin||gap>horizon)&&(pendingAt>=0||s.captured-lastProbe>=.30);
        if(probe) {
            result.reason="PARK_ABSOLUTE_VISUAL_SEARCH";
            result.absoluteReference=int(probeView);
            auto &anchor=probeView?parkingViews[parkingViews.size()-probeView]:absoluteAnchor;
            if(!relocalize(anchor,gray,currentMask,s.captured,result)&&pendingAt<0)
                probeView=(probeView+1)%(parkingViews.size()+1);
        } else if(!previous.empty()&&gap>0&&gap<=horizon) {
            // After a rejected short step, the same last frame may remain
            // blurred or tilted. Try a recent independently located view on
            // alternating misses, still using just one LK pipeline per frame.
            Reference *alternate=nullptr;
            if(parkingMode&&preserveCoordinates&&!result.estimatedOrigin&&misses%2==1) {
                for(auto it=parkingViews.rbegin();it!=parkingViews.rend();++it){
                    const double age=s.captured-it->view.captured;
                    if(age>horizon)break;
                    if(captured-it->view.captured>=.045){alternate=&it->view;break;}
                }
            }
            if(alternate){
                result.backupAttempts=1;
                attempt(alternate->image,alternate->mask,s.captured-alternate->captured,alternate->pose,result,&alternate->features);
                result.backupReferenceUsed=result.valid;
            }else attempt(previous,previousMask,gap,previousPose,result);
        } else if(!previous.empty()) {
            result.reason="REFERENCE_TOO_OLD";
        }
        const bool primaryValid=result.valid;
        references.erase(std::remove_if(references.begin(),references.end(),[&](const Reference &r){
            return s.captured-r.captured>1.2||s.captured<=r.captured;
        }),references.end());
        // 退化的最后一帧不能成为唯一恢复依据。只尝试少量已定位旧图，仍按同一
        // 刚体/尺度/内点条件验证，并检查与最后可靠位姿的连续性，不重置地图或盲补里程。
        // 计算预算：每帧只尝试 1 张（轮转），且复用该参考图缓存的角点。
        // 旧实现每帧把 4 张全部重跑角点检测+双向光流，把导航从约 25ms 抬到 58~77ms，
        // 直接导致规划结果超过 0.18s 新鲜度而停车。
        // Parking already has an explicitly labelled applied-command fallback.
        // A second pair of LK passes on a failed frame made field processing
        // take 204-246ms, expiring both the camera frame and the fallback.
        if(!parkingMode&&!result.valid&&preserveCoordinates&&gap>0&&gap<=.8&&!references.empty()) {
            const size_t count=references.size();
            Reference &ref=references[backupCursor%count];
            ++backupCursor;
            if(std::abs(ref.captured-captured)>1e-6) {
                ++result.backupAttempts;auto candidate=result;
                attempt(ref.image,ref.mask,s.captured-ref.captured,ref.pose,candidate,&ref.features);
                const float extra=std::clamp(float(gap)-.08f,0.f,.57f);
                if(candidate.valid&&
                   std::hypot(candidate.pose.x-previousPose.x,candidate.pose.z-previousPose.z)<=.04f+.4f*extra&&
                   std::abs(candidate.pose.angle-previousPose.angle)<=.09f+.8f*extra) {
                    candidate.backupReferenceUsed=true;result=std::move(candidate);
                }
            }
        }
        if(primaryValid&&result.features>=12&&result.matches>=10&&gap>0&&gap<=.18&&
           (references.empty()||captured-references.back().captured>=.12)) {
            references.push_back({previous,previousMask,captured,previousPose,{}});
            if(references.size()>4)references.erase(references.begin());
        }
        sequence=s.sequence;
        // 失败时保留最后一张已定位图，恢复后直接拟合完整位移，不能默默漏掉失败帧间的路程。
        // 无效期间valid仍为false，控制器停车。超过三次/280ms才重建坐标系。
        if(!result.valid&&!previous.empty()) {
            result.misses=++misses;
            // 已承诺绕行时不能把近桶地图归零后再等待近桶重新检出。
            // 只从同一可靠关键帧验证恢复；超过有限窗口仍无效，由控制器保持停车/锁停。
            if(preserveCoordinates){
                result.coordinatesHeld=true;
                // 这里早退不更新 captured/previous，参考帧会一直冻结：gap 一路涨到超过
                // horizon 之后，每帧都只能报 REFERENCE_TOO_OLD，永远无法再用当前画面
                // 验证（现场 1793190973677 就有 540 帧停在这个状态，只能锁停重启）。
                // 连续 3 帧无效后把当前帧重锚为新参考：位姿保持不变、不进步数，
                // 旧关键帧仍留在 references 里供备用恢复，只是不再拿过旧帧做逐帧比对。
                if(misses>=3&&allowReanchor){
                    previous=gray;previousMask=currentMask;captured=s.captured;sequence=s.sequence;
                    result.reason="REFERENCE_REANCHORED";
                }
                return result;
            }
            if(misses<3&&gap>0&&gap<=.28)return result;
            if(!discontinuous)++result.epoch;
            result.pose={};discontinuous=true;references.clear();
        }else if(result.valid){misses=0;result.misses=0;discontinuous=false;}
        if(parkingMode&&result.valid&&!result.estimatedOrigin){
            absoluteAnchor={};absoluteAnchor.view={gray,currentMask,s.captured,result.pose,{}};
            const auto p=result.pose;
            if(parkingViews.empty()||std::hypot(p.x-parkingViews.back().view.pose.x,p.z-parkingViews.back().view.pose.z)>=.06f||
               std::abs(std::remainder(p.angle-parkingViews.back().view.pose.angle,float(2*CV_PI)))>=.14f){
                parkingViews.push_back(absoluteAnchor);
                if(parkingViews.size()>20)parkingViews.erase(parkingViews.begin());
            }
            probeView=0;pendingAt=-1;
        }
        previous=gray;previousMask=currentMask;captured=s.captured;return result;
    }
};
