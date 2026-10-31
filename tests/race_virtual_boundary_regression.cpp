#include "race_virtual_boundary.hpp"
#include <iostream>
#include <filesystem>
#include <opencv2/imgcodecs.hpp>
static void require(bool ok,const char *why){if(!ok)throw std::runtime_error(why);}
static PredictResult cone(int x=95,int y=100,int width=25,int height=60,int type=LABEL_CONE) {return {type,"",.95f,x,y,width,height};}
struct Fixture {
    RaceConfig race;RaceNavigationConfig nav;RaceVirtualBoundary planner;
    cv::Mat binary,image;Track track;RaceSnapshot snap;double time=0;uint64_t seq=0;
    Fixture():planner(nav){}
    RaceObstaclePreviewPlan step(std::vector<PredictResult> ds={},bool automatic=true,float speed=.1f,double dt=.07,bool fresh=true) {
        time+=dt;
        if(fresh) {
            binary=cv::Mat::zeros(240,320,CV_8UC1);cv::rectangle(binary,{40,0,240,240},{255},-1);
            for(auto d:ds)cv::rectangle(binary,{d.x,d.y,d.width,d.height},{0},-1);
            track.handle(binary);cv::cvtColor(binary,image,cv::COLOR_GRAY2BGR);
            snap={++seq,time,std::move(ds),false,{},image};
        }
        return planner.update(snap,time,binary,track,race,automatic,speed);
    }
    RaceObstaclePreviewPlan repeat(std::vector<PredictResult> ds,int n=3,bool automatic=true,float speed=.1f) {
        RaceObstaclePreviewPlan p;while(n--)p=step(ds,automatic,speed);return p;
    }
};
static void verify(const RaceObstaclePreviewPlan &p) {
    require(p.virtualBoundary&&!p.metric&&!p.passed&&!p.motionValid,"Virtual path claims metric motion or rear clearance");
    require(p.candidate&&!p.stop&&p.centers.size()>=35,"Expected usable virtual corridor");
    for(size_t i=0;i<p.centers.size();++i) {
        const auto c=p.centers[i];const float half=(279-40)*(.18f/.45f*.5f+.005f/.45f),pad=6;
        require(c.y-half>=40&&c.y+half<=279,"Projected vehicle leaves observed road");
        require(p.left[i].y>=40&&p.right[i].y<=279&&p.left[i].y<p.right[i].y,"Virtual boundary leaves road");
        for(auto b:p.obstacles)if(c.x>=b.y-pad&&c.x<=b.br().y+pad)
            require(c.y-half>=b.br().x+pad||c.y+half<=b.x-pad,"Projected envelope intersects cone");
    }
}
static void recovery(const cv::Mat &image,cv::Rect box) {
    RaceVirtualBoundary planner;RaceConfig race;AutoPilot pilot;Track track;
    cv::Mat gray,bin;cv::cvtColor(image,gray,cv::COLOR_BGR2GRAY);cv::threshold(gray,bin,150,255,cv::THRESH_BINARY);
    track.handle(bin);RaceSnapshot snap;RaceObstaclePreviewPlan p;double now=0;
    auto step=[&](bool detected,const cv::Mat &current,double dt=.07) {
        now+=dt;snap={++snap.sequence,now,{},false,{},current};
        if(detected)snap.detections.push_back(cone(box.x,box.y,box.width,box.height));
        cv::cvtColor(current,gray,cv::COLOR_BGR2GRAY);cv::threshold(gray,bin,150,255,cv::THRESH_BINARY);track.handle(bin);
        p=planner.update(snap,now,bin,track,race,true,.1f);
        std::vector<PointX> l,r;
        if(p.candidate&&!p.stop) {
            for(auto v:p.left)l.emplace_back(v.x,int(std::lround(v.y)));
            for(auto v:p.right)r.emplace_back(v.x,int(std::lround(v.y)));
        }
        pilot.update(l,r,float(dt));
    };
    for(int n=0;n<7;++n)step(true,image);
    require(p.candidate&&pilot.ready,"Recovery fixture does not produce a control-ready path");
    for(int cycle=0;cycle<3;++cycle) {
        // Captured run has 0.70s model gaps. Each intermediate image must
        // still match; this is not a blind replay of the last green polyline.
        for(int n=0;n<10;++n) {
            step(false,image);require(p.candidate&&!p.stop&&pilot.ready,"Intermittent inference clears route or resets v6 recovery");
            require(p.reason=="MATCHED_IMAGE_CONE_RECOVERY"&&!p.passed,"Recovery lacks provenance or claims rear pass");
        }
        step(true,image);require(p.candidate,"Reacquisition unnecessarily reconfirms same cone");
    }
    for(int n=0;n<16;++n)step(false,image);
    require(p.stop&&!p.candidate,"Template recovery runs beyond bounded AI horizon");
    planner.reset();for(int n=0;n<7;++n)step(true,image);
    cv::Mat absent=image.clone();cv::rectangle(absent,box,{225,225,225},-1);step(false,absent);
    require(p.stop&&!p.candidate,"Absent cone silently uses cached image or path");
    planner.reset();step(true,image);require(!p.candidate,"Explicit reset keeps stale confirmations");
}
static void turnField(const std::filesystem::path &run) {
    const auto sample=run/"bypass-stop-evidence/ai-sample-2";
    const auto image=cv::imread((sample/"camera.png").string());require(!image.empty(),"Turning field image missing");
    std::ifstream input(sample/"detections.json");nlohmann::json record;input>>record;
    std::vector<PredictResult> detections;
    for(const auto &d:record.at("detections"))detections.push_back({d.at("type"),"",d.at("score"),d.at("x"),d.at("y"),d.at("width"),d.at("height")});
    cv::Mat gray,bin;cv::cvtColor(image,gray,cv::COLOR_BGR2GRAY);cv::threshold(gray,bin,0,255,cv::THRESH_BINARY|cv::THRESH_OTSU);
    Track track;track.handle(bin);RaceConfig race;RaceVirtualBoundary planner;RaceObstaclePreviewPlan p;
    AutoPilot lane,virtualPilot;
    for(int n=1;n<=10;++n) {
        RaceSnapshot s{uint64_t(n),n*.07,detections,false,{},image};p=planner.update(s,s.captured,bin,track,race,true,.1f);
        if(n>=3)require(p.candidate&&!p.stop,("Captured clear gap rejected: "+p.reason).c_str());
        std::vector<PointX> l,r;for(auto v:p.left)l.emplace_back(v.x,int(std::lround(v.y)));for(auto v:p.right)r.emplace_back(v.x,int(std::lround(v.y)));
        virtualPilot.update(l,r,.07f,.1f);lane.update(track.pointsEdgeLeft,track.pointsEdgeRight,.07f,.1f);
    }
    require(virtualPilot.ready,"Turning field path cannot reach control-ready state");
    std::cout<<"Captured turning image: lane PWM="<<lane.servo<<", virtual PWM="<<virtualPilot.servo<<", center="<<virtualPilot.center<<'\n';
    require(virtualPilot.servo>lane.servo+35,"Bypass steering is still diluted into nearly straight lane steering");
    for(size_t i=0;i<p.centers.size();++i) {
        const float width=p.visibleRight[i].y-p.visibleLeft[i].y,half=width*(.18f/.45f*.5f+.005f/.45f),pad=std::max(6.f,width*.025f);
        const auto c=p.centers[i];
        for(auto b:p.obstacles)if(c.x>=b.y-pad&&c.x<=b.br().y+pad)
            require(c.y-half>=b.br().x+pad-.001f||c.y+half<=b.x-pad+.001f,"Turning field path really intersects cone");
    }
    std::cout<<"PASS captured turning stop: clear corridor accepted, earlier right command and cone clearance\n";
}
static void recentFields(const std::filesystem::path &root) {
    for(const auto &relative:std::vector<std::string>{
        "1793355522566-1178/bypass-stop-evidence/ai-sample-1",
        "1793356290816-8412/bypass-stop-evidence/ai-sample-1",
        "1793356290816-8412/bypass-stop-evidence/ai-sample-2",
        "1793356290816-8412/bypass-stop-evidence/ai-sample-3",
        "1793356290816-8412/bypass-stop-evidence/ai-sample-5",
        "1793355415674-32378/bypass-stop-evidence/ai-sample-1"}) {
        const auto sample=root/relative;const auto image=cv::imread((sample/"camera.png").string());
        require(!image.empty(),"Recent sample missing");nlohmann::json record;std::ifstream input(sample/"detections.json");input>>record;
        std::vector<PredictResult> ds;for(const auto &d:record.at("detections"))ds.push_back({d.at("type"),"",d.at("score"),d.at("x"),d.at("y"),d.at("width"),d.at("height")});
        Fixture f;if(ds.empty()){f.repeat({cone()});f.repeat({cone(95,125)});f.repeat({cone(95,150)});}
        cv::Mat gray,bin;cv::cvtColor(image,gray,cv::COLOR_BGR2GRAY);cv::threshold(gray,bin,0,255,cv::THRESH_BINARY|cv::THRESH_OTSU);Track track;track.handle(bin);
        AutoPilot pilot;bool returned=false;
        for(int n=0;n<(ds.empty()?90:12);++n) {
            f.time+=.07;RaceSnapshot snap{++f.seq,f.time,ds,false,{},image};auto p=f.planner.update(snap,f.time,bin,track,f.race,true,.1f);
            if(p.virtualVisualReturn){returned=true;break;}
            if(n<3&&!ds.empty())continue;
            require(p.candidate&&!p.stop,(relative+": "+p.reason).c_str());
            std::vector<PointX> l,r;
            for(size_t i=0;i<p.centers.size();++i) {
                if(i)require(std::abs(p.centers[i].y-p.centers[i-1].y)<=.8f*(p.centers[i-1].x-p.centers[i].x)+.501f,"Repaired path still jumps");
                l.emplace_back(p.left[i].x,int(std::lround(p.left[i].y)));r.emplace_back(p.right[i].x,int(std::lround(p.right[i].y)));
            }
            pilot.update(l,r,.07f,.1f);
        }
        require(ds.empty()?returned:pilot.ready,"Recent path never returns or never becomes controllable");
        std::cout<<"PASS recent field "<<relative<<'\n';
    }
}
static void clippedBend(const std::filesystem::path &run) {
    RaceVirtualBoundary planner;RaceConfig race;Track track;RaceSnapshot snap;RaceObstaclePreviewPlan p;
    auto feed=[&](const cv::Mat &image,std::vector<PredictResult> ds,bool automatic=true) {
        cv::Mat gray,bin;cv::cvtColor(image,gray,cv::COLOR_BGR2GRAY);
        cv::threshold(gray,bin,0,255,cv::THRESH_BINARY|cv::THRESH_OTSU);track.handle(bin);
        snap={snap.sequence+1,snap.captured+.07,std::move(ds),false,{},image};
        return planner.update(snap,snap.captured,bin,track,race,automatic,.1f);
    };
    const auto before=cv::imread((run/"bypass-stop-evidence/ai-sample-1/camera.png").string());
    const auto bend=cv::imread((run/"bypass-stop-evidence/ai-sample-2/camera.png").string());
    require(!before.empty()&&!bend.empty(),"Latest bend fixtures missing");
    // Controlled continuity test using real images, not a replay of measured
    // travel between these two samples (the samples were captured further apart).
    for(int n=0;n<3;++n)feed(before,{});
    for(int n=0;n<7;++n)p=feed(bend,{cone(274,186,45,54)});
    require(p.candidate&&!p.stop&&p.direction==-1,("Bend with recent width still blocked: "+p.reason).c_str());
    require(p.reason=="CURRENT_EDGES_RECENT_WIDTH","Bad bend fit is learned instead of using width reference");
    for(int n=0;n<30;++n)p=feed(bend,{cone(274,186,45,54)});
    require(p.candidate&&!p.stop&&p.virtualWidthUncertain&&p.virtualLineCrossing,"Expired width still stops instead of current-road fallback");
    planner.reset();p=feed(bend,{cone(274,186,45,54)});
    require(p.stop,"Reset retains width from previous scene");
    std::cout<<"PASS bend image: expired width switches to current-road guidance; reset clears reference\n";
}
static void stoppedRecovery(const std::filesystem::path &run) {
    const auto image=cv::imread((run/"bypass-stop-evidence/ai-sample-2/camera.png").string());
    require(!image.empty(),"Latest width-expiry stop image missing");
    Fixture f;f.repeat({cone(200,150,25,60)},5);AutoPilot pilot;RaceObstaclePreviewPlan p;
    auto feed=[&](const cv::Mat &current,std::vector<PredictResult> detections={}) {
        cv::Mat gray,bin;cv::cvtColor(current,gray,cv::COLOR_BGR2GRAY);cv::threshold(gray,bin,0,255,cv::THRESH_BINARY|cv::THRESH_OTSU);
        Track track;track.handle(bin);f.time+=.07;RaceSnapshot snap{++f.seq,f.time,detections,false,{},current};
        p=f.planner.update(snap,f.time,bin,track,f.race,true,.1f);
        std::vector<PointX> l,r;for(auto v:p.left)l.emplace_back(v.x,int(std::lround(v.y)));for(auto v:p.right)r.emplace_back(v.x,int(std::lround(v.y)));
        pilot.update(l,r,.07f,.1f);
    };
    for(int n=0;n<80;++n) {
        feed(image);require(p.candidate&&!p.stop&&p.state=="VIRTUAL_REACQUIRE","Current visible road recovery stops after width timeout");
        if(n>8)require(pilot.ready&&pilot.speedFor(.1f,.07f)>0,"Recovery path cannot sustain actual pilot speed command");
    }
    const int before=pilot.center;cv::Mat mirror;cv::flip(image,mirror,1);
    for(int n=0;n<12;++n)feed(mirror);
    std::cout<<"Recovery current image centers: "<<before<<" -> mirrored "<<pilot.center<<'\n';
    require(std::abs(pilot.center-before)>15&&(pilot.center-160)*(before-160)<0,
        "Recovery replays an old path instead of following current road");
    {
        auto nextCone=f.planner;cv::Mat nextImage(240,320,CV_8UC3,cv::Scalar(0,0,0));
        cv::rectangle(nextImage,{40,0,240,240},{230,230,230},-1);
        cv::Mat gray,bin;cv::cvtColor(nextImage,gray,cv::COLOR_BGR2GRAY);
        cv::threshold(gray,bin,0,255,cv::THRESH_BINARY|cv::THRESH_OTSU);Track track;track.handle(bin);
        auto seq=f.seq;auto time=f.time;RaceObstaclePreviewPlan q;
        for(int n=0;n<3;++n){time+=.07;RaceSnapshot s{++seq,time,{cone(95,70,25,40)},false,{},nextImage};q=nextCone.update(s,time,bin,track,f.race,true,.1f);}
        require(q.candidate&&q.direction==1,"New cone inherits old left bypass direction");
        time+=.07;RaceSnapshot s{++seq,time,{},false,{},nextImage};q=nextCone.update(s,time,bin,track,f.race,true,.1f);
        require(q.stop&&q.state=="VIRTUAL_LOST_BEFORE_NEAR","New far cone inherits previous near/pass evidence");
    }
    feed(image,{cone(100,100,25,60,LABEL_PERSON)});require(p.stop&&p.state=="VIRTUAL_PERSON_STOP","Recovery ignores a newly observed person");
    cv::Mat flat(240,320,CV_8UC3,cv::Scalar(0,0,0));cv::rectangle(flat,{40,0,240,240},{230,230,230},-1);
    bool exited=false;for(int n=0;n<25&&!exited;++n){feed(flat);exited=p.virtualVisualReturn;}
    require(exited&&!p.passed,"Restored lane cannot hand off or falsely claims measured rear clearance");
    std::cout<<"PASS latest stop image: >5s continuous nonzero recovery commands, current-road response, person priority, normal-lane handoff\n";
}
static void farHorizonStops(const std::filesystem::path &run) {
    for(int sample:{2,3}) {
        const auto folder=run/("bypass-stop-evidence/ai-sample-"+std::to_string(sample));
        const auto image=cv::imread((folder/"camera.png").string());require(!image.empty(),"Horizon stop image missing");
        nlohmann::json record;std::ifstream input(folder/"detections.json");input>>record;
        std::vector<PredictResult> ds;for(auto d:record["detections"])ds.push_back({d["type"],"",d["score"],d["x"],d["y"],d["width"],d["height"]});
        // This frame used a visually recovered box, not a model detection.
        // Replay its recorded geometry separately from inference accuracy.
        if(sample==2&&ds.empty()) {
            nlohmann::json nav;std::ifstream saved(folder/"navigation.json");saved>>nav;
            for(auto b:nav["obstacles"])ds.push_back(cone(b[0],b[1],b[2],b[3]));
        }
        Fixture f;if(ds.empty())f.repeat({cone(200,150,25,60)},5);else f.repeat(ds,5);AutoPilot pilot;
        cv::Mat gray,bin;cv::cvtColor(image,gray,cv::COLOR_BGR2GRAY);cv::threshold(gray,bin,0,255,cv::THRESH_BINARY|cv::THRESH_OTSU);
        Track track;track.handle(bin);RaceObstaclePreviewPlan p;
        for(int n=0;n<60;++n) {
            f.time+=.07;RaceSnapshot s{++f.seq,f.time,ds,false,{},image};p=f.planner.update(s,f.time,bin,track,f.race,true,.1f);
            if(n<3)continue;
            require(p.candidate&&!p.stop,("Far horizon still stops sample "+std::to_string(sample)+": "+p.reason).c_str());
            std::vector<PointX> l,r;for(auto v:p.left)l.emplace_back(v.x,int(std::lround(v.y)));for(auto v:p.right)r.emplace_back(v.x,int(std::lround(v.y)));
            pilot.update(l,r,.07f,.1f);
            if(n>10)require(pilot.ready&&pilot.speedFor(.1f,.07f)>0,"Far horizon resets pilot or zeros recovery speed");
        }
        std::cout<<"PASS real far-horizon stop sample "<<sample<<": continuous current near-road control, rows="<<p.centers.size()<<'\n';
    }
}
int main(int argc,char **argv) {
    try {
        for(int sign:{-1,1}) {
            RaceVirtualSteeringFeedback feedback;RaceObstaclePreviewPlan route;
            route.fresh=route.candidate=true;route.stop=false;route.state=sign>0?"VIRTUAL_RIGHT":"VIRTUAL_LEFT";
            uint16_t pwm=PWMSERVOMID;const uint16_t base=PWMSERVOMID+sign*90;
            for(int n=1;n<=30;++n) {
                route.sequence=n;route.captured=n*.07;
                const auto next=feedback.update(route,sign*30,base,pwm,.07f,.1f);
                require(std::abs(int(next)-int(pwm))<=43,"Feedback bypasses steering slew limit");pwm=next;
            }
            require(sign*(int(pwm)-int(base))>60&&std::abs(feedback.correction())<=120,"Persistent error does not strengthen bounded left/right steering");
            const auto held=feedback.correction();
            for(int n=0;n<20;++n)pwm=feedback.update(route,sign*30,base,pwm,.07f,.1f);
            require(feedback.correction()==held,"Duplicate AI snapshot accumulates steering correction");
            for(int n=31;n<=40;++n){route.sequence=n;route.captured=n*.07;pwm=feedback.update(route,sign*(30-(n-30)*3),base,pwm,.07f,.1f);}
            require(feedback.correction()==0,"Correction persists after target alignment");
            for(int n=41;n<=60;++n){route.sequence=n;route.captured=n*.07;pwm=feedback.update(route,sign*30,base,pwm,.07f,0);}
            require(feedback.correction()==0,"Stopped vehicle winds up steering");
            for(int n=61;n<=75;++n){route.sequence=n;route.captured=n*.07;pwm=feedback.update(route,sign*30,base,pwm,.07f,.1f);}
            route.fresh=false;feedback.update(route,sign*30,base,pwm,.07f,.1f);require(feedback.correction()==0,"Stale observation keeps correction");
            route.fresh=true;
            for(int n=76;n<=90;++n){route.sequence=n;route.captured=n*.07;pwm=feedback.update(route,sign*30,base,pwm,.07f,.1f);}
            route.state="VIRTUAL_RETURN";feedback.update(route,sign*30,base,pwm,.07f,.1f);
            require(feedback.correction()==0,"Returning to lane keeps outward steering correction");
        }
        std::cout<<"PASS feedback: persistent left/right error strengthens steering; improving error unwinds; stop/stale reset; duplicates do not accumulate; bounded slew\n";
        {
            RaceVirtualSteeringFeedback f;RaceObstaclePreviewPlan p;p.fresh=p.candidate=true;p.stop=false;
            p.direction=1;p.state="VIRTUAL_RIGHT";p.primary={90,100,25,60};p.obstacles={p.primary};p.sequence=1;p.captured=.07;
            auto pwm=f.update(p,30,1590,1590,.07f,.1f);require(pwm>1590&&f.correction()<=40,"Entry lead missing or excessive");
            p.state="VIRTUAL_HOLD_SIDE";p.obstacles.clear();++p.sequence;p.captured+=.07;
            f.update(p,30,1590,pwm,.07f,.1f);require(f.correction()==0,"Entry lead persists after cone disappears");
        }
        Fixture left;auto p=left.step({cone()},false,0);
        require(p.stop&&!p.candidate&&p.state=="VIRTUAL_CONFIRMING","Single detection authorizes bypass");
        p=left.step({},false,0,.07,false);require(!p.candidate,"Repeated snapshot counts as confirmation");
        p=left.repeat({cone()},3,false,0);verify(p);require(p.direction==1,"Left cone must pass right");
        auto original=left.binary.clone();AutoPilot rightPilot;
        for(int n=0;n<8;++n) {
            std::vector<PointX> l,r;for(auto v:p.left)l.emplace_back(v.x,int(std::lround(v.y)));for(auto v:p.right)r.emplace_back(v.x,int(std::lround(v.y)));
            rightPilot.update(l,r,.07f);
        }
        require(rightPilot.ready&&rightPilot.servo>PWMSERVOMID,"Virtual boundary cannot drive v6 right steering");
        require(cv::norm(left.binary,original)==0,"Planner modifies raw binary/track input");
        require(!left.planner.decision(p,false).authorizeCone,"Manual mode authorizes cone");
        Fixture right;auto q=right.repeat({cone(200)});verify(q);require(q.direction==-1,"Right cone must pass left");
        Fixture blocked;p=blocked.repeat({cone(140,100,40)});
        require(p.candidate&&!p.stop&&p.virtualLineCrossing,"Painted line crossing still treated as mandatory stop");
        for(const auto &c:p.centers)if(c.x>=94&&c.x<=166)
            require(c.y-(239*(.18f/.45f*.5f+.005f/.45f))>=186-.001f||
                c.y+(239*(.18f/.45f*.5f+.005f/.45f))<=134+.001f,"Line crossing relaxes cone clearance too");
        Fixture two;p=two.repeat({cone(95),cone(200)});require(p.stop&&!p.candidate,"Two blocked sides ignored");
        Fixture person;p=person.repeat({cone(),cone(220,100,25,60,LABEL_PERSON)});
        require(p.stop&&p.state=="VIRTUAL_PERSON_STOP"&&!person.planner.decision(p,true).authorizeCone,"Person priority bypassed");
        Fixture early;p=early.repeat({cone()});p=early.step({});
        require(p.stop&&p.state=="VIRTUAL_LOST_BEFORE_NEAR","Far cone disappearance becomes successful pass");
        Fixture stale;stale.repeat({cone()});p=stale.step({},true,.1f,.4,false);
        require(p.stop&&!p.fresh&&!stale.planner.decision(p,true).authorizeCone,"Stale image allows vehicle control");
        Fixture exit;exit.repeat({cone()});exit.repeat({cone(95,125)});exit.repeat({cone(95,150)});
        p=exit.step({});verify(p);require(p.state=="VIRTUAL_HOLD_SIDE","Near disappearance immediately recentres");
        for(int n=0;n<70;++n)p=exit.step({},true,0);
        require(p.state=="VIRTUAL_HOLD_SIDE","Zero-speed waiting advances visual clearance timer");
        bool returned=false;
        for(int n=0;n<80&&!returned;++n){p=exit.step({});returned=p.virtualVisualReturn;}
        require(returned&&!p.passed&&p.state=="VIRTUAL_VISUAL_EXIT","Visual return missing or labelled measured rear pass");
        Fixture interrupted;interrupted.repeat({cone()});p=interrupted.step({cone()},false,0);
        p=interrupted.step({cone()},true,0);require(p.stop&&p.state=="VIRTUAL_FAULT","Manual interruption blindly resumes old path");
        interrupted.planner.reset();p=interrupted.repeat({cone()});verify(p);
        Fixture intermittent;intermittent.step({cone()});intermittent.step({});
        p=intermittent.repeat({cone()},2);verify(p);
        cv::Mat textured(240,320,CV_8UC3,cv::Scalar(0,0,0));cv::rectangle(textured,{40,0,240,240},{225,225,225},-1);
        cv::fillConvexPoly(textured,std::vector<cv::Point>{{95,160},{107,100},{120,160}},{0,150,230});
        recovery(textured,{95,100,25,60});
        {
            RaceVirtualBoundary tracker;RaceConfig race;Track track;cv::Mat bin,gray;RaceObstaclePreviewPlan p;
            for(int n=0;n<9;++n) {
                auto current=textured.clone();
                if(n>=3) {
                    cv::rectangle(current,{95,100,25,60},{225,225,225},-1);
                    cv::fillConvexPoly(current,std::vector<cv::Point>{{99,163},{111,103},{124,163}},{0,150,230});
                }
                cv::cvtColor(current,gray,cv::COLOR_BGR2GRAY);cv::threshold(gray,bin,150,255,cv::THRESH_BINARY);track.handle(bin);
                RaceSnapshot s{uint64_t(n+1),(n+1)*.07,{},false,{},current};if(n<3)s.detections={cone()};
                p=tracker.update(s,s.captured,bin,track,race,true,.1f);
            }
            require(p.candidate&&p.primary.x>=98&&p.primary.y>=102,"Visual recovery reuses old box instead of following current cone");
        }
        if(argc>=2) {
            const auto image=cv::imread((std::filesystem::path(argv[1])/"bypass-stop-evidence/ai-sample-1/camera.png").string());
            require(!image.empty(),"Field stop image missing");recovery(image,{124,15,16,37});
            std::cout<<"PASS captured white-road stop image: 0.70s detection gaps keep current-image route and v6 ready; expired/missing visual target stops\n";
        }
        if(argc>=3)turnField(argv[2]);
        if(argc>=4)recentFields(argv[3]);
        if(argc>=5)clippedBend(argv[4]);
        if(argc>=6)stoppedRecovery(argv[5]);
        if(argc>=7)farHorizonStops(argv[6]);
        // Explicit cone authorization only removes the cone wait. All original
        // traffic priorities still run in RaceManager with the same snapshots.
        Fixture traffic;RaceManager manager(traffic.race);RacePlan plan;
        for(int n=0;n<6;++n) {
            p=traffic.step({cone()});auto d=traffic.planner.decision(p,true);
            plan=manager.update(traffic.snap,traffic.time,.07f,true,traffic.binary,traffic.track,traffic.image,d.authorizeCone);
        }
        require(!plan.stop,"Matched authorized virtual cone path still waits on cone");
        for(int n=0;n<5;++n) {
            p=traffic.step({cone(),cone(190,140,70,30,LABEL_GATE)});
            auto d=traffic.planner.decision(p,true);plan=manager.update(traffic.snap,traffic.time,.07f,true,traffic.binary,traffic.track,traffic.image,d.authorizeCone);
        }
        require(plan.stop&&std::string(plan.state)=="GATE_WAIT","Virtual bypass overrides closed gate");
        RaceManager ordinary(traffic.race);
        for(int n=0;n<6;++n){p=traffic.step({cone()});plan=ordinary.update(traffic.snap,traffic.time,.07f,true,traffic.binary,traffic.track,traffic.image);}
        require(plan.stop,"Ordinary drive cone stop changed");
        std::cout<<"PASS virtual boundary: left/right v6 steering, joint clearance, person/gate priority, stale/duplicate images, early loss, stop-time freeze, visual return and unchanged ordinary cone stop\n";
    }catch(const std::exception &e){std::cerr<<"FAIL "<<e.what()<<'\n';return 1;}
}
