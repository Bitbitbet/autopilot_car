#include "race_ui.hpp"
#include <iostream>
static void check(bool ok,const char *why) { if(!ok) throw std::runtime_error(why); }
int main() {
    try {
        RaceConfig cfg;
        cv::Mat live(240,320,CV_8UC3,cv::Scalar(20,40,60));
        cv::Mat bin(240,320,CV_8UC1,cv::Scalar(255));
        RaceSnapshot s; s.sequence=123; s.captured=1;
        s.image=cv::Mat(240,320,CV_8UC3,cv::Scalar(90,110,130));
        auto image=raceCompactView(live,bin,s,1.1,cfg,false);
        check(image.size()==cv::Size(640,300),"compact display size");
        check(image.at<cv::Vec3b>(100,100)==cv::Vec3b(20,40,60),"live lane missing");
        check(image.at<cv::Vec3b>(100,420)==cv::Vec3b(90,110,130),"AI image must use its matched old frame");
        image=raceCompactView(live,bin,s,2,cfg,true);
        check(image.at<cv::Vec3b>(100,420)==cv::Vec3b(255,255,255),"binary toggle missing");
        check(s.image.at<cv::Vec3b>(100,100)==cv::Vec3b(90,110,130),"preview mutated AI image");
        s.image.release(); image=raceCompactView(live,bin,s,2,cfg,false);
        check(image.at<cv::Vec3b>(100,420)==cv::Vec3b(0,0,0),"missing AI cannot be replaced with current frame");
        check(raceUiKey(-1)==-1&&!raceUiQuit(-1),"no key must not exit");
        check(raceUiQuit(raceUiKey(27))&&raceUiQuit(raceUiKey(0x10001b)),"Esc modifiers must be accepted");
        check(raceUiQuit(raceUiKey(0x01000000)),"Qt Escape must not become zero");
        check(raceUiQuit(raceUiKey(0xff1b)),"X11 Escape must be accepted");
        check(raceUiQuit('q')&&raceUiQuit('Q')&&!raceUiQuit('a'),"quit key mapping");
        check(raceUiKey(0x100020)==' ',"emergency space mapping");
        std::cout<<"PASS compact matched frames, view toggle, immutable images, Escape/Q and emergency key mapping\n";
    } catch(const std::exception &e) { std::cerr<<"FAIL "<<e.what()<<'\n'; return 1; }
}
