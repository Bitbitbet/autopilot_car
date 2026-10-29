#pragma once
#include <opencv2/imgproc.hpp>

// All road consumers receive the same convention: drivable surface is white.
// RGB images stay unchanged for AI, signs and crosswalk appearance checks.
inline cv::Mat roadBinary(const cv::Mat &image, int thresholdValue, bool blackRoad, bool repairInterior=true) {
    cv::Mat gray, binary;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    const int polarity = blackRoad ? cv::THRESH_BINARY_INV : cv::THRESH_BINARY;
    cv::threshold(gray, binary, thresholdValue < 0 ? 0 : thresholdValue, 255,
                  polarity | (thresholdValue < 0 ? cv::THRESH_OTSU : 0));
    if(blackRoad && repairInterior) {
        // Specular highlights and white paint become dark holes after inversion.
        // Only repair bounded interior components. The outside, body connected to
        // the lower border and fork dividers connected to the far border survive.
        cv::Mat exterior;
        cv::copyMakeBorder(binary,exterior,1,1,1,1,cv::BORDER_CONSTANT,cv::Scalar(0));
        cv::floodFill(exterior,{0,0},cv::Scalar(255));
        cv::Mat holes;cv::bitwise_not(exterior(cv::Rect(1,1,binary.cols,binary.rows)),holes);
        cv::Mat labels,stats,centers;
        const int count=cv::connectedComponentsWithStats(holes,labels,stats,centers,8,CV_32S);
        for(int component=1;component<count;++component) {
            if(stats.at<int>(component,cv::CC_STAT_AREA)>int(binary.total()/10) ||
               stats.at<int>(component,cv::CC_STAT_WIDTH)>binary.cols*45/100 ||
               stats.at<int>(component,cv::CC_STAT_HEIGHT)>binary.rows*60/100)continue;
            const cv::Rect box(stats.at<int>(component,cv::CC_STAT_LEFT),stats.at<int>(component,cv::CC_STAT_TOP),
                               stats.at<int>(component,cv::CC_STAT_WIDTH),stats.at<int>(component,cv::CC_STAT_HEIGHT));
            binary(box).setTo(255,labels(box)==component);
        }
    }
    return binary;
}
