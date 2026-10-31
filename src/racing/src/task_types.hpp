#pragma once
#include "predict_result.hpp"
#include "race_labels.hpp"
#include <opencv2/opencv.hpp>
using namespace cv;
inline constexpr int IMAGE_COL=640,IMAGE_ROW=480,CAMERA_COL=640,CAMERA_ROW=480;
// Task edge interchange: x is row, y is column, both native 640x480 pixels.
struct PointX {int x=0,y=0;float slope=0;PointX()=default;PointX(int row,int col):x(row),y(col){};PointX(int row,int col,float s):x(row),y(col),slope(s){};};
inline constexpr int racePx(int oldPixel){return oldPixel*2;}
inline constexpr float racePx(float oldPixel){return oldPixel*2;}
inline constexpr double racePx(double oldPixel){return oldPixel*2;}
inline constexpr int raceArea(int oldArea){return oldArea*4;}
