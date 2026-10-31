#pragma once
#include <string>
// Shared image-coordinate contract; model tensors may be smaller than the camera.
struct PredictResult {int type;std::string label;float score;int x,y,width,height;};
