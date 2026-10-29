add_library(detection STATIC src/detection.cpp)
target_include_directories(detection PUBLIC include)
target_link_libraries(detection PUBLIC PkgConfig::ppnc PkgConfig::onnx ${OpenCV_LIBRARIES})
target_link_libraries(detection PRIVATE nlohmann_json)