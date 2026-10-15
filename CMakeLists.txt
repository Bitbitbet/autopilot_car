cmake_minimum_required(VERSION 3.4...3.18)
project(autopilot_car)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
add_compile_options(-Wall -Wextra)

find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui calib3d videoio imgcodecs dnn)
find_package(PkgConfig)
pkg_search_module(evdev REQUIRED IMPORTED_TARGET libevdev)
pkg_search_module(udev REQUIRED IMPORTED_TARGET libudev)

add_subdirectory(third_party/nlohmann_json)

add_subdirectory(src/control)
add_subdirectory(src/camera)
add_subdirectory(src/tools)
add_subdirectory(src/detection)
add_subdirectory(src/show)
add_subdirectory(src/track)

# 相机测试
add_executable(camera_tool src/tool/camera_tool.cpp)
target_link_libraries(camera_tool PRIVATE camera ${OpenCV_LIBRARIES})

# 图像合成
add_executable(img2video src/tool/img2video.cpp)
target_link_libraries(img2video PRIVATE ${OpenCV_LIBRARIES})

# 查看轨道
add_executable(trackview src/tool/track_view.cpp)
target_link_libraries(trackview PRIVATE control track camera tools nlohmann_json PkgConfig::evdev PkgConfig::udev pthread)

# 相机矫正
add_executable(calibration src/tool/calibration.cpp)
target_link_libraries(calibration PRIVATE show tools)

# 地面坐标投影 相机矫正
add_executable(calibrate_homography src/tool/calibrate_homography.cpp)
target_link_libraries(calibrate_homography PRIVATE ${OpenCV_LIBRARIES})

# 目标检测
add_executable(detection_tool src/tool/detection_tool.cpp)
target_link_libraries(detection_tool PRIVATE camera detection)

# 图像采集
add_executable(collection src/tool/collection.cpp)
target_link_libraries(collection PRIVATE control camera tools pthread PkgConfig::evdev PkgConfig::udev ${OpenCV_LIBRARIES})

# 主程序（src/icar 仍在用已删除的 Track 增量搜索重载，暂不构建）
# add_subdirectory(src/icar)
