cmake_minimum_required(VERSION 3.4...3.18)
project(autopilot_car)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)


find_package(OpenCV REQUIRED)
# message(STATUS "${OpenCV_LIBRARIES}")
find_package(PkgConfig)

add_subdirectory(third_party/nlohmann_json)

add_subdirectory(src/uart)
add_subdirectory(src/predeal)
add_subdirectory(src/tools)
add_subdirectory(src/detection)
add_subdirectory(src/show)

# 相机测试
add_executable(camera src/tool/camera.cpp)
target_link_libraries(camera PRIVATE predeal ${OpenCV_LIBRARIES})

# 图像合成
add_executable(img2video src/tool/img2video.cpp)
target_link_libraries(img2video PRIVATE ${OpenCV_LIBRARIES})

# 相机矫正
add_executable(calibration src/tool/calibration.cpp)
target_link_libraries(calibration PRIVATE show tools)

# 图像测试
add_executable(imging src/tool/imging.cpp)
target_link_libraries(imging PRIVATE predeal tools)

# 目标检测
add_executable(detection_tool src/tool/detection_tool.cpp)
target_link_libraries(detection_tool PRIVATE predeal detection)

# 图像采集
add_executable(collection src/tool/collection.cpp)
target_link_libraries(collection PRIVATE uart predeal tools pthread ${OpenCV_LIBRARIES})

# 主程序
add_subdirectory(src/icar)