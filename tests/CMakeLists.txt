add_executable(track_regression track_regression.cpp ../src/icar/src/track.cpp)
target_include_directories(track_regression PRIVATE ../src/icar/src)
target_link_libraries(track_regression PRIVATE tools pthread)
# OpenCV inline checked access and test assertions are required even in Release.
target_compile_options(track_regression PRIVATE -UNDEBUG -D_DEBUG)

add_executable(stop_regression stop_regression.cpp
    ../src/icar/src/center.cpp ../src/icar/src/motion.cpp
    ../src/icar/src/params.cpp ../src/icar/src/track.cpp)
target_include_directories(stop_regression PRIVATE ../src/icar/src)
target_link_libraries(stop_regression PRIVATE tools nlohmann_json)
target_compile_options(stop_regression PRIVATE -UNDEBUG -D_DEBUG)

add_executable(joystick_regression joystick_regression.cpp ../src/control/src/control.cpp)
target_include_directories(joystick_regression BEFORE PRIVATE mock ../src/control/include ../src/tool)
target_link_libraries(joystick_regression PRIVATE pthread PkgConfig::udev)
target_compile_options(joystick_regression PRIVATE -UNDEBUG)

add_executable(result_regression result_regression.cpp)
target_include_directories(result_regression PRIVATE ../src/icar/src)
target_link_libraries(result_regression PRIVATE pthread)
target_compile_options(result_regression PRIVATE -UNDEBUG)
