add_executable(trackview_regression trackview_regression.cpp)
target_include_directories(trackview_regression PRIVATE ../src/tool)
target_compile_definitions(trackview_regression PRIVATE _DEBUG)
target_link_libraries(trackview_regression PRIVATE track control nlohmann_json)
add_test(NAME trackview_regression COMMAND trackview_regression)

# 使用真实控制代码和成对套接字模拟串口，不连接车辆设备。
add_executable(serial_regression serial_regression.cpp ../src/control/src/control.cpp)
target_include_directories(serial_regression BEFORE PRIVATE mock ../src/control/include)
target_link_libraries(serial_regression PRIVATE pthread)
add_test(NAME serial_regression COMMAND serial_regression)

add_executable(gui_stall_regression gui_stall_regression.cpp ../src/control/src/control.cpp)
target_include_directories(gui_stall_regression BEFORE PRIVATE mock ../src/control/include ../src/tool)
target_link_libraries(gui_stall_regression PRIVATE tools nlohmann_json pthread)
add_test(NAME gui_stall_regression COMMAND gui_stall_regression)

add_executable(race_regression race_regression.cpp)
target_include_directories(race_regression PRIVATE ../src/tool)
target_compile_definitions(race_regression PRIVATE _DEBUG)
target_link_libraries(race_regression PRIVATE track control nlohmann_json pthread)
add_test(NAME race_regression COMMAND race_regression)

add_executable(race_direction_regression race_direction_regression.cpp)
target_include_directories(race_direction_regression PRIVATE ../src/tool)
target_link_libraries(race_direction_regression PRIVATE tools nlohmann_json pthread)

add_executable(race_scene_regression race_scene_regression.cpp)
target_include_directories(race_scene_regression PRIVATE ../src/tool)
target_link_libraries(race_scene_regression PRIVATE track nlohmann_json pthread)

add_executable(race_trace_regression race_trace_regression.cpp)
target_include_directories(race_trace_regression PRIVATE ../src/tool)
target_link_libraries(race_trace_regression PRIVATE track nlohmann_json pthread)

add_executable(race_topology_regression race_topology_regression.cpp)
target_include_directories(race_topology_regression PRIVATE ../src/tool)
target_link_libraries(race_topology_regression PRIVATE track nlohmann_json pthread)
