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

add_executable(race_boundary_audit race_boundary_audit.cpp)
target_include_directories(race_boundary_audit PRIVATE ../src/tool)
target_link_libraries(race_boundary_audit PRIVATE track nlohmann_json pthread)

add_executable(race_crop_regression race_crop_regression.cpp)
target_include_directories(race_crop_regression PRIVATE ../src/tool)
target_link_libraries(race_crop_regression PRIVATE track nlohmann_json pthread)

add_executable(race_ui_regression race_ui_regression.cpp)
target_include_directories(race_ui_regression PRIVATE ../src/tool)
target_link_libraries(race_ui_regression PRIVATE tools nlohmann_json pthread)
add_test(NAME race_ui_regression COMMAND race_ui_regression)

add_executable(race_appearance_regression race_appearance_regression.cpp)
target_include_directories(race_appearance_regression PRIVATE ../src/tool)
target_link_libraries(race_appearance_regression PRIVATE track nlohmann_json pthread)

add_executable(race_trial_regression race_trial_regression.cpp ../src/control/src/control.cpp)
target_include_directories(race_trial_regression BEFORE PRIVATE mock ../src/control/include ../src/tool)
target_link_libraries(race_trial_regression PRIVATE pthread)
add_test(NAME race_trial_regression COMMAND race_trial_regression)

add_executable(stop_candidate_review_regression stop_candidate_review_regression.cpp)
target_include_directories(stop_candidate_review_regression PRIVATE ../src/tool)
target_link_libraries(stop_candidate_review_regression PRIVATE tools nlohmann_json)
add_test(NAME stop_candidate_review_regression COMMAND stop_candidate_review_regression)

add_executable(race_stop_regression race_stop_regression.cpp)
target_include_directories(race_stop_regression PRIVATE ../src/tool)
target_link_libraries(race_stop_regression PRIVATE track tools nlohmann_json pthread)

add_executable(race_backend_contract_regression race_backend_contract_regression.cpp ../src/tool/race_backend_loader.cpp)
target_include_directories(race_backend_contract_regression PRIVATE ../src/tool)
target_link_libraries(race_backend_contract_regression PRIVATE tools nlohmann_json pthread dl)
add_library(race_mock_backend_legacy SHARED race_mock_backend.cpp)
target_compile_definitions(race_mock_backend_legacy PRIVATE LEGACY_ONLY=1)
add_library(race_mock_backend_wrong SHARED race_mock_backend.cpp)
target_compile_definitions(race_mock_backend_wrong PRIVATE PROBE_ABI=1)
add_library(race_mock_backend_good SHARED race_mock_backend.cpp)
target_compile_definitions(race_mock_backend_good PRIVATE PROBE_ABI=2)
