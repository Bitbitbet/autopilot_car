pkg_search_module(ppnc REQUIRED IMPORTED_TARGET ppnc)
pkg_search_module(onnx REQUIRED IMPORTED_TARGET onnx)

add_library(detection STATIC src/detection.cpp)
target_include_directories(detection PUBLIC include)
target_link_libraries(detection PUBLIC tools PkgConfig::ppnc PkgConfig::onnx)
target_link_libraries(detection PRIVATE nlohmann_json)