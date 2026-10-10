pkg_search_module(serial REQUIRED IMPORTED_TARGET libserial)
add_library(control STATIC src/control.cpp)
target_include_directories(control PUBLIC include ${CAR_SERIAL_INCLUDE_DIR})
target_link_libraries(control PUBLIC PkgConfig::serial)
