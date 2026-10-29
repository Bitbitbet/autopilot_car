pkg_search_module(serial REQUIRED IMPORTED_TARGET libserial)
add_library(race_compat_control STATIC src/control.cpp)
target_include_directories(race_compat_control PUBLIC include)
target_link_libraries(race_compat_control PUBLIC PkgConfig::serial)
