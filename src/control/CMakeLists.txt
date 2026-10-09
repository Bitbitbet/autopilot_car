pkg_search_module(serial REQUIRED IMPORTED_TARGET libserial)
find_path(CAR_SERIAL_INCLUDE_DIR NAMES libserial/SerialPort.h
    HINTS ${serial_INCLUDE_DIRS}
    PATHS /usr/local/include /usr/include REQUIRED)

add_library(control STATIC src/control.cpp)
target_include_directories(control PUBLIC include ${CAR_SERIAL_INCLUDE_DIR})
target_link_libraries(control PUBLIC PkgConfig::serial)
