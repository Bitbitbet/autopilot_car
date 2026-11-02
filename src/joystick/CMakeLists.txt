add_library(joystick STATIC src/joystick.cpp)
target_include_directories(joystick PUBLIC include)
target_link_libraries(joystick PRIVATE control PkgConfig::evdev PkgConfig::udev pthread)
