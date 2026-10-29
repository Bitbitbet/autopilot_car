add_library(joystick INTERFACE)
target_include_directories(joystick INTERFACE include)
target_link_libraries(joystick INTERFACE control PkgConfig::evdev PkgConfig::udev pthread)
