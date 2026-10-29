file(GLOB_RECURSE RACING_SRC_LIST CONFIGURE_DEPENDS
    "src/*.cpp"
)

add_executable(racing ${RACING_SRC_LIST})
target_include_directories(racing PRIVATE src ../tool)
target_link_libraries(racing PRIVATE camera control track tools detection PkgConfig::evdev PkgConfig::udev pthread)
