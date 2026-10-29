file(GLOB_RECURSE RACING_SRC_LIST CONFIGURE_DEPENDS
    "src/*.cpp"
)

add_executable(racing ${RACING_SRC_LIST})
target_link_libraries(racing PRIVATE camera control track detection joystick)
