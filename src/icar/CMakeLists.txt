file(GLOB_RECURSE ICAR_SRC_LIST CONFIGURE_DEPENDS
    "src/*.cpp"
)

add_executable(icar ${ICAR_SRC_LIST})
target_include_directories(icar PRIVATE src)
target_link_libraries(icar PRIVATE predeal uart show detection nlohmann_json pthread)