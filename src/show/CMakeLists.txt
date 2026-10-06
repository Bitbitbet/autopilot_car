add_library(show STATIC src/show.cpp)

target_link_libraries(show PRIVATE tools)

target_include_directories(show PUBLIC include)