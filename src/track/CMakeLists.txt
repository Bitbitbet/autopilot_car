add_library(track STATIC src/track.cpp)
target_include_directories(track PUBLIC include)
target_link_libraries(track PUBLIC tools)
