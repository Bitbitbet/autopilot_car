add_library(race_compat_tools STATIC src/tools.cpp src/mapping.cpp)

target_include_directories(race_compat_tools PUBLIC include)
target_link_libraries(race_compat_tools PUBLIC ${OpenCV_LIBRARIES})