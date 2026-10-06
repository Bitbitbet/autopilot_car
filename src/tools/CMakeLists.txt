add_library(tools STATIC src/tools.cpp src/mapping.cpp)

target_include_directories(tools PUBLIC include)
target_link_libraries(tools PUBLIC ${OpenCV_LIBRARIES})