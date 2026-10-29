add_library(camera STATIC src/camera.cpp)

target_include_directories(camera PUBLIC include)
target_link_libraries(camera PUBLIC ${OpenCV_LIBRARIES})