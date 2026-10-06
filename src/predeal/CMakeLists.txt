add_library(predeal STATIC src/predeal.cpp)

target_include_directories(predeal PUBLIC include)
target_link_libraries(predeal PRIVATE ${OpenCV_LIBRARIES})