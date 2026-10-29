add_library(race_compat_predeal STATIC src/predeal.cpp)

target_include_directories(race_compat_predeal PUBLIC include)
target_link_libraries(race_compat_predeal PRIVATE ${OpenCV_LIBRARIES} race_compat_tools)