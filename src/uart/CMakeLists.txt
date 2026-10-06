pkg_search_module(serial REQUIRED IMPORTED_TARGET libserial)

add_library(uart STATIC src/uart.cpp)
target_include_directories(uart PUBLIC include)
target_link_libraries(uart PRIVATE PkgConfig::serial)