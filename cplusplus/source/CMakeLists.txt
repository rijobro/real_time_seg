cmake_minimum_required(VERSION 3.14)
SET(proj real_time_seg)
project(${proj})

set(CMAKE_CXX_STANDARD 14)

find_package( OpenCV REQUIRED )

add_executable(${proj} video_seg.cpp)

target_link_directories(${proj} PUBLIC "${OPENCV_LIB_DIR}")
target_link_libraries(${proj} ${OpenCV_LIBS})
target_include_directories(${proj} PUBLIC ${OpenCV_INCLUDE_DIR})
