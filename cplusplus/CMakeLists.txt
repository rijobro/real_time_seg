cmake_minimum_required(VERSION 3.14)
SET(proj real_time_seg)
project(${proj})

set(CMAKE_CXX_STANDARD 14)

add_executable(${proj} video_seg.cpp)

# pytorch
find_package(Torch REQUIRED)
target_link_libraries(${proj} "${TORCH_LIBRARIES}")
message(STATUS "TORCH_LIBRARIES=${TORCH_LIBRARIES}")

# opencv
find_package(OpenCV REQUIRED)
#target_link_libraries(${proj} opencv_videoio)
target_link_libraries(${proj} ${OpenCV_LIBS})
message(STATUS "OpenCV_LIBS=${OpenCV_LIBS}")

