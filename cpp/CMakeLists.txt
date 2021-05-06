set(exe_name video_seg)
add_executable(${exe_name} video_seg.cpp utils.cpp)

# pytorch
find_package(Torch REQUIRED)
target_link_libraries(${exe_name} "${TORCH_LIBRARIES}")
message(STATUS "TORCH_LIBRARIES=${TORCH_LIBRARIES}")

# opencv
find_package(OpenCV REQUIRED)
target_link_libraries(${exe_name} ${OpenCV_LIBS})
message(STATUS "OpenCV_LIBS=${OpenCV_LIBS}")

# install
install(TARGETS ${exe_name}
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
)
