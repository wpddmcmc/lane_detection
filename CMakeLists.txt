cmake_minimum_required (VERSION 2.8)

project (autocar)
# using C++11 
set(CMAKE_CXX_FLAGS "${CAMKE_CXX_FLAGS} -std=c++11 -pthread")

#if u have OpenCV version more than one, set the build path which one u want to use
set(OpenCV_DIR "/home/mingcongchen/app/opencv-3.4.0/my_build_dir/")

find_package(OpenCV REQUIRED)
find_package(OpenGL REQUIRED)

#if use realsense,uncommit this
#set(DEPENDENCIES realsense ${OPENGL_LIBRARIES})

message(STATUS "OpenCV version: ${OpenCV_VERSION}")
#message(STATUS "OpenCV include path: ${OpenCV_INCLUDE_DIRS}")
#message(STATUS "    libraries: ${OpenCV_LIBS}")

find_path(GLFW_INCLUDE_DIR NAMES GLFW/glfw3.h
        PATHS /usr/X11R6/include
              /usr/include/X11
              /opt/graphics/OpenGL/include
              /opt/graphics/OpenGL/contrib/libglfw
              /usr/local/include
              /usr/include/GL
              /usr/include
    )

find_library(GLFW_LIBRARIES NAMES glfw glfw3
            PATHS /usr/lib64
                  /usr/lib
                  /usr/lib/${CMAKE_LIBRARY_ARCHITECTURE}
                  /usr/local/lib64
                  /usr/local/lib
                  /usr/local/lib/${CMAKE_LIBRARY_ARCHITECTURE}
                  /usr/X11R6/lib
    )

aux_source_directory(./src/ DIR_SRCS)

add_executable(autocar ${DIR_SRCS})

include_directories (
    ${OpenCV_INCLUDE_DIRS}
    /usr/local/include
    /usr/include 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
#darknet path
    /home/mingcongchen/app/darknet/include
    /home/mingcongchen/app/darknet/src
#cuda path
    /usr/local/cuda-9.0/include    
)

target_link_libraries(autocar
${OpenCV_LIBS}
/usr/lib
/usr/local/lib
${DEPENDENCIES}
#darknet lib path
/home/mingcongchen/app/darknet/libdarknet.so
    )