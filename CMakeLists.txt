cmake_minimum_required(VERSION 3.0.0)
project(RetinaFace VERSION 0.1.0)
set(CMAKE_CXX_STANDARD 11)

set(OpenCV_DIR /use/local/lib)
set(OpenCV_INCLUDE_DIRS /user/local/include/opencv4)
set(MNN_DIR /home/vagrant/opt/MNN-2.3.0/build/libMNN.so)
set(MNN_INCLUDE_DIRS /user/local/include/MNN)

find_package( OpenCV REQUIRED )

include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories( ${MNN_INCLUDE_DIRS} )

add_executable(RetinaFace main.cpp retinaface.cpp)


target_link_libraries(RetinaFace ${OpenCV_LIBS} ${MNN_DIR})

