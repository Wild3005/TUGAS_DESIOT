cmake_minimum_required(VERSION 3.10)
project(detect)

find_package(OpenCV REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(TESSERACT REQUIRED tesseract)

include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${TESSERACT_INCLUDE_DIRS}
)

add_executable(detect src/detect.cpp)
target_link_libraries(detect
    ${OpenCV_LIBS}
    ${TESSERACT_LIBRARIES}
    tesseract
    lept
)
