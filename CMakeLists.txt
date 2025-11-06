cmake_minimum_required(VERSION 3.10)
project(detect)

set(OpenCV_DIR "/usr/local/lib/cmake/opencv4")

find_package(OpenCV REQUIRED)
find_package(PkgConfig REQUIRED)
find_package(PahoMqttCpp REQUIRED)


pkg_check_modules(TESSERACT REQUIRED tesseract)

include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${TESSERACT_INCLUDE_DIRS}
)

add_executable(detect src/detect.cpp)
# add_executable(detect src/onnx_test.cpp)
target_link_libraries(detect
    ${OpenCV_LIBS}
    ${TESSERACT_LIBRARIES}
    tesseract
    lept
    PahoMqttCpp::paho-mqttpp3
)
