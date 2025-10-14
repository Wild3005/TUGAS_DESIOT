#include <iostream>

#include "opencv4/opencv2/opencv.hpp"

int main()
{
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open camera." << std::endl;
        return -1;
    }

    cv::namedWindow("Camera", cv::WINDOW_AUTOSIZE);

    while (true)
    {
        cv::Mat frame;
        cap >> frame;
        if (frame.empty()) {
            std::cerr << "Error: Could not read frame." << std::endl;
            break;
        }

        cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);
        cv::imshow("Camera", frame);
        if (cv::waitKey(30) == 27) break;
    }
    
}