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
        cv::Mat frame, frame_ori, result;
        cap >> frame;
        if (frame.empty()) {
            std::cerr << "Error: Could not read frame." << std::endl;
            break;
        }

        frame_ori = frame.clone();
        cv::imshow("Camera_", frame);

        cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);
        cv::threshold(frame, frame, 180, 255, cv::THRESH_BINARY);

        std::vector<std::vector<cv::Point>> contours;
        std::vector<cv::Vec4i> hierarchy;
        cv::findContours(frame, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        for(size_t i = 0; i < contours.size(); i++){
            if(hierarchy[i][3] == -1){
                cv::drawContours(frame, contours, int(i), cv::Scalar(255,0,0), cv::FILLED);
            }
        }

        cv::bitwise_and(frame_ori,frame_ori,result,frame);


        cv::imshow("Camera", result);
        if (cv::waitKey(30) == 27) break;
    }
    
}