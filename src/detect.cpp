#include <iostream>
#include <string>
#include <thread>
#include <algorithm>
#include <atomic>
#include <chrono>

#include <opencv2/opencv.hpp>
#include <tesseract/baseapi.h>
#include <mqtt/client.h>

const std::string SERVER_ADDRESS("tcp://localhost:1883");
const std::string CLIENT_ID("LicensePlateDetectorClient");
const std::string TOPIC_REQ("license/plate");
const std::string TOPIC_RES("license/plate/response");

class detect : public mqtt::callback{
    public:
    detect(mqtt::async_client& cli) : client(cli), running(false){
            //Buka kamera
            cap.open(PATH_CAM, cv::CAP_V4L2);
            if (!cap.isOpened()) {
                std::cerr << "Error: Could not open camera.\n";
                return ;
            }

            //Inisialisasi Tesseract OCR
            if (ocr.Init(nullptr, "eng")) {
                std::cerr << "Error: Could not initialize tesseract.\n";
                return ;
            }
            ocr.SetVariable("tessedit_char_whitelist", "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789");
    }

    ~detect(){
        ocr.End();
        cap.release();
        cv::destroyAllWindows();
    }

    private:
    cv::VideoCapture cap;
    tesseract::TessBaseAPI ocr;
    mqtt::async_client& client;
    std::atomic<bool> running;
    std::thread worker;

    const std::string PATH_CAM = "/dev/v4l/by-id/usb-FLH-H62-200508_Integrated_Camera-video-index0";

    void start();
    void stop();
    void single_capture();
    void image_process(const cv::Mat& frame);

    void message_arrived(mqtt::const_message_ptr msg) override;
    void connection_lost(const std::string& cause) override;

};

void detect::connection_lost(const std::string& cause) {
    std::cerr << "[MQTT] Connection lost: " << cause << std::endl;
}

void detect::message_arrived(mqtt::const_message_ptr msg) {
    std::string payload = msg->to_string();
    std::cout << "[MQTT] Received: " << payload << std::endl;

    if (payload == "capture") {
        single_capture();
    } 
    else if (payload == "start") {
        start();
    } 
    else if (payload == "stop") {
        stop();
    }
}

void detect::start(){
    if(running.load()) return;
    running.store(true);

    worker = std::thread([this](){
        while (running.load())
        {
            cv::Mat frame;

            if(!cap.isOpened()){
                std::this_thread::sleep_for(std::chrono::milliseconds(200));
                continue;
            }
            cap >> frame;
            if (frame.empty()) continue;
            image_process(frame);

            std::this_thread::sleep_for(std::chrono::milliseconds(100));

        }
        
    });
}

void detect::stop(){
    if(!running.load()) return;
    running.store(false);
    if(worker.joinable()) worker.join();
}

void detect::single_capture(){
    if (!cap.isOpened()) return;
    cv::Mat frame;
    cap >> frame;
    if (!frame.empty()) image_process(frame);
}

void detect::image_process(const cv::Mat& frame){
        cv::Mat gray, blurImg, edges;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        cv::GaussianBlur(gray, blurImg, cv::Size(5,5), 0);
        cv::Canny(blurImg, edges, 100, 200);

        // Temukan kontur
        std::vector<std::vector<cv::Point>> contours;
        std::vector<cv::Vec4i> hierarchy;
        cv::findContours(edges, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        cv::Mat output = frame.clone();

        for (size_t i = 0; i < contours.size(); i++) {
            cv::Rect rect = cv::boundingRect(contours[i]);
            float aspectRatio = (float)rect.width / rect.height;
            int area = rect.area();

            // Filter kontur yang mungkin plat
            if (area > 2000 && area < 20000 && aspectRatio > 2.0 && aspectRatio < 6.5) {
                cv::rectangle(output, rect, cv::Scalar(0,255,0), 2);

                // === Preprocessing ROI plat ===
                cv::Mat roi = frame(rect);
                cv::Mat roiGray, roiThresh;

                cv::cvtColor(roi, roiGray, cv::COLOR_BGR2GRAY);
                cv::resize(roiGray, roiGray, cv::Size(), 2.0, 2.0);  // perbesar biar OCR lebih mudah
                cv::GaussianBlur(roiGray, roiGray, cv::Size(3,3), 0);
                cv::adaptiveThreshold(roiGray, roiThresh, 255, 
                                      cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, 31, 15);
                cv::morphologyEx(roiThresh, roiThresh, cv::MORPH_CLOSE, 
                                 cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3,3)));

                // === OCR Tesseract ===
                ocr.SetImage(roiThresh.data, roiThresh.cols, roiThresh.rows, 1, roiThresh.step);
                std::string text = std::string(ocr.GetUTF8Text());

                // Hapus spasi berlebih dari OCR
                text.erase(std::remove_if(text.begin(), text.end(), ::isspace), text.end());

                if (!text.empty()) {
                    std::cout << "Detected Plate: " << text << std::endl;
                    cv::putText(output, text, cv::Point(rect.x, rect.y - 5),
                                cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0,0,255), 2);

                    client.publish(TOPIC_RES, text,1,false); // publish
                    
                }

                cv::imshow("Plate ROI", roiThresh);
            }
        }

        cv::imshow("Camera", output);
        if (cv::waitKey(10) == 27) stop(); // tekan ESC untuk keluar

}

int main() {
    mqtt::async_client client(SERVER_ADDRESS, CLIENT_ID);
    detect detector(client);
    client.set_callback(detector);

    mqtt::connect_options connOpts;
    connOpts.set_clean_session(true);

    try{
        client.connect(connOpts)->wait();
        client.subscribe(TOPIC_REQ, 1)->wait();

        std::cout << "service active at " << TOPIC_REQ << std::endl; 
        std::cout << "waiting for request..." << std::endl;

        while (true) std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    catch(const mqtt::exception& e){
        std::cout << "Error: " << e.what() << std::endl;
    }

    return 0;
}
