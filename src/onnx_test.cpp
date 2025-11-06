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

            net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
            net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
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

    const std::string PATH_CAM = "/dev/v4l/by-id/usb-GENERAL_XVV-6320S_JH1706_20211203_v004-video-index0";
    const std::string MODEL_PATH = "/home/ichbinwil/OpenCV_ws/src/data_plat.onnx";

    cv::dnn::Net net = cv::dnn::readNetFromONNX(MODEL_PATH);

    std::vector<std::string> classNames = {"license_plate"};


    void start();
    void stop();
    void single_capture();
    void image_process(const cv::Mat& frame);

    void message_arrived(mqtt::const_message_ptr msg) override;
    void connection_lost(const std::string& cause) override;

    void drawPred(int classId, float conf, int left, int top, int right, int bottom,
              cv::Mat& frame, const std::vector<std::string>& classNames);

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
            // buang buffer lama
            for (int i = 0; i < 5; ++i)
                cap.grab();

            if(!cap.isOpened()){
                std::this_thread::sleep_for(std::chrono::milliseconds(200));
                continue;
            }

            cv::Mat frame;
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

    cv::destroyAllWindows();
}

void detect::single_capture(){
    if (!cap.isOpened()) return;
    cv::Mat frame;
    cap >> frame;
    if (!frame.empty()) image_process(frame);
}

void detect::image_process(const cv::Mat& frame){
        if (frame.empty()) return;

        cv::Mat frame_copy = frame;
        int inputWidth = 480, inputHeight = 480;
        float confThreshold = 0.4, nmsThreshold = 0.45;

        // === Preprocess for YOLO ===
        cv::Mat blob;
        cv::dnn::blobFromImage(frame_copy, blob, 1/255.0, cv::Size(inputWidth, inputHeight), cv::Scalar(), true, false);
        net.setInput(blob);
        cv::Mat output = net.forward();

        const int numDetections = output.size[1];
        const int dimensions = output.size[2];
        const int numClasses = dimensions - 5;
        const float* data = (float*)output.data;

        std::vector<int> classIds;
        std::vector<float> confidences;
        std::vector<cv::Rect> boxes;

        for (int i = 0; i < numDetections; i++)
        {
            float box_conf = data[i * dimensions + 4];
            if (box_conf < confThreshold) continue;

            cv::Mat scores(1, numClasses, CV_32FC1, (void*)(data + i * dimensions + 5));
            cv::Point classIdPoint;
            double maxClassScore;
            minMaxLoc(scores, 0, &maxClassScore, 0, &classIdPoint);
            float confidence = box_conf * (float)maxClassScore;

            if (confidence > confThreshold)
            {
                float cx = data[i * dimensions + 0];
                float cy = data[i * dimensions + 1];
                float w  = data[i * dimensions + 2];
                float h  = data[i * dimensions + 3];

                int left = int((cx - w / 2) * frame_copy.cols / inputWidth);
                int top  = int((cy - h / 2) * frame_copy.rows / inputHeight);
                int width  = int(w * frame_copy.cols / inputWidth);
                int height = int(h * frame_copy.rows / inputHeight);

                boxes.push_back(cv::Rect(left, top, width, height));
                confidences.push_back(confidence);
                classIds.push_back(classIdPoint.x);
            }
        }

        // === Non-Max Suppression ===
        std::vector<int> indices;
        cv::dnn::NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);

        for (int idx : indices)
        {
            cv::Rect box = boxes[idx];
            drawPred(classIds[idx], confidences[idx],
                     box.x, box.y, box.x + box.width, box.y + box.height, frame_copy, classNames);

            // Crop ROI untuk OCR
            cv::Rect validBox = box & cv::Rect(0, 0, frame_copy.cols, frame_copy.rows);
            cv::Mat roi = frame_copy(validBox).clone();

            if (roi.empty() || roi.cols < 50 || roi.rows < 20) continue; // hindari ROI kecil

            // === Preprocess untuk OCR ===
            cv::Mat gray, thresh;
            cv::cvtColor(roi, gray, cv::COLOR_BGR2GRAY);
            cv::resize(gray, gray, cv::Size(), 2.0, 2.0);
            cv::GaussianBlur(gray, gray, cv::Size(3,3), 0);
            cv::adaptiveThreshold(gray, thresh, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, 31, 15);

            // === Jalankan OCR ===
            ocr.SetImage(thresh.data, thresh.cols, thresh.rows, 1, thresh.step);
            std::string text = std::string(ocr.GetUTF8Text());

            // Bersihkan hasil OCR
            text.erase(remove_if(text.begin(), text.end(),
                                 [](unsigned char c){ return !isalnum(c); }), text.end());

            if (!text.empty()) {
                std::cout << "Plat Terdeteksi: " << text << std::endl;
                putText(frame_copy, text, cv::Point(validBox.x, validBox.y - 5),
                        cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0,0,255), 2);
            }

            imshow("warp", roi);
        }

        cv::imshow("Camera", frame_copy);
        if (cv::waitKey(10) == 27) stop(); // tekan ESC untuk keluar

}

void detect::drawPred(int classId, float conf, int left, int top, int right, int bottom,
              cv::Mat& frame, const std::vector<std::string>& classNames)
{
    rectangle(frame, cv::Point(left, top), cv::Point(right, bottom), cv::Scalar(0, 255, 0), 2);
    std::string label = cv::format("%.2f", conf);
    if (!classNames.empty() && classId < (int)classNames.size())
        label = classNames[classId] + ": " + label;
    putText(frame, label, cv::Point(left, top > 15 ? top - 5 : top + 15),
            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
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
