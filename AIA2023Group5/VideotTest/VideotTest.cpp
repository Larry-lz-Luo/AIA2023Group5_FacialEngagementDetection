// VideotTest.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>

using namespace cv;
using namespace gaze_estimation;

#define CVUI_IMPLEMENTATION
#include "cvui.h"

// Get the horizontal and vertical screen sizes in pixel
void GetDesktopResolution(int& horizontal, int& vertical)
{
    RECT desktop;
    // Get a handle to the desktop window
    const HWND hDesktop = GetDesktopWindow();
    // Get the size of screen to the variable desktop
    GetWindowRect(hDesktop, &desktop);
    // The top left corner will have coordinates (0,0)
    // and the bottom right corner will have coordinates
    // (horizontal, vertical)
    horizontal = desktop.right;
    vertical = desktop.bottom;
}

std::string FLAGS_m_fd = "..\\models\\intel\\face-detection-retail-0004\\FP32\\face-detection-retail-0004.xml"
, FLAGS_d_fd = "GPU"
, FLAGS_m_hp = "..\\models\\intel\\head-pose-estimation-adas-0001\\FP32\\head-pose-estimation-adas-0001.xml"
, FLAGS_d_hp = "GPU"
, FLAGS_m_lm = "..\\models\\intel\\facial-landmarks-35-adas-0002\\FP32\\facial-landmarks-35-adas-0002.xml"
, FLAGS_d_lm = "GPU"
, FLAGS_m_es = "..\\models\\public\\open-closed-eye-0001\\FP32\\open-closed-eye-0001.xml"
, FLAGS_d_es = "GPU"
, FLAGS_m = "..\\models\\intel\\gaze-estimation-adas-0002\\FP32\\gaze-estimation-adas-0002.xml"
, FLAGS_d = "GPU";
ResultsMarker resultsMarker(true, true, true, true, true);

FaceDetector* faceDetector;
HeadPoseEstimator* headPoseEstimator;
LandmarksEstimator* landmarksEstimator;
EyeStateEstimator* eyeStateEstimator;
GazeEstimator* gazeEstimator;
// Put pointers to all estimators in an array so that they could be processed uniformly in a loop
std::vector< BaseEstimator*> estimators;

std::string windowName = "AIA2023 Group5 VideoTest";
std::string sizeString = "640x480";
cv::Size frameSize = stringToSize(sizeString);

cv::Size downSize = cv::Size(640 , 480);
cv::Mat status = cv::Mat(cv::Size(1000, 100), CV_8UC3);
std::unique_ptr<ImagesCapture> cap;

#define DO_ESTIMATORS

bool isRunning = false;

//video path
std::string videoPath =  //"0";//"0" is camera
"https://sample-videos.com/video123/mp4/720/big_buck_bunny_720p_20mb.mp4";

void Init() {

    // Load OpenVINO runtime
    slog::info << ov::get_openvino_version() << slog::endl;

    ov::Core core;

    // Set up face detector and estimators
    faceDetector = new FaceDetector(core, FLAGS_m_fd, FLAGS_d_fd, 0.5, false);

#ifdef DO_ESTIMATORS
    headPoseEstimator = new HeadPoseEstimator(core, FLAGS_m_hp, FLAGS_d_hp);
    landmarksEstimator = new LandmarksEstimator(core, FLAGS_m_lm, FLAGS_d_lm);
    eyeStateEstimator = new EyeStateEstimator(core, FLAGS_m_es, FLAGS_d_es);
    gazeEstimator = new GazeEstimator(core, FLAGS_m, FLAGS_d);
    estimators.push_back(headPoseEstimator);
    estimators.push_back(landmarksEstimator);
    estimators.push_back(eyeStateEstimator);
    estimators.push_back(gazeEstimator);
#endif // DO_ESTIMATORS


}

void Release() {

    if (faceDetector)delete faceDetector;

    if (headPoseEstimator)delete headPoseEstimator;
    if (landmarksEstimator)delete  landmarksEstimator;
    if (eyeStateEstimator)delete eyeStateEstimator;
    if (gazeEstimator)delete gazeEstimator;

    estimators.clear();
}

std::mutex mu;
std::vector<float> recordStatus;

cv::Mat RunScene2(cv::Mat canvas) {

    cv::Mat frame = cap->read();
    cv::Size graphSize{ frame.cols / 4, 60 };
    // Infer results
    if (!estimators.empty()) {
        //find main face
        int maxArea = 0;
        int maxFace = -1;
        auto inferenceResults = faceDetector->detect(frame);
        for (int i = 0; i < inferenceResults.size(); i++) {

            auto& inferenceResult = inferenceResults[i];
            for (auto estimator : estimators) {
                estimator->estimate(frame, inferenceResult);
            }

            int area = inferenceResult.faceBoundingBox.width * inferenceResult.faceBoundingBox.height;
            if (area > maxArea)
            {
                maxArea = area;
                maxFace = i;
            }
        }

        float gazeH = 100;
        float gazeV = 100;
        if (maxFace >= 0) {
            auto const& inferenceResult = inferenceResults[maxFace];
            resultsMarker.mark(frame, inferenceResult);
            cv::Point2f gazeAngles;
            gazeVectorToGazeAngles(inferenceResult.gazeVector, gazeAngles);
            gazeH = gazeAngles.x;
            gazeV = gazeAngles.y;
        }

        //cv::putText(frame, "gaze angle H: " + std::to_string(std::round(gazeH)) + " V: " + std::to_string(std::round(gazeV)), cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);

        //check gaze
        {
            std::unique_lock<std::mutex> locker(mu);
            if (fabs(gazeH) > 21 || fabs(gazeV) > 12) {
                recordStatus.push_back(1);
            }
            else {
                recordStatus.push_back(0);
            }
            locker.unlock();
        }

    }

    cv::resize(frame, frame, downSize, INTER_LINEAR);
    int x = (canvas.cols /2)- (downSize.width /2);
    int y = 100;
    cvui::image(canvas, x, y , frame);

    return frame;

}

void RunCheckGaze() {
    std::thread([&]() {
        while (isRunning)
        {
            std::unique_lock<std::mutex> locker(mu);
            if (!recordStatus.empty()) {

                std::cout << "recordStatus.size():" << recordStatus.size() << "\n";
                float avg = std::accumulate(recordStatus.begin(), recordStatus.end(), 0.0f) / recordStatus.size();
                std::cout << "avg:" << avg << "\n";
                if (avg > 0.47) {
                    cv::putText(status, "Not concentrated", cv::Point(10, 50), cv::FONT_HERSHEY_SIMPLEX, 2, cv::Scalar(0, 0, 255), 2); // 在圖像上添加警報文字
                }
                else {
                    status.setTo(cv::Scalar(0, 0, 0));
                }

            }

            recordStatus.clear();
            locker.unlock();
            Sleep(1500);
        }
        }).detach();
}

int main()
{
    std::cout << "VideoTest\n";

    Init();

    int horizontal = 0, vertical = 0;
    GetDesktopResolution(horizontal, vertical);
    cvui::init(windowName);
    cv::resizeWindow(windowName, horizontal, vertical);
    cv::moveWindow(windowName, 0, 0);
    // Create a frame
    cv::Mat canvas = cv::Mat(cv::Size(horizontal, vertical), CV_8UC3);

    isRunning = true;
    //RunCheckGaze();
    //Load video
    cap = openImagesCapture(videoPath, false, read_type::efficient, 0, std::numeric_limits<size_t>::max(), frameSize);

    while (isRunning)
    {
        RunScene2(canvas);
        cvui::image(canvas, 0, 0, status);
        cvui::update();
        cvui::imshow(windowName, canvas);
        if (waitKey(1) == 27) {
            isRunning = false;
            Sleep(1000);
            break;
        }

        Sleep(33);
    }

    Release();
}
