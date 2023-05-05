// AIA2023Group5.cpp : This file contains the 'main' function. Program execution begins and ends there.
//


#include "wtypes.h"
#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core.hpp>

#include "openvino/openvino.hpp"

#include <utils/images_capture.h>
#include <utils/slog.hpp>

#include "gazeEstimation/face_inference_results.hpp"
#include "gazeEstimation/face_detector.hpp"
#include "gazeEstimation/base_estimator.hpp"
#include "gazeEstimation/head_pose_estimator.hpp"
#include "gazeEstimation/landmarks_estimator.hpp"
#include "gazeEstimation/eye_state_estimator.hpp"
#include "gazeEstimation/gaze_estimator.hpp"
#include "gazeEstimation/results_marker.hpp"
#include "gazeEstimation/utils.hpp"

using namespace cv;
using namespace gaze_estimation;

#define CVUI_IMPLEMENTATION
#include "cvui.h"

#include <fstream>
#include <sstream>
#include <chrono>
#include <ctime> 
#include <time.h>

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

std::string sizeString = "640x360";

cv::Size frameSize = stringToSize(sizeString);

int main()
{
    std::cout << "AIA2023 Group5 Demo\n";

    // Load OpenVINO runtime
    slog::info << ov::get_openvino_version() << slog::endl;

    ov::Core core;
    std::string FLAGS_m_fd = "..\\intel\\face-detection-retail-0004\\FP32\\face-detection-retail-0004.xml"
        , FLAGS_d_fd = "GPU"
        , FLAGS_m_hp = "..\\intel\\head-pose-estimation-adas-0001\\FP32\\head-pose-estimation-adas-0001.xml"
        , FLAGS_d_hp = "GPU"
        , FLAGS_m_lm = "..\\intel\\facial-landmarks-35-adas-0002\\FP32\\facial-landmarks-35-adas-0002.xml"
        , FLAGS_d_lm = "GPU"
        , FLAGS_m_es = "..\\public\\open-closed-eye-0001\\FP32\\open-closed-eye-0001.xml"
        , FLAGS_d_es = "GPU"
        , FLAGS_m = "..\\intel\\gaze-estimation-adas-0002\\FP32\\gaze-estimation-adas-0002.xml"
        , FLAGS_d = "GPU";

    // Set up face detector and estimators
    FaceDetector faceDetector(core, FLAGS_m_fd, FLAGS_d_fd, 0.5, false);
    HeadPoseEstimator headPoseEstimator(core, FLAGS_m_hp, FLAGS_d_hp);
    LandmarksEstimator landmarksEstimator(core, FLAGS_m_lm, FLAGS_d_lm);
    EyeStateEstimator eyeStateEstimator(core, FLAGS_m_es, FLAGS_d_es);
    GazeEstimator gazeEstimator(core, FLAGS_m, FLAGS_d);

    // Put pointers to all estimators in an array so that they could be processed uniformly in a loop
    BaseEstimator* estimators[] = {
        &headPoseEstimator,
        &landmarksEstimator,
        &eyeStateEstimator,
        &gazeEstimator
    };
    // Each element of the vector contains inference results on one face
    std::vector<FaceInferenceResults> inferenceResults;
    bool flipImage = false;
    ResultsMarker resultsMarker(true, true, true, true, true);
    int delay = 1;
    std::string windowName = "Gaze estimation demo";

    std::unique_ptr<ImagesCapture> cap = openImagesCapture("0", false, read_type::efficient, 0, std::numeric_limits<size_t>::max(), frameSize);

    cvui::init("MainSource");
    cv::resizeWindow("MainSource", frameSize.width, frameSize.height);
    int horizontal = 0;
    int vertical = 0;
    GetDesktopResolution(horizontal, vertical);
    cv::moveWindow("MainSource", 0, 0);

    while (true)
    {
        cv::Mat frame = cap->read();
        cv::Size graphSize{ frame.cols / 4, 60 };

        // Infer results
        auto inferenceResults = faceDetector.detect(frame);
        for (auto& inferenceResult : inferenceResults) {
            for (auto estimator : estimators) {
                estimator->estimate(frame, inferenceResult);
            }
        }

        float gazeH = 100;
        float gazeV = 100;
        if (inferenceResults.empty())
        {
        }
        else
        {
            auto const& inferenceResult = inferenceResults[0];
            resultsMarker.mark(frame, inferenceResult);
            cv::Point2f gazeAngles;
            gazeVectorToGazeAngles(inferenceResult.gazeVector, gazeAngles);
            gazeH = gazeAngles.x;
            gazeV = gazeAngles.y;

        }
        cv::putText(frame, "gaze angle H: " + std::to_string(std::round(gazeH)) + " V: " + std::to_string(std::round(gazeV)), cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);

        // Update cvui stuff and show everything on the screen
        cvui::imshow("MainSource", frame);
        if (waitKey(1) == 27) { break; }
    }
}

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file
