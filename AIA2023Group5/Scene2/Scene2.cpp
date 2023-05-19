// Scene2.cpp : This file contains the 'main' function. Program execution begins and ends there.
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

char RecordTime[100];
int FaceBoundingBox_X, FaceBoundingBox_Y, FaceBoundingBox_W, FaceBoundingBox_H;
float HeadPoseAngles_X, HeadPoseAngles_Y, HeadPoseAngles_Z;
int LeftEyeBoundingBox_X, LeftEyeBoundingBox_Y, LeftEyeBoundingBox_W, LeftEyeBoundingBox_H;
int RightEyeBoundingBox_X, RightEyeBoundingBox_Y, RightEyeBoundingBox_W, RightEyeBoundingBox_H;
float LeftEyeMidPoint_X, LeftEyeMidPoint_Y, RightEyeMidPoint_X, RightEyeMidPoint_Y;
float GazeVector_X, GazeVector_Y, GazeVector_Z;
bool EyeState_Left , EyeState_Right;
int RecordType=0;

bool recoding = false;
std::fstream file;

void setRecordTime() {
    time_t now = time(nullptr);
    auto tm_info = localtime(&now);
    strftime(RecordTime, 100, "%Y-%m-%d %H%M%S", tm_info);
    puts(RecordTime);
}

void resetAtt() {
    FaceBoundingBox_X = 0;FaceBoundingBox_Y = 0; FaceBoundingBox_W = 0; FaceBoundingBox_H = 0;
    HeadPoseAngles_X = 1000; HeadPoseAngles_Y = 1000; HeadPoseAngles_Z = 1000;
    LeftEyeBoundingBox_X = 0; LeftEyeBoundingBox_Y = 0; LeftEyeBoundingBox_W = 0; LeftEyeBoundingBox_H = 0;
    RightEyeBoundingBox_X = 0; RightEyeBoundingBox_Y = 0; RightEyeBoundingBox_W = 0; RightEyeBoundingBox_H = 0;
    LeftEyeMidPoint_X = 0; LeftEyeMidPoint_Y = 0; RightEyeMidPoint_X = 0; RightEyeMidPoint_Y = 0;
    float GazeVector_X = 1000; GazeVector_Y = 1000; GazeVector_Z = 1000;
}

void writeAtt(FaceInferenceResults ir) {
    FaceBoundingBox_X = ir.faceBoundingBox.x;
    FaceBoundingBox_Y = ir.faceBoundingBox.y;
    FaceBoundingBox_W = ir.faceBoundingBox.width;
    FaceBoundingBox_H = ir.faceBoundingBox.height;
    HeadPoseAngles_X = ir.headPoseAngles.x;
    HeadPoseAngles_Y = ir.headPoseAngles.y;
    HeadPoseAngles_Z = ir.headPoseAngles.z;
    LeftEyeBoundingBox_X = ir.leftEyeBoundingBox.x;
    LeftEyeBoundingBox_Y = ir.leftEyeBoundingBox.y;
    LeftEyeBoundingBox_W = ir.leftEyeBoundingBox.width;
    LeftEyeBoundingBox_H = ir.leftEyeBoundingBox.height;
    RightEyeBoundingBox_X = ir.rightEyeBoundingBox.x;
    RightEyeBoundingBox_Y = ir.rightEyeBoundingBox.y;
    RightEyeBoundingBox_W = ir.rightEyeBoundingBox.width;
    RightEyeBoundingBox_H = ir.rightEyeBoundingBox.height;
    LeftEyeMidPoint_X = ir.leftEyeMidpoint.x;
    LeftEyeMidPoint_Y = ir.leftEyeMidpoint.y;
    RightEyeMidPoint_X = ir.rightEyeMidpoint.x;
    RightEyeMidPoint_Y = ir.rightEyeMidpoint.y;
    GazeVector_X = ir.gazeVector.x;
    GazeVector_Y = ir.gazeVector.y;
    GazeVector_Z = ir.gazeVector.z;
    EyeState_Left = ir.leftEyeState;
    EyeState_Right = ir.rightEyeState;
    setRecordTime();
    file << RecordTime << "," << FaceBoundingBox_X << "," << FaceBoundingBox_Y << "," << FaceBoundingBox_W << "," << FaceBoundingBox_H;

    for (int i = 0; i < 35; i++) {

        if (i >= ir.faceLandmarks.size()) {
            //no data
            file << "," << "" << "," << "";
        }
        else {
            file << "," << ir.faceLandmarks[i].x << "," << ir.faceLandmarks[i].y;
        }
        
    }
    file << "," << HeadPoseAngles_X << "," << HeadPoseAngles_Y << "," << HeadPoseAngles_Z;
    file << "," << EyeState_Left << "," << EyeState_Right;
    file << "," << LeftEyeBoundingBox_X << "," << LeftEyeBoundingBox_Y << "," << LeftEyeBoundingBox_W << "," << LeftEyeBoundingBox_H;
    file << "," << RightEyeBoundingBox_X << "," << RightEyeBoundingBox_Y << "," << RightEyeBoundingBox_W << "," << RightEyeBoundingBox_H;

    std::vector<cv::Point2f> eyeLandmarks = ir.getEyeLandmarks();
    for (int i = 0; i < 4; i++) {
        if (i >= eyeLandmarks.size()) {
            //no data
            file << "," << "" << "," << "";
        }
        else {
            file << "," << eyeLandmarks[i].x << "," << eyeLandmarks[i].y;
        }
    }
    file << "," << LeftEyeMidPoint_X << "," << LeftEyeMidPoint_Y << "," << RightEyeMidPoint_X << "," << RightEyeMidPoint_Y;
    file << "," << GazeVector_X << "," << GazeVector_Y << "," << GazeVector_Z;
    file << "," << RecordType << "\n";
}

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

int main()
{
    std::cout << "Hello World!\n";

    // Load OpenVINO runtime
    slog::info << ov::get_openvino_version() << slog::endl;

    ov::Core core;
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

    std::unique_ptr<ImagesCapture> cap = openImagesCapture( "0", false, read_type::efficient, 0, std::numeric_limits<size_t>::max(), stringToSize("1280x720"));

    auto startTime = std::chrono::steady_clock::now();

    cvui::init("MainSource");
    // Resize the Window
    cv::resizeWindow("MainSource", 640, 360);
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

         if (recoding) {
             cv::putText(frame, "recoding type: "+ std::to_string(RecordType)+"..............", cv::Point(10, 100), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);

             if (inferenceResults.empty()) {
                //no data
             }
             else
             {
                 //get only first
                 FaceInferenceResults ir = inferenceResults[0];
                 resetAtt();
                 writeAtt(ir);
                
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
             //if (inferenceResult.leftEyeState && inferenceResult.rightEyeState) {
                 gazeVectorToGazeAngles(inferenceResult.gazeVector, gazeAngles);
                 gazeH = gazeAngles.x;
                 gazeV = gazeAngles.y;

             //}

         }
         cv::putText(frame, "gaze angle H: " + std::to_string(std::round(gazeH)) + " V: " + std::to_string(std::round(gazeV)), cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);
         

         //imshow("MainSource", frame);
         if (cvui::button(frame, (horizontal/2)-100, (vertical / 2)-100, "&Record   Concentration  ")) {
             if (recoding) {
                 file.close();
             }
             else{

                 //writefile
                 setRecordTime();
                 std::string fileName = RecordTime;
                 fileName= fileName.append(".csv");
                 file.open(fileName ,std::ios::out | std::ios::app);

                 file << "RecordTime" << "," << "FaceBoundingBox_X" << "," << "FaceBoundingBox_Y" << "," << "FaceBoundingBox_W" << "," << "FaceBoundingBox_H";

                 for (int i = 1; i <= 35; i++) {
                     file << "," << "FaceLandmarks_" << i << "_X" << "," << "FaceLandmarks_" << i << "_Y";
                 }
                 file << "," << "HeadPoseAngles_X" << "," << "HeadPoseAngles_Y" << "," << "HeadPoseAngles_Z";
                 file << "," << "EyeState_Left" << "," << "EyeState_Right";
                 file << "," << "LeftEyeBoundingBox_X" << "," << "LeftEyeBoundingBox_Y" << "," << "LeftEyeBoundingBox_W" << "," << "LeftEyeBoundingBox_H";
                 file << "," << "RightEyeBoundingBox_X" << "," << "RightEyeBoundingBox_Y" << "," << "RightEyeBoundingBox_W" << "," << "RightEyeBoundingBox_H";
                 for (int i = 1; i <= 4; i++) {
                     file << "," << "EyeLandmarks_" << i << "_X" << "," << "EyeLandmarks_" << i << "_Y";
                 }
                 file << "," << "LeftEyeMidPoint_X" << "," << "LeftEyeMidPoint_Y" << "," << "RightEyeMidPoint_X" << "," << "RightEyeMidPoint_Y";
                 file << "," << "GazeVector_X" << "," << "GazeVector_Y" << "," << "GazeVector_Z";
                 file << "," << "RecordType" << "\n";
             }
             RecordType = 1;
             recoding = !recoding;
         }
         if (cvui::button(frame, (horizontal / 2)-100, (vertical / 2)-50, "&Record Not Concentration")) {
             if (recoding) {
                 file.close();
             }
             else {

                 //writefile
                 setRecordTime();
                 std::string fileName = RecordTime;
                 fileName = fileName.append(".csv");
                 file.open(fileName, std::ios::out | std::ios::app);

                 file << "RecordTime" << "," << "FaceBoundingBox_X" << "," << "FaceBoundingBox_Y" << "," << "FaceBoundingBox_W" << "," << "FaceBoundingBox_H";

                 for (int i = 1; i <= 35; i++) {
                     file << "," << "FaceLandmarks_" << i << "_X" << "," << "FaceLandmarks_" << i << "_Y";
                 }
                 file << "," << "HeadPoseAngles_X" << "," << "HeadPoseAngles_Y" << "," << "HeadPoseAngles_Z";
                 file << "," << "EyeState_Left" << "," << "EyeState_Right";
                 file << "," << "LeftEyeBoundingBox_X" << "," << "LeftEyeBoundingBox_Y" << "," << "LeftEyeBoundingBox_W" << "," << "LeftEyeBoundingBox_H";
                 file << "," << "RightEyeBoundingBox_X" << "," << "RightEyeBoundingBox_Y" << "," << "RightEyeBoundingBox_W" << "," << "RightEyeBoundingBox_H";
                 for (int i = 1; i <= 4; i++) {
                     file << "," << "EyeLandmarks_" << i << "_X" << "," << "EyeLandmarks_" << i << "_Y";
                 }
                 file << "," << "LeftEyeMidPoint_X" << "," << "LeftEyeMidPoint_Y" << "," << "RightEyeMidPoint_X" << "," << "RightEyeMidPoint_Y";
                 file << "," << "GazeVector_X" << "," << "GazeVector_Y" << "," << "GazeVector_Z";
                 file << "," << "RecordType" << "\n";
             }
             RecordType = 0;
             recoding = !recoding;
         }

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
