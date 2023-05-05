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

std::string windowName = "AIA2023 Group5 Demo";
std::string sizeString = "1280x720";

cv::Size frameSize = stringToSize(sizeString);

// resize down
cv::Size downSize = cv::Size(640 / 3, 360 / 3);
cv::Size downSizeVideo = cv::Size(1280 - (640 / 3) - 10, 720 - (360 / 3));
cv::Size reSize = cv::Size(640 , 360 );
std::unique_ptr<ImagesCapture> cap;

std::string FLAGS_m_fd = //"..\\intel\\face-detection-retail-0004\\FP32\\face-detection-retail-0004.xml"
"..\\intel\\face-detection-adas-0001\\FP32\\face-detection-adas-0001.xml"
, FLAGS_d_fd = "GPU"
, FLAGS_m_hp = "..\\intel\\head-pose-estimation-adas-0001\\FP32\\head-pose-estimation-adas-0001.xml"
, FLAGS_d_hp = "GPU"
, FLAGS_m_lm = "..\\intel\\facial-landmarks-35-adas-0002\\FP32\\facial-landmarks-35-adas-0002.xml"
, FLAGS_d_lm = "GPU"
, FLAGS_m_es = "..\\public\\open-closed-eye-0001\\FP32\\open-closed-eye-0001.xml"
, FLAGS_d_es = "GPU"
, FLAGS_m = "..\\intel\\gaze-estimation-adas-0002\\FP32\\gaze-estimation-adas-0002.xml"
, FLAGS_d = "GPU";
ResultsMarker resultsMarker(true, true, true, true, true);
// Put pointers to all estimators in an array so that they could be processed uniformly in a loop
std::vector< BaseEstimator*> estimators;

cv::Mat RunScene2(FaceDetector faceDetector) {

    cv::Mat frame = cap->read();
    cv::Size graphSize{ frame.cols / 4, 60 };

    // Infer results
    if (!estimators.empty()) {
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

    }
    resize(frame, frame, downSize, INTER_LINEAR);

    return frame;

}

std::vector<std::string> ids;
std::vector<std::string> names;
std::vector<Mat> features;

std::vector<std::string> stringSplit(std::string str, char delimiter)
{
    std::vector<std::string> tokens;
    std::string token;
    std::istringstream tokenStream(str);

    while (getline(tokenStream, token, delimiter))
    {
        tokens.push_back(token);
    }

    return tokens;
}

void loadDB(Ptr<FaceDetectorYN> detector, Ptr<FaceRecognizerSF> faceRecognizer) {

    features.clear();
    names.clear();
    ids.clear();

    std::string folder_path = "..\\faceDB\\";
    std::vector<std::string> filenames;
    glob(folder_path + "*.jpg", filenames, false);

    for (size_t i = 0; i < filenames.size(); i++)
    {
        Mat image = imread(filenames[i]);
        cv::Mat image1Resize;
        resize(image, image1Resize, reSize, INTER_LINEAR);
        Mat faces0;
        detector->detect(image1Resize, faces0);

        float ration = image.rows / reSize.height;
        Mat aligned_face;
        faceRecognizer->alignCrop(image1Resize, faces0.row(0), aligned_face);

        // Run feature extraction with given aligned_face
        Mat feature;
        faceRecognizer->feature(aligned_face, feature);
        features.push_back(feature.clone());
        std::vector<std::string> tokens = stringSplit(filenames[i].substr(folder_path.length()), '_');

        ids.push_back(tokens[0]);
        names.push_back(tokens[1]);

    }
}

//#define DO_ESTIMATORS

bool isRunning = false;

int main()
{
    std::cout << "AIA2023 Group5 Demo\n";

    int horizontal = 0;
    int vertical = 0;
    GetDesktopResolution(horizontal, vertical);

    // Load OpenVINO runtime
    slog::info << ov::get_openvino_version() << slog::endl;

    ov::Core core;
    // Set up face detector and estimators
    //FaceDetector faceDetector(core, FLAGS_m_fd, FLAGS_d_fd, 0.5, false);

#ifdef DO_ESTIMATORS
    HeadPoseEstimator headPoseEstimator(core, FLAGS_m_hp, FLAGS_d_hp);
    LandmarksEstimator landmarksEstimator(core, FLAGS_m_lm, FLAGS_d_lm);
    EyeStateEstimator eyeStateEstimator(core, FLAGS_m_es, FLAGS_d_es);
    GazeEstimator gazeEstimator(core, FLAGS_m, FLAGS_d);
    estimators.push_back(&headPoseEstimator);
    estimators.push_back(&landmarksEstimator);
    estimators.push_back(&eyeStateEstimator);
    estimators.push_back(&gazeEstimator);
#endif // DO_ESTIMATORS

    cap = openImagesCapture("0", false, read_type::efficient, 0, std::numeric_limits<size_t>::max(), frameSize);

    cvui::init(windowName);
    cv::resizeWindow(windowName, horizontal ,vertical);
    cv::moveWindow(windowName, 0, 0);
    // Create a frame
    cv::Mat canvas = cv::Mat(cv::Size(horizontal, vertical), CV_8UC3);

    VideoCapture vid_capture("https://sample-videos.com/video123/mp4/720/big_buck_bunny_720p_20mb.mp4");
    // Print error message if the stream is invalid
    if (!vid_capture.isOpened())
    {
        std::cout << "Error opening video stream or file\n";
    }

    // cv::Mat cameraFrame = cv::Mat(downSize, CV_8UC3);

    isRunning = true;
    /*
     std::thread([&]() {
        while (isRunning)
        {
            Mat frame;
            // Initialize a boolean to check if frames are there or not
            bool isSuccess = vid_capture.read(frame);

            // If frames are present, show it
            if (isSuccess == true)
            {
                resize(frame, frame, downSizeVideo, INTER_LINEAR);
                cvui::image(canvas, 0, 0, frame);
            }
        }
        }).detach();
    */
   
    double cosine_similar_thresh = 0.363;
    double l2norm_similar_thresh = 1.128;
    // Initialize FaceDetectorYN
    Ptr<FaceDetectorYN> detector = FaceDetectorYN::create("..\\faceDB\\face_detection_yunet_2022mar.onnx", "", reSize);
    // Initialize FaceRecognizerSF
    Ptr<FaceRecognizerSF> faceRecognizer = FaceRecognizerSF::create("..\\faceDB\\face_recognition_sface_2021dec_int8.onnx", "");
    
    loadDB(detector, faceRecognizer);
   
    while (isRunning)
    {
        
        cv::Mat frame = cap->read();
        cv::Mat frameResize;
        resize(frame, frameResize, reSize, INTER_LINEAR);
        Mat faces;
        detector->detect(frameResize, faces);

        //find main face
        int maxArea = 0;
        int maxFace=-1;
        for (int i = 0; i < faces.rows;i++) {
            cv::Mat face = faces.row(i);
            cv::Rect box(face.at<float>(0), face.at<float>(1) , face.at<float>(2) , face.at<float>(3));
            
            cv::rectangle(frameResize, box, Scalar(255, 0, 0), 2);
            int area = box.width * box.height;
            if (area > maxArea)
            {
                maxArea = area;
                maxFace =i;
            }
        }

        if (maxFace >= 0)
        {
            //has max face
            cv::Mat face = faces.row(maxFace);
            cv::Rect box(face.at<float>(0), face.at<float>(1), face.at<float>(2), face.at<float>(3));
            Mat aligned_face;
            Mat feature_target;
            faceRecognizer->alignCrop(frameResize, face, aligned_face);
            faceRecognizer->feature(aligned_face, feature_target);

            for (Mat feature : features) {

                double cos_score = faceRecognizer->match(feature, feature_target, FaceRecognizerSF::DisType::FR_COSINE);
                double L2_score = faceRecognizer->match(feature, feature_target, FaceRecognizerSF::DisType::FR_NORM_L2);

                if (cos_score >= cosine_similar_thresh && L2_score <= l2norm_similar_thresh){
                    cv::putText(frameResize, "ID: "+ids[maxFace]+" Name: "+names[maxFace], cv::Point(box.x, box.y - 20), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);
                    break;
                }
                else
                {

                }
                /*
                
                 if (cos_score >= cosine_similar_thresh)
            {
                std::cout << "They have the same identity;";
            }
            else
            {
                std::cout << "They have different identities;";
            }
            std::cout << " Cosine Similarity: " << cos_score << ", threshold: " << cosine_similar_thresh << ". (higher value means higher similarity, max 1.0)\n";
            if (L2_score <= l2norm_similar_thresh)
            {
                std::cout << "They have the same identity;";
            }
            else
            {
                std::cout << "They have different identities.";
            }
            std::cout << " NormL2 Distance: " << L2_score << ", threshold: " << l2norm_similar_thresh << ". (lower value means higher similarity, min 0.0)\n";
                */
            }


        }

       // cv::Mat cameraFrame = RunScene2(faceDetector);
       // cvui::window(canvas, horizontal - downSize.width - 10, 10, downSize.width, downSize.height, "Student");
       // cvui::image(canvas, horizontal - downSize.width - 10, 30, cameraFrame);

        cvui::update();
        cvui::imshow(windowName, frameResize
           // canvas
        );
        if (waitKey(1) == 27) {
            isRunning = false;
            Sleep(1000);
            break; }
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
