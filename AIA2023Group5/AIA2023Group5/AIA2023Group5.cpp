// AIA2023Group5.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <windows.h>
HWND hwnd;

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


#define DO_ESTIMATORS

LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam);

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

std::string windowName = "AIA2023 Group5 Demo";
std::string sizeString = "1280x720";
cv::Size frameSize = stringToSize(sizeString);

cv::Size downSize = cv::Size(640 / 3, 360 / 3);
cv::Size downSizeVideo = cv::Size(1280 - (640 / 3) - 10, 720 - (360 / 3));
cv::Size reSize = cv::Size(640*1.5 , 360*1.5);
cv::Mat status = cv::Mat(cv::Size(1000, 50), CV_8UC3);
cv::Mat status2 = cv::Mat(cv::Size(1000, 50), CV_8UC3);
std::unique_ptr<ImagesCapture> cap;

std::string FLAGS_m_fd = "..\\intel\\face-detection-retail-0004\\FP32\\face-detection-retail-0004.xml"
, FLAGS_d_fd = "GPU"
, FLAGS_m_hp = "..\\intel\\head-pose-estimation-adas-0001\\FP32\\head-pose-estimation-adas-0001.xml"
, FLAGS_d_hp = "GPU"
, FLAGS_m_lm = "..\\intel\\facial-landmarks-35-adas-0002\\FP32\\facial-landmarks-35-adas-0002.xml"
, FLAGS_d_lm = "GPU"
, FLAGS_m_es = "..\\public\\open-closed-eye-0001\\FP32\\open-closed-eye-0001.xml"
, FLAGS_d_es = "GPU"
, FLAGS_m = "..\\intel\\gaze-estimation-adas-0002\\FP32\\gaze-estimation-adas-0002.xml"
, FLAGS_d = "GPU"
, FLAGS_m_fr= "..\\faceDB\\face_recognition_sface_2021dec_int8.onnx";
ResultsMarker resultsMarker(true, true, true, true, true);

FaceDetector *faceDetector;
HeadPoseEstimator *headPoseEstimator;
LandmarksEstimator *landmarksEstimator;
EyeStateEstimator *eyeStateEstimator;
GazeEstimator *gazeEstimator;
// Put pointers to all estimators in an array so that they could be processed uniformly in a loop
std::vector< BaseEstimator*> estimators;

Ptr<FaceRecognizerSF> faceRecognizer;

double cosine_similar_thresh = 0.45;// 0.363;
double l2norm_similar_thresh = 0.98;// 1.128;

int sceneStatus = 2;
cv::Mat cameraFrame;
bool isRunning = false;

std::string folder_path = "..\\faceDB\\";
std::vector<std::string> ids;
std::vector<std::string> names;
std::vector<Mat> features;

#include <xgboost/c_api.h>
BoosterHandle booster;

std::mutex mu;
std::vector<float> recordStatusWithXGBooster;
bool resultWithXGBooster = false;
std::vector<std::vector<float>> recordInferenceResults;


std::vector<float> getFaceInferenceData6(FaceInferenceResults inferenceResult) {

    return 
    { 
        inferenceResult.gazeVector.x,inferenceResult.gazeVector.y,inferenceResult.gazeVector.z
        ,inferenceResult.headPoseAngles.x,inferenceResult.headPoseAngles.y,inferenceResult.headPoseAngles.z
    };
}

std::vector<float> getFaceInferenceData76(FaceInferenceResults inferenceResult) {
    std::vector<float> result;

    result.push_back(inferenceResult.gazeVector.x); result.push_back(inferenceResult.gazeVector.y); 
    result.push_back(inferenceResult.gazeVector.z);
    result.push_back(inferenceResult.headPoseAngles.x); result.push_back(inferenceResult.headPoseAngles.y);
    result.push_back(inferenceResult.headPoseAngles.z);
    for (int i = 0; i < inferenceResult.faceLandmarks.size(); i++) {
        //shift to BoundingBox 0,0 for normalize
        result.push_back(inferenceResult.faceLandmarks[i].x - inferenceResult.faceBoundingBox.x);
        result.push_back(inferenceResult.faceLandmarks[i].y - inferenceResult.faceBoundingBox.y);
    }

    return result;
}

std::vector<float> getFaceInferenceDataEDA(FaceInferenceResults inferenceResult) {
    std::vector<float> result;

    result.push_back(inferenceResult.faceBoundingBox.y); 
    result.push_back(inferenceResult.faceLandmarks[2-1].y- inferenceResult.faceBoundingBox.y);
    result.push_back(inferenceResult.faceLandmarks[13-1].y- inferenceResult.faceBoundingBox.y);
    result.push_back(inferenceResult.faceLandmarks[14-1].y- inferenceResult.faceBoundingBox.y);
    result.push_back(inferenceResult.faceLandmarks[15-1].y- inferenceResult.faceBoundingBox.y);
    result.push_back(inferenceResult.faceLandmarks[16-1].x- inferenceResult.faceBoundingBox.x);
    result.push_back(inferenceResult.faceLandmarks[16-1].y- inferenceResult.faceBoundingBox.y);
    result.push_back(inferenceResult.faceLandmarks[17-1].x- inferenceResult.faceBoundingBox.x);
    result.push_back(inferenceResult.faceLandmarks[17-1].y- inferenceResult.faceBoundingBox.y);
    result.push_back(inferenceResult.faceLandmarks[19-1].y- inferenceResult.faceBoundingBox.y);
    result.push_back(inferenceResult.headPoseAngles.y);
    result.push_back(inferenceResult.headPoseAngles.z);
    result.push_back(inferenceResult.leftEyeState);
    result.push_back(inferenceResult.rightEyeState);
    result.push_back(inferenceResult.leftEyeBoundingBox.y);
    result.push_back(inferenceResult.rightEyeBoundingBox.x); result.push_back(inferenceResult.rightEyeBoundingBox.y);
    result.push_back(inferenceResult.getEyeLandmarks()[2-1].y);
    result.push_back(inferenceResult.leftEyeMidpoint.y);
    result.push_back(inferenceResult.gazeVector.z);

    return result;
}

std::vector<float> getFaceInferenceDataEDA_jasonTest(FaceInferenceResults inferenceResult) {
    std::vector<float> result;

    result.push_back(inferenceResult.faceBoundingBox.y);
    result.push_back(inferenceResult.faceLandmarks[2 - 1].y - inferenceResult.faceBoundingBox.y);
    result.push_back(inferenceResult.faceLandmarks[13 - 1].y - inferenceResult.faceBoundingBox.y);
    result.push_back(inferenceResult.faceLandmarks[14 - 1].y - inferenceResult.faceBoundingBox.y);
    result.push_back(inferenceResult.faceLandmarks[15 - 1].y - inferenceResult.faceBoundingBox.y);
    result.push_back(inferenceResult.faceLandmarks[16 - 1].x - inferenceResult.faceBoundingBox.x);
    result.push_back(inferenceResult.faceLandmarks[16 - 1].y - inferenceResult.faceBoundingBox.y);
    result.push_back(inferenceResult.faceLandmarks[17 - 1].x - inferenceResult.faceBoundingBox.x);
    result.push_back(inferenceResult.faceLandmarks[17 - 1].y - inferenceResult.faceBoundingBox.y);
    result.push_back(inferenceResult.faceLandmarks[19 - 1].y - inferenceResult.faceBoundingBox.y);
    result.push_back(inferenceResult.headPoseAngles.y);
    result.push_back(inferenceResult.headPoseAngles.z);
    result.push_back(inferenceResult.leftEyeState);
    result.push_back(inferenceResult.rightEyeState);
    result.push_back(inferenceResult.rightEyeBoundingBox.x); 
    result.push_back(inferenceResult.getEyeLandmarks()[2 - 1].y);
    result.push_back(inferenceResult.gazeVector.z);

    return result;
}

std::vector<float> getFaceInferenceDataEDA_6pos6neg(FaceInferenceResults inferenceResult) {
    std::vector<float> result;

    result.push_back(inferenceResult.faceBoundingBox.y);
    result.push_back(inferenceResult.faceLandmarks[13 - 1].y - inferenceResult.faceBoundingBox.y);
    result.push_back(inferenceResult.faceLandmarks[14 - 1].y - inferenceResult.faceBoundingBox.y);
    result.push_back(inferenceResult.faceLandmarks[16 - 1].x - inferenceResult.faceBoundingBox.x);
    result.push_back(inferenceResult.faceLandmarks[17 - 1].x - inferenceResult.faceBoundingBox.x);
    result.push_back(inferenceResult.headPoseAngles.y);
    result.push_back(inferenceResult.headPoseAngles.z);
    result.push_back(inferenceResult.leftEyeState);
    result.push_back(inferenceResult.rightEyeState);
    result.push_back(inferenceResult.leftEyeBoundingBox.y);
    result.push_back(inferenceResult.rightEyeBoundingBox.x);
    result.push_back(inferenceResult.gazeVector.z);

    return result;
}

std::vector<float> getFaceInferenceDataEDA_FI(FaceInferenceResults inferenceResult) {
    std::vector<float> result;

    result.push_back(inferenceResult.headPoseAngles.y);
    result.push_back(inferenceResult.headPoseAngles.z);
    result.push_back(inferenceResult.gazeVector.x); 
    result.push_back(inferenceResult.gazeVector.y);
    result.push_back(inferenceResult.gazeVector.z);

    return result;
}

std::vector<float> getFaceInferenceData(FaceInferenceResults inferenceResult) {
    return 
        //getFaceInferenceData6(inferenceResult);
        //getFaceInferenceDataEDA(inferenceResult);
        //getFaceInferenceData76(inferenceResult);
        //getFaceInferenceDataEDA_jasonTest(inferenceResult);
        //getFaceInferenceDataEDA_6pos6neg(inferenceResult);
        getFaceInferenceDataEDA_FI(inferenceResult);
}

void loadXGBoosterSingle() {
    // 載入模型
    //int res = XGBoosterLoadModel(booster, "..\\models\\XGB_model_76features2.json");
    //int res = XGBoosterLoadModel(booster, "..\\models\\XGB_normalized_model.json");
    // 
    //Jason
    //int res = XGBoosterLoadModel(booster, "..\\models\\XGB_normalized_model_test.json");
    //int res = XGBoosterLoadModel(booster, "..\\models\\XGB_normalized_6pos6neg_model.json");
    
    //feature imporment
    int res = XGBoosterLoadModel(booster, "..\\models\\XGB_normalized_top5_model.json");
    
    std::cout << "XGBoosterLoadModel: " << res << "\n";
}

void loadXGBooster30Frame() {
    // 載入模型
    int res = XGBoosterLoadModel(booster, "..\\models\\XGB_model_DAiSEE.json");
    std::cout << "loadXGBooster30FrameLoadModel: " << res << "\n";
}

//single frame with gaze angle x y z
void checkGazeWithXGBoosterSingle(FaceInferenceResults inferenceResult) {
    // 載入預測資料
    recordInferenceResults.push_back(getFaceInferenceData(inferenceResult));
    //float data[1][3] = { };
    // 設定預測參數
    bst_ulong num_row = recordInferenceResults.size();
    bst_ulong num_col = recordInferenceResults[0].size();
    // 執行預測
    DMatrixHandle dtest;
    int ret = XGDMatrixCreateFromMat(&recordInferenceResults[0][0], num_row, num_col, NAN, &dtest);
    if (ret == 0) {
        bst_ulong out_len;
        const float* out_result = NULL;
        /* Run prediction with DMatrix object. */
        ret = XGBoosterPredict(booster, dtest, 0, 0, &out_len, &out_result);
        if (ret == 0) {
            // 輸出預測結果
            std::cout << "checkGazeWithXGBoosterSingle Predict result：" << out_result[0] << std::endl;
            recordStatusWithXGBooster.push_back(out_result[0]);
            // 釋放資源
            XGDMatrixFree(dtest);
        }

    }
    recordInferenceResults.clear();

    std::unique_lock<std::mutex> locker(mu);
    if (recordStatusWithXGBooster.size() >= 30) {
        float avg = std::accumulate(recordStatusWithXGBooster.begin(), recordStatusWithXGBooster.end(), 0.0f) / recordStatusWithXGBooster.size();
        std::cout << "recordStatusWithXGBooster avg:" << avg << "\n";
        if (avg < 0.5) {
            //not concentrated
            resultWithXGBooster = false;
        }
        else {
            resultWithXGBooster = true;
        }

        recordStatusWithXGBooster.clear();
    }
    locker.unlock();

}

void checkGazeWithXGBooster30Frame(FaceInferenceResults inferenceResult) {

    std::unique_lock<std::mutex> locker(mu);
    recordInferenceResults.push_back(getFaceInferenceData(inferenceResult));
    if (recordInferenceResults.size() >= 30) {
        // 載入預測資料
        // 設定預測參數
        bst_ulong num_row = 30;
        bst_ulong num_col = 3;
        // 執行預測
        DMatrixHandle dtest;
        int ret = XGDMatrixCreateFromMat(&recordInferenceResults[0][0], num_row, num_col, NAN, &dtest);
        if (ret == 0) {
            bst_ulong out_len;
            const float* out_result = NULL;
            /* Run prediction with DMatrix object. */
            ret = XGBoosterPredict(booster, dtest, 0, 0, &out_len, &out_result);
            if (ret == 0) {
                // 輸出預測結果
                std::cout << "checkGazeWithXGBooster30Frame Predict result：" << out_result[0] << std::endl;
                if (out_result[0] < 0.5) {
                    //not concentrated
                    resultWithXGBooster = false;
                }
                else {
                    resultWithXGBooster = true;
                }
                // 釋放資源
                XGDMatrixFree(dtest);
            }

        }
        recordInferenceResults.clear();
    }
    locker.unlock();

}

void checkGazeWithXGBooster(FaceInferenceResults inferenceResult) {
    checkGazeWithXGBoosterSingle(inferenceResult);
    //checkGazeWithXGBooster30Frame(inferenceResult);
}

void loadXGBooster() {

    int res = XGBoosterCreate(NULL, 0, &booster);
    std::cout << "XGBoosterCreate: " << res << "\n";
    loadXGBoosterSingle();
    //loadXGBooster30Frame();
}

std::vector<float> recordStatusWithAngles;
bool resultWithAngles = false;
void checkGazeWithAngles(FaceInferenceResults inferenceResult) {

    float gazeH = 100;
    float gazeV = 100;
    cv::Point2f gazeAngles;
    gazeVectorToGazeAngles(inferenceResult.gazeVector, gazeAngles);
    //check gaze
    {

        gazeH = gazeAngles.x;
        gazeV = gazeAngles.y;
        std::unique_lock<std::mutex> locker(mu);
        if (fabs(gazeH) > 21 || fabs(gazeV) > 12) {
            //not concentrated
            recordStatusWithAngles.push_back(0);
        }
        else {
            recordStatusWithAngles.push_back(1);
        }

        if (recordStatusWithAngles.size() >= 30) {
            float avg = std::accumulate(recordStatusWithAngles.begin(), recordStatusWithAngles.end(), 0.0f) / recordStatusWithAngles.size();
           // std::cout << "GazeAngles avg:" << avg << "\n";
            if (avg < 0.47) {
                resultWithAngles = false;
            }
            else {
                resultWithAngles = true;
            }
            
            recordStatusWithAngles.clear();
        }
        locker.unlock();
    }
}

void updateStatusThread() {

    std::thread([&]() {
        while (sceneStatus == 3 && isRunning)
        {
            std::unique_lock<std::mutex> locker(mu);
            if (!resultWithAngles) {
                cv::putText(status, "Not concentrated", cv::Point(10, 40), cv::FONT_HERSHEY_SIMPLEX, 2, cv::Scalar(0, 0, 255), 2); // 在圖像上添加警報文字
            }
            else {
                status.setTo(cv::Scalar(0, 0, 0));
            }

            if (!resultWithXGBooster) {
                cv::putText(status2, "Not concentrated", cv::Point(10, 40), cv::FONT_HERSHEY_SIMPLEX, 2, cv::Scalar(255, 0, 0), 2); // 在圖像上添加警報文字
            }
            else {
                status2.setTo(cv::Scalar(0, 0, 0));
            }
            locker.unlock();
            Sleep(1000);
        }
        }).detach();
}

void loadDB() {

    features.clear();
    names.clear();
    ids.clear();

    std::vector<std::string> filenames;
    glob(folder_path + "*.jpg", filenames, false);

    for (size_t i = 0; i < filenames.size(); i++)
    {
        Mat image = imread(filenames[i]);
        resize(image, image, reSize, INTER_LINEAR);

        auto inferenceResults = faceDetector->detect(image);

        //find main face
        int maxArea = 0;
        int maxFace = -1;
        for (int i = 0; i < inferenceResults.size(); i++) {

            auto& inferenceResult = inferenceResults[i];
            //cv::rectangle(image, inferenceResult.faceBoundingBox, Scalar(255, 0, 0), 2);
            int area = inferenceResult.faceBoundingBox.width * inferenceResult.faceBoundingBox.height;
            if (area > maxArea)
            {
                maxArea = area;
                maxFace = i;
            }
        }

        if (maxFace >= 0)
        {
            cv::Rect box = inferenceResults[maxFace].faceBoundingBox;
            Mat aligned_face = image(box);
            // Run feature extraction with given aligned_face
            Mat feature;
            faceRecognizer->feature(aligned_face, feature);
            features.push_back(feature.clone());
            std::vector<std::string> tokens = stringSplit(filenames[i].substr(folder_path.length()), '_');

            ids.push_back(tokens[0]);
            names.push_back(tokens[1]);
        }

    }
}

void Init() {

    // Load OpenVINO runtime
    slog::info << ov::get_openvino_version() << slog::endl;

    ov::Core core;

    // Set up face detector and estimators
    faceDetector = new FaceDetector(core, FLAGS_m_fd, FLAGS_d_fd, 0.5, false);

    // Initialize FaceRecognizerSF
    faceRecognizer = FaceRecognizerSF::create(FLAGS_m_fr, "");

#ifdef DO_ESTIMATORS
    headPoseEstimator=new HeadPoseEstimator(core, FLAGS_m_hp, FLAGS_d_hp);
    landmarksEstimator=new LandmarksEstimator (core, FLAGS_m_lm, FLAGS_d_lm);
    eyeStateEstimator=new EyeStateEstimator (core, FLAGS_m_es, FLAGS_d_es);
    gazeEstimator=new GazeEstimator (core, FLAGS_m, FLAGS_d);
    estimators.push_back(headPoseEstimator);
    estimators.push_back(landmarksEstimator);
    estimators.push_back(eyeStateEstimator);
    estimators.push_back(gazeEstimator);
#endif // DO_ESTIMATORS

    //Alert UI Window
    {
        WNDCLASS wc = {};
        wc.lpfnWndProc = WindowProc;
        wc.hInstance = GetModuleHandle(NULL);
        wc.lpszClassName = L"MyClass";
        RegisterClass(&wc);
        // 建立視窗
        hwnd = CreateWindow(L"MyClass", L"User Name", WS_OVERLAPPEDWINDOW & ~WS_SYSMENU,
            CW_USEDEFAULT, CW_USEDEFAULT, 330, 150,
            NULL, NULL, GetModuleHandle(NULL), NULL);

        // 建立 Text Box 控制項
        HWND hTextBox = CreateWindowEx(WS_EX_CLIENTEDGE, L"EDIT", NULL,
            WS_CHILD | WS_VISIBLE | ES_MULTILINE | ES_AUTOVSCROLL | ES_AUTOHSCROLL,
            10, 20, 200, 30, hwnd, NULL, GetModuleHandle(NULL), NULL);

        // 建立按鈕控制項
        HWND hButton = CreateWindow(L"BUTTON", L"Register",
            WS_TABSTOP | WS_VISIBLE | WS_CHILD | BS_DEFPUSHBUTTON,
            210, 10, 80, 30, hwnd, (HMENU)1, GetModuleHandle(NULL), NULL);

        HWND hButton2 = CreateWindow(L"BUTTON", L"Cancel", WS_TABSTOP | WS_VISIBLE | WS_CHILD | BS_DEFPUSHBUTTON,
            210, 50, 80, 30, hwnd, (HMENU)2, GetModuleHandle(NULL), NULL);

        std::thread([&]() {
            // 訊息迴圈
            MSG msg = {};
        while (GetMessage(&msg, NULL, 0, 0))
        {
            TranslateMessage(&msg);
            DispatchMessage(&msg);

        }
            }).detach();
    }

    cap = openImagesCapture("0", false, read_type::efficient, 0, std::numeric_limits<size_t>::max(), frameSize);
    
}

void Release() {

    if (faceDetector)delete faceDetector;

    if (headPoseEstimator)delete headPoseEstimator;
    if (landmarksEstimator)delete  landmarksEstimator;
    if (eyeStateEstimator)delete eyeStateEstimator;
    if (gazeEstimator)delete gazeEstimator;

    estimators.clear();

    XGBoosterFree(booster);
}

void RunScene3() {
    cameraFrame = cap->read();
    sceneStatus = 4;
    // 顯示視窗
    ShowWindow(hwnd, SW_SHOWDEFAULT);
    UpdateWindow(hwnd);
}

cv::Mat RunScene1(cv::Mat canvas) {

    cv::Mat frame = cap->read();
    cv::resize(frame, frame, reSize, INTER_LINEAR);

    auto inferenceResults = faceDetector->detect(frame);

    //find main face
    int maxArea = 0;
    int maxFace = -1;
    for (int i = 0; i < inferenceResults.size(); i++) {

        auto& inferenceResult = inferenceResults[i];
        cv::rectangle(frame, inferenceResult.faceBoundingBox, Scalar(255, 0, 0), 2);
        int area = inferenceResult.faceBoundingBox.width * inferenceResult.faceBoundingBox.height;
        Mat feature_target;
        Mat aligned_face = frame(inferenceResult.faceBoundingBox);
        faceRecognizer->feature(aligned_face, feature_target);
        for (int j = 0;j< features.size();j++) {

            Mat feature = features[j];
            double cos_score = faceRecognizer->match(feature, feature_target, FaceRecognizerSF::DisType::FR_COSINE);
            double L2_score = faceRecognizer->match(feature, feature_target, FaceRecognizerSF::DisType::FR_NORM_L2);

            if (cos_score >= cosine_similar_thresh && L2_score <= l2norm_similar_thresh) {
                cv::putText(frame, "ID: " + ids[j] + " Name: " + names[j], cv::Point(inferenceResult.faceBoundingBox.x, inferenceResult.faceBoundingBox.y - 20), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);
                break;
            }
            else
            {

            }
        }
        if (area > maxArea)
        {
            maxArea = area;
            maxFace = i;
        }
    }

    Mat aligned_face;
    bool isMember = false;
    int indexId = 0;
    if (maxFace >= 0)
    {
        //has max face
        cv::Rect box = inferenceResults[maxFace].faceBoundingBox;
        Mat feature_target;
        aligned_face = frame(box);
        faceRecognizer->feature(aligned_face, feature_target);

        for (Mat feature : features) {

            double cos_score = faceRecognizer->match(feature, feature_target, FaceRecognizerSF::DisType::FR_COSINE);
            double L2_score = faceRecognizer->match(feature, feature_target, FaceRecognizerSF::DisType::FR_NORM_L2);

            if (cos_score >= cosine_similar_thresh && L2_score <= l2norm_similar_thresh) {
               // cv::putText(frame, "ID: " + ids[maxFace] + " Name: " + names[maxFace], cv::Point(box.x, box.y - 20), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);
                isMember = true;
                break;
            }
            else
            {

            }
            indexId++;
        }


    }

    int x = 50;
    int y = (canvas.rows / 2) - (reSize.height / 2);
    cvui::image(canvas, x, y, frame);

    if (cvui::button(canvas, reSize.width+x,y, "LOGIN")) {

        if (isMember) {
            status.setTo(cv::Scalar(0, 0, 0));
            cv::putText(status, "Welcome!! " + names[indexId]+" please wait.....", cv::Point(10, 50), cv::FONT_HERSHEY_SIMPLEX, 2, cv::Scalar(0, 255, 0), 2); // 在圖像上添加警報文字
            sceneStatus = 2;
        }
        else {
            status.setTo(cv::Scalar(0, 0, 0));
            cv::putText(status, "Login Fail" , cv::Point(10, 50), cv::FONT_HERSHEY_SIMPLEX, 2, cv::Scalar(0, 0, 255), 2); // 在圖像上添加警報文字
        }
        
    }

    if (cvui::button(canvas, reSize.width + x, y+30, "SIGN UP")) {

        RunScene3();
    }

    return frame;

}

cv::Mat RunScene2(cv::Mat canvas) {

    cv::Mat frame = cap->read();
    //cv::Size graphSize{ frame.cols / 4, 60 };
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

        if (maxFace >= 0){
            auto const& inferenceResult = inferenceResults[maxFace];
            resultsMarker.mark(frame, inferenceResult);
            checkGazeWithAngles(inferenceResult);
            checkGazeWithXGBooster(inferenceResult);

        }
        //cv::putText(frame, "gaze angle H: " + std::to_string(std::round(gazeH)) + " V: " + std::to_string(std::round(gazeV)), cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);

    }


    cv::resize(frame, frame, downSize, INTER_LINEAR);

    int x = canvas.cols - downSize.width - 10;
    int y = 10;
    cvui::window(canvas,x , y, downSize.width, downSize.height, "Participant");
    cvui::image(canvas, x, y+20, frame);

    
    if (cvui::button(canvas, x, y+ downSize.height+30, "LEAVE")) {
        sceneStatus = 0;
    }

    return frame;

}

int main()
{
    std::cout << "AIA2023 Group5 Demo\n";

    loadXGBooster();
   
    Init();

    int horizontal = 0, vertical = 0;
    GetDesktopResolution(horizontal, vertical);
    cvui::init(windowName);
    cv::resizeWindow(windowName, horizontal ,vertical);
    cv::moveWindow(windowName, 0, 0);
    // Create a frame
    cv::Mat canvas = cv::Mat(cv::Size(horizontal, vertical), CV_8UC3);

    isRunning = true;
   
    while (isRunning)
    {
        
        if (sceneStatus == 0) {
            status.setTo(cv::Scalar(0, 0, 0));
            canvas.setTo(cv::Scalar(0, 0, 0));
            loadDB();
            sceneStatus = 1;
        }
        else if (sceneStatus==1) {
            
             RunScene1(canvas);
            cvui::image(canvas, 0, 0, status);
        }
        else if (sceneStatus == 2) {
            VideoCapture vid_capture("https://sample-videos.com/video123/mp4/720/big_buck_bunny_720p_20mb.mp4");
            // Print error message if the stream is invalid
            if (!vid_capture.isOpened())
            {
                std::cout << "Error opening video stream or file\n";
            }
            else
            {
                status.setTo(cv::Scalar(0, 0, 0));
                status2.setTo(cv::Scalar(0, 0, 0));
                canvas.setTo(cv::Scalar(0, 0, 0));
                std::thread([&](VideoCapture vid_capture) {
                        while (sceneStatus >= 2 && isRunning)
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
                            else {
                        
                                if (frame.empty()) { // 如果影片播放完畢，則從頭開始播放
                                    vid_capture.set(cv::CAP_PROP_POS_FRAMES, 0);
                                    continue;
                                }
                            }
                            Sleep(33);
                        }
                        vid_capture.release(); // 釋放資源

                        canvas.setTo(cv::Scalar(0, 0, 0));
                    }, vid_capture).detach();
                sceneStatus = 3;
                updateStatusThread();
            }
        }
        else if (sceneStatus == 3) {
            cameraFrame = RunScene2(canvas);
            cvui::image(canvas, 0, vertical-200, status);
            cvui::image(canvas, 0, vertical-120, status2);
        }
        else if (sceneStatus ==4) {

            cvui::image(canvas, 0, 0, cameraFrame);
        }

        cvui::update();
        cvui::imshow(windowName,canvas);
        if (waitKey(1) == 27) {
            isRunning = false;
            Sleep(1000);
            break; }
    }

    Release();
}

LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
    switch (uMsg)
    {
    case WM_COMMAND:
        if (LOWORD(wParam) == 1)
        {
            // 取得 Text Box 控制項中的文字
            TCHAR buffer[1024];
            GetWindowText(GetDlgItem(hwnd, 0), buffer, sizeof(buffer) / sizeof(buffer[0]));

            // 輸出到控制台
            OutputDebugString(buffer);
            OutputDebugString(L"\n");
            std::wstring ws(buffer);
            std::string name(ws.begin(), ws.end());

            if (name.size() > 0) {

                bool result = cv::imwrite(folder_path + std::to_string(ids.size()) + "_" + name + "_.jpg", cameraFrame);

                // 檢查是否儲存成功
                if (!result)
                {
                    std::cerr << "Failed to save image!" << std::endl;
                }

                ShowWindow(hwnd, SW_HIDE);
                UpdateWindow(hwnd);

                sceneStatus = 0;
            }
        }
        else if (LOWORD(wParam) == 2)
        {
            OutputDebugString(L"cancel");
            OutputDebugString(L"\n");
            ShowWindow(hwnd, SW_HIDE);
            UpdateWindow(hwnd);
            sceneStatus = 0;
        }
        break;
    case WM_DESTROY:
        PostQuitMessage(0);
        return 0;

    default:
        return DefWindowProc(hwnd, uMsg, wParam, lParam);
    }
}
