#pragma once
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


#include <xgboost/c_api.h>

class GazeUtils
{
public:
	GazeUtils();
	~GazeUtils();

    FaceDetector* faceDetector;

    cv::Mat checkConcentrated(cv::Mat frame);

    std::vector<float> getFaceInferenceData(FaceInferenceResults inferenceResult);

    void checkGazeWithXGBooster(FaceInferenceResults inferenceResult);

    void checkGazeWithAngles(FaceInferenceResults inferenceResult);

    bool getResultWithAngles();

    bool getResultWithXGBooster();

private:
    std::string FLAGS_m_fd = "..\\models\\intel\\face-detection-retail-0004\\FP32\\face-detection-retail-0004.xml"
        , FLAGS_d_fd = "GPU"
        , FLAGS_m_hp = "..\\models\\intel\\head-pose-estimation-adas-0001\\FP32\\head-pose-estimation-adas-0001.xml"
        , FLAGS_d_hp = "GPU"
        , FLAGS_m_lm = "..\\models\\intel\\facial-landmarks-35-adas-0002\\FP32\\facial-landmarks-35-adas-0002.xml"
        , FLAGS_d_lm = "GPU"
        , FLAGS_m_es = "..\\models\\public\\open-closed-eye-0001\\FP32\\open-closed-eye-0001.xml"
        , FLAGS_d_es = "GPU"
        , FLAGS_m = "..\\models\\intel\\gaze-estimation-adas-0002\\FP32\\gaze-estimation-adas-0002.xml"
        , FLAGS_d = "GPU"
        , FLAGS_m_fr = "..\\faceDB\\face_recognition_sface_2021dec_int8.onnx";

    //Gaze models
	HeadPoseEstimator* headPoseEstimator;
	LandmarksEstimator* landmarksEstimator;
	EyeStateEstimator* eyeStateEstimator;
	GazeEstimator* gazeEstimator;

    // Put pointers to all estimators in an array so that they could be processed uniformly in a loop
    std::vector< BaseEstimator*> estimators;

    ResultsMarker *resultsMarker=new ResultsMarker(true, true, true, true, true);

    //XGBooster
	BoosterHandle booster;
	std::vector<float> recordStatusWithXGBooster;
	bool resultWithXGBooster = false;
	std::vector<std::vector<float>> recordInferenceResults;

    std::vector<float> recordStatusWithAngles;
    bool resultWithAngles = false;


    std::vector<float> getFaceInferenceData76(FaceInferenceResults inferenceResult);

    std::vector<float> getFaceInferenceDataEDA(FaceInferenceResults inferenceResult);

    std::vector<float> getFaceInferenceDataFI(FaceInferenceResults inferenceResult);
};

