#include "GazeUtils.h"


GazeUtils::GazeUtils() {

    // Load OpenVINO runtime
    slog::info << ov::get_openvino_version() << slog::endl;

    ov::Core core;

    // Set up face detector and estimators
    faceDetector = new FaceDetector(core, FLAGS_m_fd, FLAGS_d_fd, 0.5, false);
    headPoseEstimator = new HeadPoseEstimator(core, FLAGS_m_hp, FLAGS_d_hp);
    landmarksEstimator = new LandmarksEstimator(core, FLAGS_m_lm, FLAGS_d_lm);
    eyeStateEstimator = new EyeStateEstimator(core, FLAGS_m_es, FLAGS_d_es);
    gazeEstimator = new GazeEstimator(core, FLAGS_m, FLAGS_d);
    estimators.push_back(headPoseEstimator);
    estimators.push_back(landmarksEstimator);
    estimators.push_back(eyeStateEstimator);
    estimators.push_back(gazeEstimator);

    int res = XGBoosterCreate(NULL, 0, &booster);
    std::cout << "XGBoosterCreate: " << res << "\n";
    res = XGBoosterLoadModel(booster,
        //"..\\models\\XGB_normalized_top5_model.json"
        "..\\models\\XGB_normalized_top5_model_20230517.json"
    );

    std::cout << "XGBoosterLoadModel: " << res << "\n";

}

GazeUtils::~GazeUtils() {

    if (faceDetector)delete faceDetector;

    if (headPoseEstimator)delete headPoseEstimator;
    if (landmarksEstimator)delete  landmarksEstimator;
    if (eyeStateEstimator)delete eyeStateEstimator;
    if (gazeEstimator)delete gazeEstimator;

    estimators.clear();

    XGBoosterFree(booster);
}

cv::Mat GazeUtils::checkConcentrated(cv::Mat frame) {

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

        if (maxFace >= 0) {
            auto const& inferenceResult = inferenceResults[maxFace];
            if(showResultsMarker)resultsMarker->mark(frame, inferenceResult);
            checkGazeWithAngles(inferenceResult);
            checkGazeWithXGBooster(inferenceResult);

        }
    }

    return frame;
}

std::vector<float> GazeUtils::getFaceInferenceData76(FaceInferenceResults inferenceResult) {
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

std::vector<float> GazeUtils::getFaceInferenceDataEDA(FaceInferenceResults inferenceResult) {
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
    result.push_back(inferenceResult.leftEyeBoundingBox.y);
    result.push_back(inferenceResult.rightEyeBoundingBox.x); result.push_back(inferenceResult.rightEyeBoundingBox.y);
    result.push_back(inferenceResult.getEyeLandmarks()[2 - 1].y);
    result.push_back(inferenceResult.leftEyeMidpoint.y);
    result.push_back(inferenceResult.gazeVector.z);

    return result;
}

std::vector<float> GazeUtils::getFaceInferenceDataFI(FaceInferenceResults inferenceResult) {
    std::vector<float> result;

    result.push_back(inferenceResult.headPoseAngles.y);
    result.push_back(inferenceResult.headPoseAngles.z);
    result.push_back(inferenceResult.gazeVector.x);
    result.push_back(inferenceResult.gazeVector.y);
    result.push_back(inferenceResult.gazeVector.z);

    return result;
}

std::vector<float> GazeUtils::getFaceInferenceData(FaceInferenceResults inferenceResult) {
    return
        //getFaceInferenceData6(inferenceResult);
        //getFaceInferenceDataEDA(inferenceResult);
        //getFaceInferenceData76(inferenceResult);
        //getFaceInferenceDataEDA_jasonTest(inferenceResult);
        //getFaceInferenceDataEDA_6pos6neg(inferenceResult);
        getFaceInferenceDataFI(inferenceResult);
}

void GazeUtils::checkGazeWithXGBooster(FaceInferenceResults inferenceResult) {
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
}

void GazeUtils::checkGazeWithAngles(FaceInferenceResults inferenceResult) {

    float gazeH = 100;
    float gazeV = 100;
    cv::Point2f gazeAngles;
    gazeVectorToGazeAngles(inferenceResult.gazeVector, gazeAngles);
    //check gaze
    {

        gazeH = gazeAngles.x;
        gazeV = gazeAngles.y;
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
    }
}

bool GazeUtils::getResultWithAngles() {

    return resultWithAngles;
}

bool GazeUtils::getResultWithXGBooster() {

    return resultWithXGBooster;
}