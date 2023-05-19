#include "FaceRecognizerUtils.h"

FaceRecognizerUtils::FaceRecognizerUtils() {

    // Initialize FaceRecognizerSF
    faceRecognizer = FaceRecognizerSF::create(FLAGS_m_fr, "");
}

FaceRecognizerUtils::~FaceRecognizerUtils() {};

void FaceRecognizerUtils::loadDB(cv::Size reSize, gaze_estimation::FaceDetector* faceDetector) {

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

bool FaceRecognizerUtils::saveToDB(std::string name, cv::Mat cameraFrame) {
    return cv::imwrite(folder_path + std::to_string(ids.size()) + "_" + name + "_.jpg", cameraFrame);
}

cv::Mat FaceRecognizerUtils::recongnizer(cv::Mat frame, gaze_estimation::FaceDetector* faceDetector) {

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
        for (int j = 0; j < features.size(); j++) {

            Mat feature = features[j];
            double cos_score = faceRecognizer->match(feature, feature_target, FaceRecognizerSF::DisType::FR_COSINE);
            double L2_score = faceRecognizer->match(feature, feature_target, FaceRecognizerSF::DisType::FR_NORM_L2);

            if (cos_score >= cosine_similar_thresh && L2_score <= l2norm_similar_thresh) {
                cv::putText(frame, "ID: " + ids[j] + " Name: " + names[j], cv::Point(inferenceResult.faceBoundingBox.x, inferenceResult.faceBoundingBox.y - 20), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 0, 0), 2);
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
    isMember = false;
    indexId = 0;
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

    return frame;
}

std::string FaceRecognizerUtils::getCurrentMemberName() {

    return names[indexId];
}