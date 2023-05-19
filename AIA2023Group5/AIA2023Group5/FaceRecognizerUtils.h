#pragma once
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/objdetect/face.hpp>
#include <gazeEstimation/face_detector.hpp>

using namespace cv;
class FaceRecognizerUtils
{
public:
	FaceRecognizerUtils();
    ~FaceRecognizerUtils();

    bool isMember = false;

    void loadDB(cv::Size reSize, gaze_estimation::FaceDetector* faceDetector);

    bool saveToDB(std::string name, cv::Mat cameraFrame);
    
    cv::Mat recongnizer(cv::Mat frame, gaze_estimation::FaceDetector* faceDetector);

    std::string getCurrentMemberName();

private:
	Ptr<FaceRecognizerSF> faceRecognizer;
	double cosine_similar_thresh = 0.45;// 0.363;
	double l2norm_similar_thresh = 0.98;// 1.128;

	std::string FLAGS_m_fr = "..\\faceDB\\face_recognition_sface_2021dec_int8.onnx";

	std::string folder_path = "..\\faceDB\\";
	std::vector<std::string> ids;
    std::vector<std::string> names;
	std::vector<Mat> features;
    int indexId = 0;

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
};

