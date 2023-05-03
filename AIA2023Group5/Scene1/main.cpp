# define _SILENCE_EXPERIMENTAL_FILESYSTEM_DEPRECATION_WARNING

#include <filesystem>
#include <experimental/filesystem>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>


#include <stdlib.h>
#include <chrono>

#include "openvino/openvino.hpp"

//#include "gflags/gflags.h"
//#include "monitors/presenter.h"
#include "utils/args_helper.hpp"
#include "utils/images_capture.h"
#include "utils/ocv_common.hpp"
#include "utils/slog.hpp"
#include "cnn.hpp"
#include "actions.hpp"
#include "action_detector.hpp"
#include "detector.hpp"
#include "face_reid.hpp"
#include "tracker.hpp"
#include "logger.hpp"

using namespace cv;
using namespace std;

#define POC_WIDTH 640
#define POC_HEIGHT 480

// face detect setting
string FLAGS_d_fd = "CPU";
float FLAGS_t_fd = 0.6;
int FLAGS_inh_fd = POC_HEIGHT;
int FLAGS_inw_fd = POC_WIDTH;
float FLAGS_exp_r_fd = 1.15;
bool showFaceBoundingBox = true;

string FLAGS_m_fd = "..\\intel\\face-detection-adas-0001\\FP32\\face-detection-adas-0001.xml";
const auto fd_model_path = FLAGS_m_fd;

string FLAGS_m_fr = "..\\intel\\face-reidentification-retail-0095\\FP32\\face-reidentification-retail-0095.xml";
const auto fr_model_path = FLAGS_m_fr;

string FLAGS_m_lm = "..\\intel\\landmarks-regression-retail-0009\\FP32\\landmarks-regression-retail-0009.xml";
const auto lm_model_path = FLAGS_m_lm;

// face landmark setting
string FLAGS_d_lm = "GPU";

// face re setting 
string FLAGS_d_reid = "GPU";
string FLAGS_fg = "..\\faceDB\\faces_gallery.json"; // gallery path
//string FLAGS_fg = "faces_gallery.json"; // gallery path
//const auto fg_path = "C:\\Users\\qx50\\Documents\\_AI\\intel zoo\\open_model_zoo\\demos\\build\\intel64\\Release\\utils\\faces_gallery.json";
//string FLAGS_fg = fg_path;
float FLAGS_t_reg_fd = 0.6;
float FLAGS_t_reid = 0.8;
int FLAGS_min_size_fr = 0;
bool FLAGS_greedy_reid_matching = true; //use greedy matching?
bool FLAGS_crop_gallery = false; // crop gallery while matching
#pragma region FaceRecognizer
class FaceRecognizer {
public:
    virtual ~FaceRecognizer() = default;

    virtual bool LabelExists(const std::string& label) const = 0;
    virtual std::string GetLabelByID(int id) const = 0;
    virtual std::vector<std::string> GetIDToLabelMap() const = 0;

    virtual std::vector<int> Recognize(const cv::Mat& frame, const detection::DetectedObjects& faces) = 0;
};

class FaceRecognizerNull : public FaceRecognizer {
public:
    bool LabelExists(const std::string&) const override { return false; }

    std::string GetLabelByID(int) const override {
        return EmbeddingsGallery::unknown_label;
    }

    std::vector<std::string> GetIDToLabelMap() const override { return {}; }

    std::vector<int> Recognize(const cv::Mat&, const detection::DetectedObjects& faces) override {
        return std::vector<int>(faces.size(), EmbeddingsGallery::unknown_id);
    }
};

class FaceRecognizerDefault : public FaceRecognizer {
public:
    FaceRecognizerDefault(
        const CnnConfig& landmarks_detector_config,
        const CnnConfig& reid_config,
        const detection::DetectorConfig& face_registration_det_config,
        const std::string& face_gallery_path,
        double reid_threshold,
        int min_size_fr,
        bool crop_gallery,
        bool greedy_reid_matching) :
        landmarks_detector(landmarks_detector_config),
        face_reid(reid_config),
        face_gallery(face_gallery_path, reid_threshold, min_size_fr, crop_gallery,
            face_registration_det_config, landmarks_detector, face_reid,
            greedy_reid_matching)
    {
        if (face_gallery.size() == 0) {
            slog::warn << "Face reid gallery is empty!" << slog::endl;
        }
        else {
            slog::info << "Face reid gallery size: " << face_gallery.size() << slog::endl;
        }
    }

    bool LabelExists(const std::string& label) const override {
        return face_gallery.LabelExists(label);
    }

    std::string GetLabelByID(int id) const override {
        return face_gallery.GetLabelByID(id);
    }

    std::vector<std::string> GetIDToLabelMap() const override {
        return face_gallery.GetIDToLabelMap();
    }

    std::vector<int> Recognize(const cv::Mat& frame, const detection::DetectedObjects& faces) override {
        const int maxLandmarksBatch = landmarks_detector.maxBatchSize();
        int numFaces = (int)faces.size();

        std::vector<cv::Mat> landmarks;
        std::vector<cv::Mat> embeddings;
        std::vector<cv::Mat> face_rois;

        auto face_roi = [&](const detection::DetectedObject& face) {
            return frame(face.rect);
        };
        if (numFaces < maxLandmarksBatch) {
            std::transform(faces.begin(), faces.end(), std::back_inserter(face_rois), face_roi);
            landmarks_detector.Compute(face_rois, &landmarks, cv::Size(2, 5));
            AlignFaces(&face_rois, &landmarks);
            face_reid.Compute(face_rois, &embeddings);
        }
        else {
            auto embedding = [&](cv::Mat& emb) { return emb; };
            for (int n = numFaces; n > 0; n -= maxLandmarksBatch) {
                landmarks.clear();
                face_rois.clear();
                size_t start_idx = size_t(numFaces) - n;
                size_t end_idx = start_idx + std::min(numFaces, maxLandmarksBatch);
                std::transform(faces.begin() + start_idx, faces.begin() + end_idx, std::back_inserter(face_rois), face_roi);

                landmarks_detector.Compute(face_rois, &landmarks, cv::Size(2, 5));

                AlignFaces(&face_rois, &landmarks);

                std::vector<cv::Mat> batch_embeddings;
                face_reid.Compute(face_rois, &batch_embeddings);
                std::transform(batch_embeddings.begin(), batch_embeddings.end(), std::back_inserter(embeddings), embedding);
            }
        }

        return face_gallery.GetIDsByEmbeddings(embeddings);
    }

private:
    VectorCNN landmarks_detector;
    VectorCNN face_reid;
    EmbeddingsGallery face_gallery;
};

#pragma endregion


// face reg

int main()
{
    ov::Core core;
   

    // init face detect
    std::unique_ptr<AsyncDetection<detection::DetectedObject>> face_detector;
    if (!fd_model_path.empty()) {
        // Load face detector
        detection::DetectorConfig face_config(fd_model_path);
        face_config.m_deviceName = FLAGS_d_fd;
        face_config.m_core = core;
        face_config.is_async = true;
        face_config.confidence_threshold = static_cast<float>(FLAGS_t_fd);
        face_config.input_h = FLAGS_inh_fd;
        face_config.input_w = FLAGS_inw_fd;
        face_config.increase_scale_x = static_cast<float>(FLAGS_exp_r_fd);
        face_config.increase_scale_y = static_cast<float>(FLAGS_exp_r_fd);
        face_detector.reset(new detection::FaceDetection(face_config));
    }
    else {
        face_detector.reset(new NullDetection<detection::DetectedObject>);
    }


    // init face recognization
    std::unique_ptr<FaceRecognizer> face_recognizer;

    if (!fd_model_path.empty() && !fr_model_path.empty() && !lm_model_path.empty()) {
        // Create face recognizer
        detection::DetectorConfig face_registration_det_config(fd_model_path);
        face_registration_det_config.m_deviceName = FLAGS_d_fd;
        face_registration_det_config.m_core = core;
        face_registration_det_config.is_async = false;
        face_registration_det_config.confidence_threshold = static_cast<float>(FLAGS_t_reg_fd);
        face_registration_det_config.increase_scale_x = static_cast<float>(FLAGS_exp_r_fd);
        face_registration_det_config.increase_scale_y = static_cast<float>(FLAGS_exp_r_fd);

        CnnConfig reid_config(fr_model_path, "Face Re-Identification");
        reid_config.m_deviceName = FLAGS_d_reid;
        reid_config.m_max_batch_size = 16;
        reid_config.m_core = core;

        CnnConfig landmarks_config(lm_model_path, "Facial Landmarks Regression");
        landmarks_config.m_deviceName = FLAGS_d_lm;
        landmarks_config.m_max_batch_size = 16;
        landmarks_config.m_core = core;
        face_recognizer.reset(new FaceRecognizerDefault(
            landmarks_config, reid_config,
            face_registration_det_config,
            FLAGS_fg, FLAGS_t_reid, FLAGS_min_size_fr, FLAGS_crop_gallery, FLAGS_greedy_reid_matching));

        
    }
    else {
        slog::warn << "Face Recognition models are disabled!" << slog::endl;

        face_recognizer.reset(new FaceRecognizerNull);
    }

    // Create tracker for reid
    TrackerParams tracker_reid_params;
    tracker_reid_params.min_track_duration = 1;
    tracker_reid_params.forget_delay = 150;
    tracker_reid_params.affinity_thr = 0.8f;
    tracker_reid_params.averaging_window_size_for_rects = 1;
    tracker_reid_params.averaging_window_size_for_labels = std::numeric_limits<int>::max();
    tracker_reid_params.bbox_heights_range = cv::Vec2f(10, 1080);
    tracker_reid_params.drop_forgotten_tracks = false;
    tracker_reid_params.max_num_objects_in_track = std::numeric_limits<int>::max();
    tracker_reid_params.objects_type = "face";

    // opencv also has cv::Tracker
    ::Tracker tracker_reid(tracker_reid_params);




    size_t work_num_frames = 0;
    // opencv start capture 
    cv::VideoCapture cap(0, cv::CAP_DSHOW);
    cap.set(cv::CAP_PROP_FRAME_WIDTH, POC_WIDTH);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, POC_HEIGHT);
    if (!cap.isOpened()) {
        std::cerr << "Failed to open camera" << std::endl;
        return 1;
    }
    cv::Mat frame;
    bool is_last_frame = false;
    while (cap.read(frame)) {
        cv::Mat prev_frame = std::move(frame);
        cap.read(frame);
        if (frame.data && frame.size() != prev_frame.size()) {
            throw std::runtime_error("Can't track objects on images of different size");
        }
        is_last_frame = !frame.data;
        if (!is_last_frame) {
            face_detector->enqueue(frame);
            face_detector->submitRequest();
        }
        face_detector->wait();
        detection::DetectedObjects faces = face_detector->fetchResults();
        for (size_t i = 0; i < faces.size(); i++) {
            try {
                auto ids = face_recognizer->Recognize(prev_frame, faces);
                TrackedObjects tracked_face_objects;

                for (size_t i = 0; i < faces.size(); i++) {
                    tracked_face_objects.emplace_back(faces[i].rect, faces[i].confidence, ids[i]);
                }
                tracker_reid.Process(prev_frame, tracked_face_objects, work_num_frames);
                const auto tracked_faces = tracker_reid.TrackedDetectionsWithLabels();
                
                std::string face_label = "unknown";
                std::map<int, int> frame_face_obj_id_to_action;
                for (size_t j = 0; j < tracked_faces.size(); j++) {
                    const auto& face = tracked_faces[j];
                    face_label = face_recognizer->GetLabelByID(face.label);

                    std::string label_to_draw;
                    if (face.label != EmbeddingsGallery::unknown_id)
                        label_to_draw += face_label;
                    //slog::info << "face_label get: " << face_label << slog::endl;
                
                }

                if (showFaceBoundingBox) { // 
                    try
                    {
                        const cv::Scalar text_color(255, 255, 255);
                        cv::rectangle(frame, faces[i].rect, cv::Scalar::all(255), 1);
                        int baseLine = 0;
                        const cv::Size label_size = cv::getTextSize(face_label, cv::FONT_HERSHEY_PLAIN, 1, 1, &baseLine);
                        cv::putText(frame, face_label, cv::Point(faces[i].rect.x, faces[i].rect.y), cv::FONT_HERSHEY_PLAIN, 1,
                            text_color, 1, cv::LINE_AA);
                        /*auto scale = 0.002 * faces[i].rect.width;
                        putHighlightedText(frame,
                            cv::format("Detector confidence: %0.2f",
                                static_cast<double>(faces[i].confidence)),
                            cv::Point(static_cast<int>(faces[i].rect.x),
                                static_cast<int>(faces[i].rect.y)),
                            cv::FONT_HERSHEY_COMPLEX, scale, cv::Scalar(200, 10, 10), 1);*/
                    }
                    catch (...) {}
                }
            }
            catch (...) {}
            //tracked_face_objects.emplace_back(faces[i].rect, faces[i].confidence, ids[i]);
            //printf("face detect: %d/n" , faces.size());
            auto scale = 0.002 * faces[i].rect.width;

            ++work_num_frames;
        }
        cv::imshow("Camera", frame);
        cv::waitKey(1);
    }
    return 0;
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
