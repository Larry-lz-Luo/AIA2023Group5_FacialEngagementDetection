import os
import cv2
import logging as log
import time
from argparse import ArgumentParser
import numpy as np

from mouse_controller import MouseController
from input_feeder import InputFeeder
from face_detection import FaceDetector
from head_pose_estimation import HeadPoseEstimator
from facial_landmarks_detection import FacialLandmarksDetector
from gaze_estimation import GazeEstimator
from eye_state_estimation import EyeStateEstimator
import pickle

import pandas as pd
import xgboost as xgb

import warnings
warnings.simplefilter("ignore", UserWarning)



ENGAGEMENT_MODEL = 'C:\\Users\\qx50\\Documents\\_AIA\\Gaze openvino\\Gaze-Estimation\\src\\XGB_normalized_top5_model_20230517.json'
#ENGAGEMENT_MODEL = 'C:\\Users\\qx50\\Documents\\_AIA\\Gaze openvino\\Gaze-Estimation\\src\\GB_pickle_model.pkl'

CHOISEED_FEATURE = ['HeadPoseAngles_Y','HeadPoseAngles_Z','GazeVector_X','GazeVector_Y','GazeVector_Z']

def build_argparser():
    """
    Parse command line arguments.
    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-mfd", "--model_face_detection", required=True, type=str,
                        help="Path to an xml file with a trained face detection model.")
    parser.add_argument("-mhpe", "--model_head_pose_estimation", required=True, type=str,
                        help="Path to an xml file with a trained head pose estimation model.")
    parser.add_argument("-mfld", "--model_facial_landmarks_detection", required=True, type=str,
                        help="Path to an xml file with a trained facial landmarks detection model.")
    parser.add_argument("-mge", "--model_gaze_estimation", required=True, type=str,
                        help="Path to an xml file with a trained gaze estimation model.")
    parser.add_argument("-m_es", "--model_eye_state_estimation", required=True, type=str,
                        help="Path to an xml file with a trained eye state estimation model.")
    parser.add_argument("-it", "--input_type", required=True, type=str,
                        help="Specify 'video', 'image' or 'cam' (to work with camera).")
    parser.add_argument("-i", "--input_path", required=False, type=str, default=None,
                        help="Path to image or video file.")
    parser.add_argument("-o", "--output_path", required=False, type=str, default="results",
                        help="Path to image or video file.")                        
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-r", "--raw_output_message", help="Optional. Output inference results raw values showing",
                      default=False, action="store_true")
    parser.add_argument("--show_input", help="Optional. Show input video",
                      default=False, action="store_true")
    parser.add_argument("--move_mouse", help="Optional. Move mouse based on gaze estimation",
                      default=False, action="store_true")
    return parser

def FaceLandmarksPreprocessing(df_data, verbose = False):
    for col_name in df_data.columns.to_list():
        if col_name.find('FaceLandmarks_') == 0:
            ##print(f"_X {col_name.find('_X')} , len(col_name): {len(col_name) - 2}")
            #print(f"_Y {col_name.find('_Y')} , len(col_name): {len(col_name) - 2}")
            #print(len(col_name) - 2)
            if col_name.find('_X') == len(col_name) - 2:
                df_data[col_name] = df_data[col_name] - df_data['FaceBoundingBox_X']
                if verbose:
                    print(f'Column {col_name} is subtracted by FaceBoundingBox_X.')
            elif col_name.find('_Y') == len(col_name) - 2:
                df_data[col_name] = df_data[col_name] - df_data['FaceBoundingBox_Y']
                if verbose:
                    print(f'Column {col_name} is subtracted by FaceBoundingBox_Y.')
            else:
#                 print(f'Ignore column {col_name}.')
                pass
        else:
#             print(f'Ignore column {col_name}')
            pass
    return df_data

def test_custom_model(df_train):
    #df_train = FaceLandmarksPreprocessing(df_train)
    #df_train = df_train.drop(labels = ['FaceBoundingBox_X'], axis = 1) 
    #df_train = df_train.drop(labels = ['FaceBoundingBox_Y'], axis = 1) 

    col_positive = CHOISEED_FEATURE

    df_positive = df_train.loc[:, col_positive]
    model_xgb = xgb.XGBRegressor()
    
    #model_xgb.load_model('C:\\Users\\qx50\\Documents\\_AIA\\Gaze openvino\\Gaze-Estimation\\src\\XGB_model_76features.json')

    model_xgb.load_model(ENGAGEMENT_MODEL)
    npa_test = df_positive.to_numpy()
    #print(npa_test[0])
    pred = model_xgb.predict(npa_test)
    print(f'pred : {pred[0]}')

def test_custom_model_GradientBoosting(df_train):
    #df_train = FaceLandmarksPreprocessing(df_train)
    #df_train = df_train.drop(labels = ['FaceBoundingBox_X'], axis = 1) 
    #df_train = df_train.drop(labels = ['FaceBoundingBox_Y'], axis = 1) 

    col_positive = CHOISEED_FEATURE

    df_positive = df_train.loc[:, col_positive]
    with open(ENGAGEMENT_MODEL, 'rb') as f:
        loaded_gb_model2 = pickle.load(f)
    #loaded_gb_model2 = pickle.load()
    
    #model_xgb.load_model('C:\\Users\\qx50\\Documents\\_AIA\\Gaze openvino\\Gaze-Estimation\\src\\XGB_model_76features.json')

    #model_xgb.load_model(ENGAGEMENT_MODEL)
    npa_test = df_positive.to_numpy()
    #print(npa_test[0])
    pred = loaded_gb_model2.predict(npa_test)
    print(f'pred : {pred[0]}')


def infer_on_stream(args):
    try:
        log.basicConfig(
            level=log.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                log.FileHandler("app.log"),
                log.StreamHandler()
            ])
            
        mouse_controller = MouseController(precision="low", speed="fast")

        start_model_load_time=time.time()

        face_detector = FaceDetector(args.model_face_detection)
        facial_landmarks_detector = FacialLandmarksDetector(args.model_facial_landmarks_detection)
        head_pose_estimator = HeadPoseEstimator(args.model_head_pose_estimation)
        gaze_estimator = GazeEstimator(args.model_gaze_estimation)

        eye_state_estimator = EyeStateEstimator(args.model_eye_state_estimation)
        face_detector.load_model()
        facial_landmarks_detector.load_model()
        head_pose_estimator.load_model()
        gaze_estimator.load_model()

        eye_state_estimator.load_model()

        total_model_load_time = time.time() - start_model_load_time
        log.info("Model load time: {:.1f}ms".format(1000 * total_model_load_time))

        output_directory = os.path.join(args.output_path + '\\' + args.device)
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        feed = InputFeeder(args.input_type, args.input_path)
        feed.load_data()
        out_video = feed.get_out_video(output_directory)

        frame_counter = 0
        start_inference_time=time.time()
        total_prepocess_time = 0

        
        while True:
            try:
                frame = next(feed.next_batch())
            except StopIteration:
                break
            frame_counter += 1

            face_boxes = face_detector.predict(frame)
            for face_box in face_boxes:
                face_image = get_crop_image(frame, face_box)
                if not face_image.size == 0: 
                    eye_boxes, eye_centers,normalized_landmarks = facial_landmarks_detector.predict(face_image)
                    left_eye_image, right_eye_image = [get_crop_image(face_image, eye_box) for eye_box in eye_boxes]
                    head_pose_angles = head_pose_estimator.predict(face_image)
                    gaze_x, gaze_y, gaze_z = gaze_estimator.predict(right_eye_image, head_pose_angles, left_eye_image)
                    left_eye_state = eye_state_estimator.predict(left_eye_image, head_pose_angles)
                    right_eye_state = eye_state_estimator.predict(right_eye_image, head_pose_angles)
                    #print(f'{left_eye_state} , {right_eye_state}')
                    if args.show_input:
                        cv2.imshow('im', frame)
                    if args.move_mouse:
                        mouse_controller.move(gaze_x, gaze_y)
                    total_prepocess_time += face_detector.preprocess_time + facial_landmarks_detector.preprocess_time + \
                        head_pose_estimator.preprocess_time + gaze_estimator.preprocess_time
                    break

            df_train= pd.DataFrame(
                {
                    'FaceBoundingBox_X': face_box[0],
                    'FaceBoundingBox_Y': face_box[1],
                    'FaceLandmarks_1_X': normalized_landmarks[0][0],
                    'FaceLandmarks_1_Y': normalized_landmarks[0][1],
                    'FaceLandmarks_2_X': normalized_landmarks[1][0],
                    'FaceLandmarks_2_Y': normalized_landmarks[1][1],
                    'FaceLandmarks_3_X': normalized_landmarks[2][0],
                    'FaceLandmarks_3_Y': normalized_landmarks[2][1],
                    'FaceLandmarks_4_X': normalized_landmarks[3][0],
                    'FaceLandmarks_4_Y': normalized_landmarks[3][1],
                    'FaceLandmarks_5_X': normalized_landmarks[4][0],
                    'FaceLandmarks_5_Y': normalized_landmarks[4][1],
                    'FaceLandmarks_6_X': normalized_landmarks[5][0],
                    'FaceLandmarks_6_Y': normalized_landmarks[5][1],
                    'FaceLandmarks_7_X': normalized_landmarks[6][0],
                    'FaceLandmarks_7_Y': normalized_landmarks[6][1],
                    'FaceLandmarks_8_X': normalized_landmarks[7][0],
                    'FaceLandmarks_8_Y': normalized_landmarks[7][1],
                    'FaceLandmarks_9_X': normalized_landmarks[8][0],
                    'FaceLandmarks_9_Y': normalized_landmarks[8][1],
                    'FaceLandmarks_10_X': normalized_landmarks[9][0],
                    'FaceLandmarks_10_Y': normalized_landmarks[9][1],
                    
                    'FaceLandmarks_11_X': normalized_landmarks[10][0],
                    'FaceLandmarks_11_Y': normalized_landmarks[10][1],
                    'FaceLandmarks_12_X': normalized_landmarks[11][0],
                    'FaceLandmarks_12_Y': normalized_landmarks[11][1],
                    'FaceLandmarks_13_X': normalized_landmarks[12][0],
                    'FaceLandmarks_13_Y': normalized_landmarks[12][1],
                    'FaceLandmarks_14_X': normalized_landmarks[13][0],
                    'FaceLandmarks_14_Y': normalized_landmarks[13][1],
                    'FaceLandmarks_15_X': normalized_landmarks[14][0],
                    'FaceLandmarks_15_Y': normalized_landmarks[14][1],
                    'FaceLandmarks_16_X': normalized_landmarks[15][0],
                    'FaceLandmarks_16_Y': normalized_landmarks[15][1],
                    'FaceLandmarks_17_X': normalized_landmarks[16][0],
                    'FaceLandmarks_17_Y': normalized_landmarks[16][1],
                    'FaceLandmarks_18_X': normalized_landmarks[17][0],
                    'FaceLandmarks_18_Y': normalized_landmarks[17][1],
                    'FaceLandmarks_19_X': normalized_landmarks[18][0],
                    'FaceLandmarks_19_Y': normalized_landmarks[18][1],
                    'FaceLandmarks_20_X': normalized_landmarks[19][0],
                    'FaceLandmarks_20_Y': normalized_landmarks[19][1],

                    'FaceLandmarks_21_X': normalized_landmarks[20][0],
                    'FaceLandmarks_21_Y': normalized_landmarks[20][1],
                    'FaceLandmarks_22_X': normalized_landmarks[21][0],
                    'FaceLandmarks_22_Y': normalized_landmarks[21][1],
                    'FaceLandmarks_23_X': normalized_landmarks[22][0],
                    'FaceLandmarks_23_Y': normalized_landmarks[22][1],
                    'FaceLandmarks_24_X': normalized_landmarks[23][0],
                    'FaceLandmarks_24_Y': normalized_landmarks[23][1],
                    'FaceLandmarks_25_X': normalized_landmarks[24][0],
                    'FaceLandmarks_25_Y': normalized_landmarks[24][1],
                    'FaceLandmarks_26_X': normalized_landmarks[25][0],
                    'FaceLandmarks_26_Y': normalized_landmarks[25][1],
                    'FaceLandmarks_27_X': normalized_landmarks[26][0],
                    'FaceLandmarks_27_Y': normalized_landmarks[26][1],
                    'FaceLandmarks_28_X': normalized_landmarks[27][0],
                    'FaceLandmarks_28_Y': normalized_landmarks[27][1],
                    'FaceLandmarks_29_X': normalized_landmarks[28][0],
                    'FaceLandmarks_29_Y': normalized_landmarks[28][1],
                    'FaceLandmarks_30_X': normalized_landmarks[29][0],
                    'FaceLandmarks_30_Y': normalized_landmarks[29][1],

                    'FaceLandmarks_31_X': normalized_landmarks[30][0],
                    'FaceLandmarks_31_Y': normalized_landmarks[30][1],
                    'FaceLandmarks_32_X': normalized_landmarks[31][0],
                    'FaceLandmarks_32_Y': normalized_landmarks[31][1],
                    'FaceLandmarks_33_X': normalized_landmarks[32][0],
                    'FaceLandmarks_33_Y': normalized_landmarks[32][1],
                    'FaceLandmarks_34_X': normalized_landmarks[33][0],
                    'FaceLandmarks_34_Y': normalized_landmarks[33][1],
                    'FaceLandmarks_35_X': normalized_landmarks[34][0],
                    'FaceLandmarks_35_Y': normalized_landmarks[34][1],
                    #'EyeState_Left': left_eye_state,
                    #'EyeState_Right': right_eye_state,
                    #'LeftEyeBoundingBox_Y': eye_boxes[0][1],
                    #'RightEyeBoundingBox_X': eye_boxes[1][0],
                    #'RightEyeBoundingBox_Y': eye_boxes[1][1],
                    #'EyeLandmarks_2_Y': normalized_landmarks[1][1],
                    #'LeftEyeMidPoint_Y': eye_centers[0][1],
                    
                    'HeadPoseAngles_X': head_pose_angles[0],
                    'HeadPoseAngles_Y': head_pose_angles[1],
                    'HeadPoseAngles_Z': head_pose_angles[2],

                    'GazeVector_X': gaze_x,
                    'GazeVector_Y': gaze_y,
                    'GazeVector_Z': gaze_z
                 })

            ##for i in range(35):
            #    df_train['FaceLandmarks_' + str(i +1) +'_X'] += int(df_train['FaceLandmarks_' + str(i +1) +'_X'])
            #    df_train['FaceLandmarks_' + str(i +1) +'_Y'] += int(df_train['FaceLandmarks_' + str(i +1) +'_Y'])
                

            test_custom_model(df_train)
            #test_custom_model_GradientBoosting(df_train)

            #df_train 
            if out_video is not None:
                #out_video.write(frame)
                cv2.imshow("DetectionResults", frame)
            if args.input_type == "image":
                cv2.imwrite(os.path.join(output_directory, 'output_image.jpg'), frame)

            key_pressed = cv2.waitKey(60)
            if key_pressed == 27:
                break
        
        total_time=time.time()-start_inference_time
        total_inference_time=round(total_time, 1)
        fps=frame_counter/total_inference_time
        log.info("Inference time:{:.1f}ms".format(1000* total_inference_time))
        log.info("Input/output preprocess time:{:.1f}ms".format(1000* total_prepocess_time))
        log.info("FPS:{}".format(fps))
        #print("FPS:{}".format(fps))

        with open(os.path.join(output_directory, 'stats.txt'), 'w') as f:
            f.write(str(total_inference_time)+'\n')
            f.write(str(total_prepocess_time)+'\n')
            f.write(str(fps)+'\n')
            f.write(str(total_model_load_time)+'\n')
            
        feed.close()
        cv2.destroyAllWindows()
    except Exception as e:
        log.exception("Something wrong when running inference:" + str(e))

def get_crop_image(image, box):
    xmin, ymin, xmax, ymax = box
    crop_image = image[ymin:ymax, xmin:xmax]
    return crop_image

def main():
    args = build_argparser().parse_args()
    infer_on_stream(args)

if __name__ == '__main__':
    
    
    main()