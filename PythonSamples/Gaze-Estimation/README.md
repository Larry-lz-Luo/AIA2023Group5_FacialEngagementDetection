# Refrence 

This project is based on https://github.com/Dewalade1/computer-pointer-controller

## How it Works

The application takes the input from a webcam or a video file. It detects faces in each frame, finds the eyes in the faces, estimates the head rotation angle, then estimate the gaze orientation.

To do this, the project uses the InferenceEngine API from Intel's OpenVINO Toolkit. The [Gaze Estimation](https://docs.openvinotoolkit.org/latest/_models_intel_gaze_estimation_adas_0002_description_gaze_estimation_adas_0002.html) model needs three inputs:

* The face detect
* The head pose
* The left eye image
* The right eye image
* The gaze estimation

So, the application uses three other OpenVINO models to get these inputs:

* [Face Detection](https://docs.openvino.ai/latest/omz_models_model_face_detection_adas_0001.html).
* [Head Pose Estimation](https://docs.openvino.ai/latest/omz_models_model_head_pose_estimation_adas_0001.html)
* [Facial Landmarks Detection](https://docs.openvino.ai/latest/omz_models_model_facial_landmarks_35_adas_0002.html)
* [Open Closed eye](https://docs.openvino.ai/latest/omz_models_model_open_closed_eye_0001.html)
* [Gaze estimation](https://docs.openvino.ai/latest/omz_models_model_gaze_estimation_adas_0002.html)

## Project Set Up and Installation

### Requirements

#### Hardware

- 6th to 10th generation Intel® Core™ processor with Iris® Pro graphics or Intel® HD Graphics.
- OR use of Intel® Neural Compute Stick 2 (NCS2)
- Webcam (optional)

#### Software

* Intel® Distribution of OpenVINO™ toolkit 2022.3 release
* Python 3.9
* OpenCV 4.5.5
* pyautogui 0.9.48

### Set up development environment

##### 1. Install OpenVINO Toolkit

If you haven't already, download and install the OpenVINO Toolkit. Follow the OpenVINO's get started guide [here](https://software.intel.com/content/www/us/en/develop/tools/openvino-toolkit/get-started.html).

##### 2. Install dependancies

The best way is to use [Anaconda](https://docs.conda.io/en/latest/miniconda.html):

```
conda env create -f environment.yml
```

### Download the models

Use OpenVINO's `model downloader` to download the models. From the main directory, run:

```
#Windows
omz_downloader.exe --list .\model.lst

mo --input_model <PATH_TO_MODEL>open-closed-eye.onnx --mean_values [127.0,127.0,127.0] --scale_values [255,255,255] --output 19
```

### Directory Structure

Find the source code for this project under the `/src` directory.

```
.
├── bin
├── model.lst
├── models
├── images
├── Benchmarks.ipynb
├── README.md
├── environment.yml
└── src
    ├── face_detection.py
    ├── facial_landmarks_detection.py
    ├── gaze_estimation.py
    ├── head_pose_estimation.py
    ├── inference.py
    ├── input_feeder.py
    ├── main.py
    └── mouse_controller.py
```

## Demo

Run the application with the demo video file. From the main directory:
To run it with a webcam:

```
# Windows
python main.py -mfd "models/intel/face-detection-adas-0001/FP32/face-detection-adas-0001" -mhpe "models/intel/head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001" -mfld "models/intel/facial-landmarks-35-adas-0002/FP32/facial-landmarks-35-adas-0002" -mge "models/intel/gaze-estimation-adas-0002/FP32/gaze-estimation-adas-0002" -m_es "models/public/open-closed-eye-0001/open-closed-eye" -o "results" -it "cam"
```

Find the result video under the `\results` directory. 

## Command Line Arguments

```
usage: main.py [-h] -mfd MODEL_FACE_DETECTION -mhpe MODEL_HEAD_POSE_ESTIMATION
               -mfld MODEL_FACIAL_LANDMARKS_DETECTION -mge
               MODEL_GAZE_ESTIMATION -it INPUT_TYPE [-i INPUT_PATH]
               [-o OUTPUT_PATH] [-l CPU_EXTENSION] [-d DEVICE] [-r]
               [--show_input] [--move_mouse]

optional arguments:
  -h, --help            show this help message and exit
  -mfd MODEL_FACE_DETECTION, --model_face_detection MODEL_FACE_DETECTION
                        Path to an xml file with a trained face detection
                        model.
  -mhpe MODEL_HEAD_POSE_ESTIMATION, --model_head_pose_estimation MODEL_HEAD_POSE_ESTIMATION
                        Path to an xml file with a trained head pose
                        estimation model.
  -mfld MODEL_FACIAL_LANDMARKS_DETECTION, --model_facial_landmarks_detection MODEL_FACIAL_LANDMARKS_DETECTION
                        Path to an xml file with a trained facial landmarks
                        detection model.
  -mge MODEL_GAZE_ESTIMATION, --model_gaze_estimation MODEL_GAZE_ESTIMATION
                        Path to an xml file with a trained gaze estimation
                        model.
  -m_es MODEL_EYE_STATE_ESTIMATION, --model_eye_state_estimation MODEL_EYE_STATE_ESTIMATION, 
                        Path to an xml file with a trained open close eye 
                        model.
  -it INPUT_TYPE, --input_type INPUT_TYPE
                        Specify 'video', 'image' or 'cam' (to work with
                        camera).
  -i INPUT_PATH, --input_path INPUT_PATH
                        Path to image or video file.
  -o OUTPUT_PATH, --output_path OUTPUT_PATH
                        Path to image or video file.
  -l CPU_EXTENSION, --cpu_extension CPU_EXTENSION
                        MKLDNN (CPU)-targeted custom layers.Absolute path to a
                        shared library with thekernels impl.
  -d DEVICE, --device DEVICE
                        Specify the target device to infer on: CPU, GPU, FPGA
                        or MYRIAD is acceptable. Sample will look for a
                        suitable plugin for device specified (CPU by default)
  -r, --raw_output_message
                        Optional. Output inference results raw values showing
  --show_input          Optional. Input video showing
  --move_mouse          Optional. Move mouse based on gaze estimation


```
