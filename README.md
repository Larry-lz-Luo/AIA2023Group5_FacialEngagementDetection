# AIA2023_Facial_Recognition_POC

目前有bug, 眼睛跑出CAM 會crash

test script

python src\main.py -mfd models\intel\face-detection-adas-0001\FP32\face-detection-adas-0001 
-mhpe models\intel\head-pose-estimation-adas-0001\FP32\head-pose-estimation-adas-0001
 -mfld models\intel\facial-landmarks-35-adas-0002\FP32\facial-landmarks-35-adas-0002
 -mge models\intel\gaze-estimation-adas-0002\FP32\gaze-estimation-adas-0002 -o results -it cam