# Model Creation & Evaluation #

We created some codes for model creation and evaluation. These models are for concentration detection, once input related facial features, the model will output concentration judgement result.

## Model List ##

* Decision Tree
  * scene2_DecisionTree.ipynb
* KNN
  * scene2_KNN.ipynb
* Random Forest
  * scene2_RandomForest.ipynb
* XGBoost
  * scene2_XGBoost.ipynb

## Features Evaluation ##

We evaluate the importance of all features from the input dataset, via figure "Feature importance" after fitting a XGBoost model with all the features. We do it and get the figure from code scene2_XGBoost_Evaluation.ipynb.

In our dataset we have the following features:

<em>RecordTime, FaceBoundingBox_X, FaceBoundingBox_Y, FaceBoundingBox_W, FaceBoundingBox_H, 
FaceLandmarks_1_X, FaceLandmarks_1_Y, FaceLandmarks_2_X, FaceLandmarks_2_Y, FaceLandmarks_3_X, 
FaceLandmarks_3_Y, FaceLandmarks_4_X, FaceLandmarks_4_Y, FaceLandmarks_5_X, FaceLandmarks_5_Y, 
FaceLandmarks_6_X, FaceLandmarks_6_Y, FaceLandmarks_7_X, FaceLandmarks_7_Y, FaceLandmarks_8_X, 
FaceLandmarks_8_Y, FaceLandmarks_9_X, FaceLandmarks_9_Y, FaceLandmarks_10_X, FaceLandmarks_10_Y, 
FaceLandmarks_11_X, FaceLandmarks_11_Y, FaceLandmarks_12_X, FaceLandmarks_12_Y, FaceLandmarks_13_X, 
FaceLandmarks_13_Y, FaceLandmarks_14_X, FaceLandmarks_14_Y, FaceLandmarks_15_X, FaceLandmarks_15_Y, 
FaceLandmarks_16_X, FaceLandmarks_16_Y, FaceLandmarks_17_X, FaceLandmarks_17_Y, FaceLandmarks_18_X, 
FaceLandmarks_18_Y, FaceLandmarks_19_X, FaceLandmarks_19_Y, FaceLandmarks_20_X, FaceLandmarks_20_Y, 
FaceLandmarks_21_X, FaceLandmarks_21_Y, FaceLandmarks_22_X, FaceLandmarks_22_Y, FaceLandmarks_23_X, 
FaceLandmarks_23_Y, FaceLandmarks_24_X, FaceLandmarks_24_Y, FaceLandmarks_25_X, FaceLandmarks_25_Y, 
FaceLandmarks_26_X, FaceLandmarks_26_Y, FaceLandmarks_27_X, FaceLandmarks_27_Y, FaceLandmarks_28_X, 
FaceLandmarks_28_Y, FaceLandmarks_29_X, FaceLandmarks_29_Y, FaceLandmarks_30_X, FaceLandmarks_30_Y, 
FaceLandmarks_31_X, FaceLandmarks_31_Y, FaceLandmarks_32_X, FaceLandmarks_32_Y, FaceLandmarks_33_X, 
FaceLandmarks_33_Y, FaceLandmarks_34_X, FaceLandmarks_34_Y, FaceLandmarks_35_X, FaceLandmarks_35_Y, 
HeadPoseAngles_X, HeadPoseAngles_Y, HeadPoseAngles_Z, EyeState_Left, EyeState_Right, 
LeftEyeBoundingBox_X, LeftEyeBoundingBox_Y, LeftEyeBoundingBox_W, LeftEyeBoundingBox_H, RightEyeBoundingBox_X, 
RightEyeBoundingBox_Y, RightEyeBoundingBox_W, RightEyeBoundingBox_H, EyeLandmarks_1_X, EyeLandmarks_1_Y, 
EyeLandmarks_2_X, EyeLandmarks_2_Y, EyeLandmarks_3_X, EyeLandmarks_3_Y, EyeLandmarks_4_X, 
EyeLandmarks_4_Y, LeftEyeMidPoint_X, LeftEyeMidPoint_Y, RightEyeMidPoint_X, RightEyeMidPoint_Y, 
GazeVector_X, GazeVector_Y, GazeVector_Z</em>

and the label is <em>RecordType</em>.

## Training dataset ##

For training our models, the input datasets are the csv files, with the columns of features + label, totally 104 columns (103 features + 1 label), in the order listed above.

Put the training datasets in a specific path (or many paths), then update the code to read dataset in the path(s).

## Model Input Facial Features ##

After the feature evaluation, we pick up 5 facial features for model, in order are:
* HeadPoseAngles_Y
* HeadPoseAngles_Z
* GazeVector_X
* GazeVector_Y
* GazeVector_Z

## Model Output Values ##

The model output values are:
* 0 - Not concentrated
* 1 - Concentrated
