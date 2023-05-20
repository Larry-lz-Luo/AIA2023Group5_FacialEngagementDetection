# Model Creation & Evaluation #

We created some codes for model creation and evaluation. These models are for concentration detection, once input related facial features, the model will output concentration judgement result.

## Model List ##

* Decision Tree
* KNN
* Random Forest
* XGBoost

## Model Input Facial Features ##

The facial features for model, in order are:
* HeadPoseAngles_Y
* HeadPoseAngles_Z
* GazeVector_X
* GazeVector_Y
* GazeVector_Z

## Model Output Values ##

The model output values are:
* 0 - Not concentrated
* 1 - Concentrated
