import sys
import logging as log
import cv2
import numpy as np
import math
import time

from inference import Network

class EyeStateEstimator:

    def __init__(self, model_name, device='CPU', extensions=None):
        self.network = Network(model_name, device, extensions)

    def load_model(self):
        self.network.load_model()

    def rotateImageAroundCenter(self, srcImage, angle):
        w = srcImage.shape[1]
        h = srcImage.shape[0]
        # Check if the source image has non-zero dimensions
        if w == 0 or h == 0:
            print("Error: source image has zero dimensions")
            return None
        size = (w, h)
        center = (w/2, h/2)

        rotMatrix = cv2.getRotationMatrix2D(center, angle, 1)
        dstImage = cv2.warpAffine(srcImage, rotMatrix, size, flags=cv2.BORDER_REPLICATE)

        return dstImage
    
    def predict(self, eye_image, head_pose_angles):
        _, _, roll = head_pose_angles
        eye_image, head_pose_angles, preprocess_input_time = self._preprocess_input(eye_image, head_pose_angles)
        #input_dict = {"left_eye_image": left_eye_image, "right_eye_image": right_eye_image, "head_pose_angles": head_pose_angles}
        
        self.network.exec_net(0, eye_image)
        status = self.network.wait(0)
        if status == 0:
            outputs = self.network.get_output(0)
            eye_states, preprocess_output_time = self._preprocess_output(outputs, roll)
            self.preprocess_time = preprocess_input_time + preprocess_output_time
            return eye_states

    def _preprocess_input(self, eye_image, head_pose_angles):
        start_preprocess_time = time.time()
        eye_image = self.rotateImageAroundCenter(eye_image, float(head_pose_angles[2]))
        eye_image = self._preprocess_eye_image(eye_image)
        #right_eye_image = self._preprocess_eye_image(right_eye_image)
        #head_pose_angles = self._preprocess_angels(head_pose_angles)
        total_preprocess_time = time.time() - start_preprocess_time
        return eye_image, head_pose_angles, total_preprocess_time    

    def _preprocess_angels(self, head_pose_angles):
        input_shape = self.network.get_input_shape("head_pose_angles")
        head_pose_angles = np.reshape(head_pose_angles, input_shape)
        return head_pose_angles

    def _preprocess_eye_image(self, image):
        n, c, h, w = self.network.get_input_shape()
        input_image = cv2.resize(image, (w,h), interpolation = cv2.INTER_AREA)
        input_image = input_image.transpose((2, 0, 1))
        input_image = input_image.reshape((n, c, h, w))
        return input_image

    def _preprocess_output(self, outputs, roll):
        start_preprocess_time = time.time()
        eyeState_vector = outputs[0]
        eye_states = 0
        if(eyeState_vector[0]<eyeState_vector[1]):
            eye_states = 1
        # gaze_vector_n = eyeState_vector.
        # vcos = math.cos(math.radians(roll))
        # vsin = math.sin(math.radians(roll))
        # x =  gaze_vector_n[0]*vcos + gaze_vector_n[1]*vsin
        # y = -gaze_vector_n[0]*vsin + gaze_vector_n[1]*vcos
        total_preprocess_time = time.time() - start_preprocess_time
        return eye_states, total_preprocess_time

