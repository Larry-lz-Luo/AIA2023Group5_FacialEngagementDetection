import cv2
import mediapipe as mp
import numpy as np
import math

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

mp_face_detection = mp.solutions.face_detection
face_detection=mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) 

mp_drawing = mp.solutions.drawing_utils   

cap = cv2.VideoCapture(0)
# To set the resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

def distance(p1, p2):
    return ((p1[0]-p2[0])**2+(p1[1]-p2[1])**2+(p1[2]-p2[2])**2)**0.5


e_frame=m_frame=0
while cap.isOpened():
    success, image = cap.read()

    # Flip the image horizontally for a later selfie-view display
    # Also convert the color space from BGR to RGB
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

    # To improve performance
    image.flags.writeable = False
    
    # Get the result
    fm_res = face_mesh.process(image)
    img_h, img_w, img_c = image.shape
    
    fd_res = face_detection.process(image)
    if fd_res.detections:
        for detection in fd_res.detections:
            #mp_drawing.draw_detection(image, detection)  # 標記人臉

            fd_xmin = min(math.floor(detection.location_data.relative_bounding_box.xmin * img_w), img_w - 1)
            fd_ymin = min(math.floor(detection.location_data.relative_bounding_box.ymin * img_h), img_h - 1)
            fd_width=detection.location_data.relative_bounding_box.width * img_w
            fd_height=detection.location_data.relative_bounding_box.height * img_h
            fd_score=detection.score
    
    cv2.rectangle(image, (fd_xmin,fd_ymin),(int(fd_xmin+fd_width),int(fd_ymin+fd_height)), (255,255,255), 2, cv2.LINE_AA)
    #cv2.putText(image ,"fd_score: " +str(np.round(fd_score,2)), (fd_xmin,fd_ymin-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # To improve performance
    image.flags.writeable = True
    
    # Convert the color space from RGB to BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    face_3d = []
    face_2d = []
    em_lm={}

    if fm_res.multi_face_landmarks:
        for face_landmarks in fm_res.multi_face_landmarks:
            for idx, lm in enumerate(face_landmarks.landmark):
                if idx in [33,159,133,145,362,257,263,374,57,0,287,17]:
                    em_lm[str(idx)]=(lm.x,lm.y,lm.z)
                
                if idx in [33, 263 ,1 , 61 ,291 , 199]:
                    if idx == 1:
                        nose_2d = (lm.x * img_w, lm.y * img_h)
                        nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)

                    x, y = int(lm.x * img_w), int(lm.y * img_h)

                    # Get the 2D Coordinates
                    face_2d.append([x, y])

                    # Get the 3D Coordinates
                    face_3d.append([x, y, lm.z])       
            
            # EAR 
            EAR=(distance(em_lm['159'], em_lm['145'])/distance(em_lm['33'], em_lm['133'])+\
                distance(em_lm['257'], em_lm['374'])/distance(em_lm['362'], em_lm['263']))/2
            
            #MAR 
            MAR=distance(em_lm['0'], em_lm['17'])/distance(em_lm['57'], em_lm['287'])
            

            if MAR >= 1:
                m_frame += 1
                if m_frame >= 30:
                    cv2.putText(image, 'Feel drowsy ?', (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
            else:
                m_frame=0
                              
            if EAR <=0.65:
                e_frame += 1
                if e_frame >= 30:
                    cv2.putText(image, 'Wake Up !', (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
            else:
                e_frame=0
            

            # Convert it to the NumPy array
            face_2d = np.array(face_2d, dtype=np.float64)

            # Convert it to the NumPy array
            face_3d = np.array(face_3d, dtype=np.float64)

            # The camera matrix
            focal_length = 1 * img_w

            cam_matrix = np.array([ [focal_length, 0, img_h / 2],
                                    [0, focal_length, img_w / 2],
                                    [0, 0, 1]])

            # The Distance Matrix
            dist_matrix = np.zeros((4, 1), dtype=np.float64)

            # Solve PnP
            success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

            # Get rotational matrix
            rmat, jac = cv2.Rodrigues(rot_vec)

            # Get angles
            angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

            # Get the y rotation degree
            x = angles[0] * 360
            y = angles[1] * 360
            z = angles[2] * 360

            # print(y)

            # See where the user's head tilting
            if y < -7:
                text = "Looking Left"
            elif y > 7:
                text = "Looking Right"
            elif x > 8 :
                text = "Looking Up"
            elif x < 1:
                text = "Looking Down"
            else:
                text = "Face Forward"

            # Display the nose direction
            nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)

            p1 = (int(nose_2d[0]), int(nose_2d[1]))
            p2 = (int(nose_2d[0]+y*10  ), int(nose_2d[1]-x*10))
            
            #cv2.line(image, p1, p2, (255, 0, 0), 2)
            cv2.arrowedLine(image,p1,p2,  (0, 255, 0),5)

            # Add the text on the image
            cv2.putText(image, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
            cv2.putText(image ,"Pitch: "+str(np.round(x,2)), (1080,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(image ,"Yaw: "  +str(np.round(y,2)), (1080,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(image ,"Roll: " +str(np.round(z,2)), (1080,150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(image ,"EAR: " +str(np.round(EAR,2)),(1080,200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(image ,"MAR: " +str(np.round(MAR,2)),(1080,250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Head Pose Estimation', image)

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
