# pip install opencv-python
import cv2
import numpy as np
import time
import os
new_shape=(640,360)
# モデルを読み込む
directory = os.path.dirname(__file__)
print(f"directory:{directory}")
weights = os.path.join(directory, "face_detection_yunet_2022mar.onnx")
face_detector = cv2.FaceDetectorYN_create(weights, "", new_shape)

# 初始化FaceRecognizerSF
recog_model_path = os.path.join(directory, "face_recognition_sface_2021dec.onnx")
recognizer = cv2.FaceRecognizerSF.create(recog_model_path, "" )


# Open the device at the ID 0
cap = cv2.VideoCapture(0)

# To set the resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Check whether user selected camera is opened successfully.
if not (cap.isOpened()):
    print("Could not open video device")

fps = cap.get(cv2.CAP_PROP_FPS)
print("Frames per second using cv2.CAP_PROP_FPS : {0}".format(fps))
cv2.namedWindow("output", cv2.WINDOW_NORMAL) 
# Resize the Window
cv2.resizeWindow("output", 640, 360)

feature1=np.array([])
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # 計算每一幀的處理時間
    start_time = time.time()

     # 画像が3チャンネル以外の場合は3チャンネルに変換する
    channels = 1 if len(frame.shape) == 2 else frame.shape[2]
    if channels == 1:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if channels == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    # 入力サイズを指定する
    height, width, _ = frame.shape
    newheight, newwidth=new_shape
    #rationH=height/newheight
    ration=width/newheight
    #face_detector.setInputSize((width, height))

    #print(f"rationH: {rationH} , rationW=: {rationW}")
    image = cv2.resize(frame, new_shape) 
    # 顔を検出する
    _, faces = face_detector.detect(image)
    faces = faces if faces is not None else []


    #print(f"faces: {faces}")
    # 検出した顔のバウンディングボックスとランドマークを描画する
    fnum=0
    faceStatus='Right'
    FrontFacing='Facing'
    for face in faces:
        # バウンディングボックス
        box = list(map(int, face[:4]*ration))
        #print(f"box: {box}")
        color = (0, 0, 255)
        thickness = 2
        cv2.rectangle(frame, box, color, thickness, cv2.LINE_AA)

        # ランドマーク（右目、左目、鼻、右口角、左口角）
        landmarks = list(map(int, face[4:len(face)-1]*ration))
        #print(f"landmarks1: {landmarks}")
        #cacl location
        # Right eye X 
        reX=landmarks[0]
        if(reX<=0):reX=1
        # Right lip X 
        rlX=landmarks[6]
        # noise x
        nX=landmarks[4]
        # Left eye X 
        leX=landmarks[2]
        if(leX<=0):leX=1
        # Left lip X 
        llX=landmarks[8]

        faceFontStatusWeight=abs(1-(nX/reX))-abs(1-(nX/leX))
        if(faceFontStatusWeight>0): faceStatus='Left'
        if(abs(faceFontStatusWeight)>0.05): FrontFacing='NotFacing'
        #print(f"re: {abs(1-(nX/reX))}, le:{abs(1-(nX/leX))}")
        #print(f"faceFontStatusWeight: {faceFontStatusWeight}")
        #print(f"rl: {rlX/nX}, ll:{llX/nX}")
        landmarks = np.array_split(landmarks, len(landmarks) / 2)
        for landmark in landmarks:
            radius = 5
            thickness = -1
            cv2.circle(frame, landmark, radius, color, thickness, cv2.LINE_AA)
        
        #print(f"landmarks2: {landmarks}")       
        # 信頼度
        confidence = face[-1]
        confidence = "{:.2f}".format(confidence)
        position = (box[0], box[1] - 10)
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.5
        thickness = 2
        cv2.putText(frame, confidence, position, font, scale, color, thickness, cv2.LINE_AA)
        cv2.putText(frame, f'Face:{fnum} , status: {faceStatus}',  (box[0]+100, box[1] - 40), font, 1, color, thickness, cv2.LINE_AA)
        cv2.putText(frame, f'FrontFacing: {FrontFacing}',  (box[0]+100, box[1]-10 ), font, 1, color, thickness, cv2.LINE_AA)
        
        fnum=fnum+1

        
        # 在人脸检测部分的基础上, 对齐检测到的首个人脸(faces[0]), 保存至aligned_face。
        aligned_face = recognizer.alignCrop(frame, face)
        # 在上文的基础上, 获取对齐人脸的特征feature。
        feature2 = recognizer.feature(aligned_face)

        if(feature1.size==0):feature1=feature2
        #print(f"feature: {feature}")
        # 在上文的基础上, 比对两张人脸的特征feature1，feature2以确认身份。
        # 使用consine距离作为指标
        cosineResult=False
        cosine_similarity_threshold = 0.363
        cosine_score = recognizer.match(feature1, feature2, 0) 
        if cosine_score >= cosine_similarity_threshold: 
            cosineResult=True

        # 使用normL2距离作为指标
        normL2Result=False
        l2_similarity_threshold = 1.128
        l2_score = recognizer.match(feature1, feature2, 1)
        if l2_score <= l2_similarity_threshold: 
            normL2Result=True
        
        if(cosineResult&normL2Result):print(f"recognizer: is Same")
        else:print(f"recognizer: is Not Same")
    
    # 將FPS值顯示在視頻上
    cv2.putText(frame, f"FPS: {str(int(fps))}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 0, 255), 2)
    
    # Display the resulting frame
    cv2.imshow("output",frame)

    # 計算FPS值
    end_time = time.time()
    fps = 1 / (end_time - start_time)
    
    # Waits for a user input to quit the application
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()