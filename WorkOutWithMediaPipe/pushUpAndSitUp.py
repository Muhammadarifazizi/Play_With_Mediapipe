# Add dependencies
import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils #add drawing utils
mp_pose = mp.solutions.pose #add pose estimation

#function to calculate angle
def calculate_angle(a, b, c):
    a = np.array(a) #left side
    b = np.array(b) #center join
    c = np.array(c) #bottom side
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])#convert to rad
    angle = np.abs(radians*180.0/np.pi)#rad to angle
    
    if angle > 180.0:
        angle = 360-angle
        
    return angle

#get access to webcam
cap = cv2.VideoCapture(0)

#state variable
workout = None
elbow = 0
hip = 0
status = None

#setup mediapip instance
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
  while cap.isOpened():
    ret, frame = cap.read()
    
    # Recolor image to RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
   
    # make detection
    results = pose.process(image)

    # recoloring back to BGR
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # extract landmark
    try:
        landmarks = results.pose_landmarks.landmark
        #print(landmarks)
        
        #get coordinate
        leftElbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        rightElbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
        leftWrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
        rightWrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
        leftShoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        rightShoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        leftHip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        rightHip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
        leftKnee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
        rightKnee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
        leftAnkle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
        rightAnkle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
        leftHeel = [landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].y]
        rightHeel = [landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].y]
        leftFootIndex = [landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].x, landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y]
        rightFootIndex = [landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].y]
        
        #calculate angle
        angleRightElbow = calculate_angle(rightShoulder, rightElbow, rightWrist)
        angleLeftElbow = calculate_angle(leftShoulder, leftElbow, leftWrist)
        angleRightShoulder = calculate_angle(rightHip, rightShoulder, rightElbow)
        angleLeftShoulder = calculate_angle(leftHip, leftShoulder, leftElbow)
        angleRightHip = calculate_angle(rightShoulder, rightHip, rightKnee)
        angleLeftHip = calculate_angle(leftShoulder, leftHip, leftKnee)
        angleRightKnee = calculate_angle(rightHip, rightKnee, rightAnkle)
        angleLeftKnee = calculate_angle(leftHip, leftKnee, leftAnkle)
        angleRightAnkle = calculate_angle(rightKnee, rightAnkle, rightFootIndex)
        angleLeftAnkle = calculate_angle(leftKnee, leftAnkle, leftFootIndex)
        
        #put label degree to image
        cv2.putText(image, str(angleRightShoulder),
                   # coordinate shoulder will calculate with the size image from webcam, 
                   # so we can get the real coordinate base on the real size image 
                   tuple(np.multiply(rightShoulder, [640, 480]).astype(int)),
                   # setup style for the label
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (147, 112, 219), 2, cv2.LINE_AA
                    )
        cv2.putText(image, str(angleLeftShoulder), 
                   tuple(np.multiply(leftShoulder, [640, 480]).astype(int)),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (147, 112, 219), 2, cv2.LINE_AA
                    )      
        cv2.putText(image, str(angleRightElbow),
                   tuple(np.multiply(rightElbow, [640, 480]).astype(int)),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (147, 112, 219), 2, cv2.LINE_AA
                    )
        cv2.putText(image, str(angleLeftElbow),
                   tuple(np.multiply(leftElbow, [640, 480]).astype(int)),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (147, 112, 219), 2, cv2.LINE_AA
                    )
        cv2.putText(image, str(angleRightHip),
                   tuple(np.multiply(rightHip, [640, 480]).astype(int)),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (147, 112, 219), 2, cv2.LINE_AA
                    )
        cv2.putText(image, str(angleLeftHip),
                   tuple(np.multiply(leftHip, [640, 480]).astype(int)),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (147, 112, 219), 2, cv2.LINE_AA
                    )
        cv2.putText(image, str(angleRightKnee),
                   tuple(np.multiply(rightKnee, [640, 480]).astype(int)),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (147, 112, 219), 2, cv2.LINE_AA
                    )
        cv2.putText(image, str(angleLeftKnee),
                   tuple(np.multiply(leftKnee, [640, 480]).astype(int)),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (147, 112, 219), 2, cv2.LINE_AA
                    )
        
        #detect pushup from right side
        if  angleRightShoulder < 80  and angleRightHip > 160 and angleRightKnee > 160:
            workout = "Push Up" 

        #detect sit up from right side     
        if angleRightKnee < 90 and angleRightHip < 160:
            workout = "Sit Up" 
        
        #condition Up or Down in Push Up and Sit Up
        if angleRightHip > 100 and workout=="Sit Up":
            status = "Down"
        if angleRightHip < 90 and workout=="Sit Up" and status == "Down":
            status = "Up"
            hip+=1
        if angleRightElbow > 90 and workout=="Push Up":
            status = "Up"
        if angleRightElbow < 90 and status == "Up" and workout=="Push Up":
            status = "Down"
            elbow+=1   
    except:
        pass
    
    #render curl counter
    #setup status box 
    #create rectangle (display in image, start coor position x and y, end coor position x and y, background color in RGB, tickness -1 )
    cv2.rectangle(image, (0,0), (280,60), (245, 117, 16), -1)
    
    # show data
    cv2.putText(image, "Action  |  Status  |  Count", (10,20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 192, 203), 1, cv2.LINE_AA
               )
    #create text (display in image, parameter, start coor position x and y, font type, font scale, font color in RGB, tickness, font style)
    cv2.putText(image, workout, (10,50),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 192, 203), 1, cv2.LINE_AA
               )
    if workout=="Push Up":
        cv2.putText(image, str(elbow), (200,50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 192, 203), 1, cv2.LINE_AA
                   )
    if workout=="Sit Up":
        cv2.putText(image, str(hip), (200,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 192, 203), 1, cv2.LINE_AA
                    )

    cv2.putText(image, status, (100,50),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA
               )
    
    # render detection so we doesn't need to draw point one by one
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(255,0,0), thickness=2, circle_radius=2),
                             mp_drawing.DrawingSpec(color=(147, 112, 219), thickness=2, circle_radius=2)
                             )

    # we change to image so we can see multiple landmark
    cv2.imshow('Workout with Mediapipe', image)
    # destroy visualization with press "q" button
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

  cap.release()
  cv2.destroyAllWindows()
