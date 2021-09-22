import numpy as np
import cv2
import os
import csv
import sys
sys.path.insert(1, 'D:/blajar_program/Progate/Python/LearnMediapipe/expressionDetection/face_mesh')
from FaceMesh import FaceMesh

if __name__ == '__main__':

    cap = cv2.VideoCapture(0)
    fm = FaceMesh()
    ## DO THIS MULTIPLE TIME AND DON'T FOR GET TO CHANGE CLASS NAME INTO EXPRESSION DO YOU WANT#########
    class_name = "Distraction"
    while cap.isOpened():

        ret, frame = cap.read()
        
        image, results = fm.marked(frame)
        
        #export coordinates
        try:
            #extracting pose landmark
            pose = results.pose_landmarks.landmark #give us pose landmarks
            pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten()) #store all the landmark base on it specific value in an array
            #extracting face landmark
            face = results.face_landmarks.landmark
            face_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in face]).flatten())
            #all of landmarks are wrapping in numpy array 
            #with flatten methode collapses an array into a sigle dimension 
            # ex: [[1,2],[3,4]] => [1,2,3,4]
            
            #combine the array and store it in csv file with spesific class
            row = pose_row+face_row
            #append class name into the first colom
            row.insert(0, class_name)
            
            #import all data into csv file
            with open('coords.csv', mode='a', newline='') as f:
                csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                csv_writer.writerow(row)
            
        except:
            pass
        
        flipHorizontal = cv2.flip(image, 1)
        
        cv2.imshow("Face Mesh", flipHorizontal)

        if cv2.waitKey(10) & 0xFF ==ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
