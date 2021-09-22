import mediapipe
import cv2
import os
import csv
import sys
sys.path.insert(1, 'D:/blajar_program/Progate/Python/LearnMediapipe/expressionDetection/face_mesh')
from FaceMesh import FaceMesh

if __name__ == '__main__':

    cap = cv2.VideoCapture(0)
    fm = FaceMesh()

    while cap.isOpened():

        ret, frame = cap.read()
        
        image, results = fm.marked(frame)
        
        # we are gonna check result type to prevent syntack error if type don't match
        if type(results.face_landmarks) == mediapipe.framework.formats.landmark_pb2.NormalizedLandmarkList:
            num_coords = len(results.face_landmarks.landmark)
        
        flipHorizontal = cv2.flip(image, 1)
        
        cv2.imshow("Face Mesh", flipHorizontal)

        if cv2.waitKey(10) & 0xFF ==ord('q'):
            landmarks = ['class']
            for val in range(1, num_coords+1):
                landmarks += ['x{}'.format(val), 'y{}'.format(val), 'z{}'.format(val), 'v{}'.format(val)]  

            #create a csv file that will save our expression that will be used for classification
            with open('coords.csv', mode='w', newline='') as f:
                csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                csv_writer.writerow(landmarks)
            break

    cap.release()
    cv2.destroyAllWindows()
