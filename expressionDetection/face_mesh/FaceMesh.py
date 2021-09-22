import mediapipe as mp
import pandas as pd
import cv2

class FaceMesh():

    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence = 0.5):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            min_detection_confidence = min_detection_confidence,
            min_tracking_confidence = min_tracking_confidence
        )
        self.landmark_color = self.mp_drawing.DrawingSpec(color=(74, 198, 243), thickness=1, circle_radius=1)
        self.connection_color = self.mp_drawing.DrawingSpec(color=(82, 159, 207), thickness=1, circle_radius=1)

    def marked(self, frame):
        #recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable= False

        #passing result to make detection
        results = self.holistic.process(image)

        #recoloring back into BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        #draw face landmarks
        self.mp_drawing.draw_landmarks(
            image, 
            results.face_landmarks, 
            self.mp_holistic.FACEMESH_CONTOURS,
            self.landmark_color,
            self.connection_color
        )
        # #draw face tesselation if needed
        # self.mp_drawing.draw_landmarks(
        #     image, 
        #     results.face_landmarks, 
        #     self.mp_holistic.FACEMESH_TESSELATION,
        #     self.landmark_color,
        #     self.connection_color
        # )

        # #draw left hand landmarks if needed
        # self.mp_drawing.draw_landmarks(
        #     image,
        #     results.left_hand_landmarks,
        #     self.mp_holistic.HAND_CONNECTIONS,
        #     self.landmark_color,
        #     self.connection_color
        # )

        # #draw right hand landmarks if needed
        # self.mp_drawing.draw_landmarks(
        #     image,
        #     results.right_hand_landmarks,
        #     self.mp_holistic.HAND_CONNECTIONS,
        #     self.landmark_color,
        #     self.connection_color
        # )

        # #draw pose landmarks if needed
        # self.mp_drawing.draw_landmarks(
        #     image,
        #     results.pose_landmarks,
        #     self.mp_holistic.POSE_CONNECTIONS,
        #     self.landmark_color,
        #     self.connection_color
        # )
        
        return image, results

# if __name__ == '__main__':
#     fm = FaceMesh()
    
#     cap = cv2.VideoCapture(0)
#     while cap.isOpened():
#         ret, frame = cap.read()

#         image, results = fm.marked(frame)
#         flipHorizontal = cv2.flip(image, 1)
#         cv2.imshow("Face Mesh", flipHorizontal)

#         if cv2.waitKey(10) & 0xFF ==ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()
