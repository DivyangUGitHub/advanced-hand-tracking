import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque, Counter

class AdvancedHandTracker:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.hands = self.mp_hands.Hands(
            model_complexity=1,
            min_detection_confidence=0.8,
            min_tracking_confidence=0.8,
            max_num_hands=2
        )
        
        self.gesture_history = deque(maxlen=10)
        self.gesture_mapping = {
            'fist': self._is_fist,
            'peace': self._is_peace,
            'thumbs_up': self._is_thumbs_up,
            'open_palm': self._is_open_palm,
            'pointing': self._is_pointing,
            'ok': self._is_ok,
            'rock': self._is_rock,
            'phone': self._is_phone
        }
    
    def _calculate_distance(self, point1, point2):
        return np.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)
    
    def _is_fist(self, landmarks):

        finger_tips = [8, 12, 16, 20]
        finger_mcp = [5, 9, 13, 17]
        
        for tip, mcp in zip(finger_tips, finger_mcp):
            if self._calculate_distance(landmarks[tip], landmarks[mcp]) > 0.1:
                return False
        return True
    
    def _is_peace(self, landmarks):
        
        return (landmarks[8].y < landmarks[6].y and  
                landmarks[12].y < landmarks[10].y and  
                landmarks[16].y > landmarks[14].y and  
                landmarks[20].y > landmarks[18].y)     
    
    def _is_thumbs_up(self, landmarks):
        return (landmarks[4].y < landmarks[3].y and  
                landmarks[8].y > landmarks[6].y and  
                landmarks[12].y > landmarks[10].y and
                landmarks[16].y > landmarks[14].y and
                landmarks[20].y > landmarks[18].y)
    
    def _is_open_palm(self, landmarks):
        finger_tips = [8, 12, 16, 20]
        for tip in finger_tips:
            if landmarks[tip].y > landmarks[tip-2].y:
                return False
        return True
    
    def _is_pointing(self, landmarks):
        return (landmarks[8].y < landmarks[6].y and  
                landmarks[12].y > landmarks[10].y and 
                landmarks[16].y > landmarks[14].y and
                landmarks[20].y > landmarks[18].y)
    
    def _is_ok(self, landmarks):
        
        thumb_index_dist = self._calculate_distance(landmarks[4], landmarks[8])
        return thumb_index_dist < 0.05
    
    def _is_rock(self, landmarks):
        
        return (landmarks[8].y < landmarks[6].y and 
                landmarks[12].y > landmarks[10].y and  
                landmarks[16].y > landmarks[14].y and  
                landmarks[20].y < landmarks[18].y)     
    
    def _is_phone(self, landmarks):
        
        thumb_pinky_dist = self._calculate_distance(landmarks[4], landmarks[20])
        return thumb_pinky_dist < 0.15
    
    def recognize_gesture(self, landmarks):
        gestures = []
        for gesture_name, gesture_func in self.gesture_mapping.items():
            if gesture_func(landmarks):
                gestures.append(gesture_name)
        
        if gestures:
            self.gesture_history.append(gestures[0])
           
            return Counter(self.gesture_history).most_common(1)[0][0]
        return "unknown"
    
    def process_frame(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        gesture = "No hand detected"
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                
                self.mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
                
                gesture = self.recognize_gesture(hand_landmarks.landmark)
                
                
                cv2.putText(frame, f"Gesture: {gesture}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return frame, gesture

def main():
    tracker = AdvancedHandTracker()
    cap = cv2.VideoCapture(0)
    
    print("Advanced Hand Tracker Started!")
    print("Available gestures: fist, peace, thumbs_up, open_palm, pointing, ok, rock, phone")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.flip(frame, 1)
        processed_frame, gesture = tracker.process_frame(frame)
        
        cv2.imshow('Advanced Hand Tracking', processed_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()