import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque, Counter
import pyttsx3
import screen_brightness_control as sbc
import pygame
import os

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
        
       
        self.tts_engine = pyttsx3.init()
        self.tts_engine.setProperty('rate', 150)
        self.tts_engine.setProperty('volume', 0.8)
        
        
        pygame.mixer.init()
        
        
        self.gesture_history = deque(maxlen=10)
        self.last_spoken_gesture = None
        
       
        self.current_volume = 50
        self.volume_step = 5
        
     
        self.current_brightness = 50
        
        self.gesture_mapping = {
            'fist': {'detector': self._is_fist, 'voice': "Fist detected", 'action': self._fist_action},
            'peace': {'detector': self._is_peace, 'voice': "Peace sign", 'action': self._peace_action},
            'thumbs_up': {'detector': self._is_thumbs_up, 'voice': "Thumbs up", 'action': self._thumbs_up_action},
            'thumbs_down': {'detector': self._is_thumbs_down, 'voice': "Thumbs down", 'action': self._thumbs_down_action},
            'open_palm': {'detector': self._is_open_palm, 'voice': "Open palm", 'action': self._open_palm_action},
            'pointing': {'detector': self._is_pointing, 'voice': "Pointing", 'action': self._pointing_action},
            'ok': {'detector': self._is_ok, 'voice': "Okay sign", 'action': self._ok_action},
            'rock': {'detector': self._is_rock, 'voice': "Rock on", 'action': self._rock_action},
            'volume_up': {'detector': self._is_volume_up, 'voice': "Volume up", 'action': self._volume_up_action},
            'volume_down': {'detector': self._is_volume_down, 'voice': "Volume down", 'action': self._volume_down_action},
            'brightness_up': {'detector': self._is_brightness_up, 'voice': "Brightness up", 'action': self._brightness_up_action},
            'brightness_down': {'detector': self._is_brightness_down, 'voice': "Brightness down", 'action': self._brightness_down_action}
        }
    
    def speak(self, text):
        """Speak the given text"""
        try:
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()
        except Exception as e:
            print(f"TTS Error: {e}")
    
    def set_system_volume(self, volume):
        """Set system volume (0-100)"""
        try:
            self.current_volume = max(0, min(100, volume))
            
            pygame.mixer.music.set_volume(self.current_volume / 100.0)
            print(f"Volume set to: {self.current_volume}%")
        except Exception as e:
            print(f"Volume control error: {e}")
    
    def set_system_brightness(self, brightness):
        """Set system brightness (0-100)"""
        try:
            self.current_brightness = max(0, min(100, brightness))
            sbc.set_brightness(self.current_brightness)
            print(f"Brightness set to: {self.current_brightness}%")
        except Exception as e:
            print(f"Brightness control error: {e}")
    
    def _calculate_distance(self, point1, point2):
        return np.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)
    
    def _calculate_angle(self, point1, point2, point3):
        """Calculate angle between three points"""
        v1 = np.array([point1.x - point2.x, point1.y - point2.y])
        v2 = np.array([point3.x - point2.x, point3.y - point2.y])
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        return np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
    
   
    def _is_fist(self, landmarks):
        finger_tips = [8, 12, 16, 20]
        finger_pips = [6, 10, 14, 18]
        return all(landmarks[tip].y > landmarks[pip].y for tip, pip in zip(finger_tips, finger_pips))
    
    def _is_peace(self, landmarks):
        return (landmarks[8].y < landmarks[6].y and
                landmarks[12].y < landmarks[10].y and
                landmarks[16].y > landmarks[14].y and
                landmarks[20].y > landmarks[18].y)
    
    def _is_thumbs_up(self, landmarks):
        return (landmarks[4].y < landmarks[3].y and
                all(landmarks[tip].y > landmarks[tip-2].y for tip in [8, 12, 16, 20]))
    
    def _is_thumbs_down(self, landmarks):
        return (landmarks[4].y > landmarks[3].y and
                all(landmarks[tip].y > landmarks[tip-2].y for tip in [8, 12, 16, 20]))
    
    def _is_open_palm(self, landmarks):
        finger_tips = [8, 12, 16, 20]
        return all(landmarks[tip].y < landmarks[tip-2].y for tip in finger_tips)
    
    def _is_pointing(self, landmarks):
        return (landmarks[8].y < landmarks[6].y and
                all(landmarks[tip].y > landmarks[tip-2].y for tip in [12, 16, 20]))
    
    def _is_ok(self, landmarks):
        thumb_index_dist = self._calculate_distance(landmarks[4], landmarks[8])
        return thumb_index_dist < 0.05
    
    def _is_rock(self, landmarks):
        return (landmarks[8].y < landmarks[6].y and
                landmarks[20].y < landmarks[18].y and
                landmarks[12].y > landmarks[10].y and
                landmarks[16].y > landmarks[14].y)
    
    def _is_volume_up(self, landmarks):

        return (self._is_thumbs_up(landmarks) and 
                landmarks[0].x < landmarks[9].x)  # Palm orientation
    
    def _is_volume_down(self, landmarks):
       
        return (self._is_thumbs_down(landmarks) and 
                landmarks[0].x < landmarks[9].x)  # Palm orientation
    
    def _is_brightness_up(self, landmarks):
        
        return (self._is_open_palm(landmarks) and
                landmarks[0].y < landmarks[9].y)
    
    def _is_brightness_down(self, landmarks):
        
        return (self._is_open_palm(landmarks) and
                landmarks[0].y > landmarks[9].y)
    
   
    def _fist_action(self):
        self.speak("Fist gesture detected")
    
    def _peace_action(self):
        self.speak("Peace sign detected")
    
    def _thumbs_up_action(self):
        self.speak("Thumbs up! Good job!")
        self.set_system_volume(min(100, self.current_volume + 10))
    
    def _thumbs_down_action(self):
        self.speak("Thumbs down")
        self.set_system_volume(max(0, self.current_volume - 10))
    
    def _open_palm_action(self):
        self.speak("Open palm detected")
    
    def _pointing_action(self):
        self.speak("Pointing gesture")
    
    def _ok_action(self):
        self.speak("Okay sign detected")
    
    def _rock_action(self):
        self.speak("Rock and roll!")
    
    def _volume_up_action(self):
        new_volume = min(100, self.current_volume + self.volume_step)
        self.set_system_volume(new_volume)
        self.speak(f"Volume increased to {new_volume} percent")
    
    def _volume_down_action(self):
        new_volume = max(0, self.current_volume - self.volume_step)
        self.set_system_volume(new_volume)
        self.speak(f"Volume decreased to {new_volume} percent")
    
    def _brightness_up_action(self):
        new_brightness = min(100, self.current_brightness + 10)
        self.set_system_brightness(new_brightness)
        self.speak(f"Brightness increased to {new_brightness} percent")
    
    def _brightness_down_action(self):
        new_brightness = max(0, self.current_brightness - 10)
        self.set_system_brightness(new_brightness)
        self.speak(f"Brightness decreased to {new_brightness} percent")
    
    def recognize_gesture(self, landmarks):
        best_gesture = None
        best_confidence = 0
        
        for gesture_name, gesture_data in self.gesture_mapping.items():
            if gesture_data['detector'](landmarks):
                confidence = 0.9  # Base confidence
                best_gesture = gesture_name
                best_confidence = confidence
        
        if best_gesture:
            self.gesture_history.append((best_gesture, best_confidence))
       
            gesture_counts = Counter([g[0] for g in self.gesture_history])
            return gesture_counts.most_common(1)[0][0]
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
                
                
                if gesture != self.last_spoken_gesture and gesture != "unknown":
                    gesture_data = self.gesture_mapping.get(gesture)
                    if gesture_data:
                        gesture_data['action']()
                        self.last_spoken_gesture = gesture
                
              
                cv2.putText(frame, f"Gesture: {gesture}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Volume: {self.current_volume}%", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                cv2.putText(frame, f"Brightness: {self.current_brightness}%", (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        return frame, gesture

def main():
    tracker = AdvancedHandTracker()
    cap = cv2.VideoCapture(0)
    
    print("🎯 Advanced Hand Tracker with Voice Control Started!")
    print("🔊 Available gestures with voice feedback:")
    print("   ✊ Fist")
    print("   ✌️ Peace Sign")  
    print("   👍 Thumbs Up (Volume +10%)")
    print("   👎 Thumbs Down (Volume -10%)")
    print("   🖐️ Open Palm")
    print("   👉 Pointing")
    print("   👌 OK Sign")
    print("   🤘 Rock Sign")
    print("   📢 Volume Up Gesture")
    print("   📢 Volume Down Gesture")
    print("   💡 Brightness Up Gesture")
    print("   💡 Brightness Down Gesture")
    print("   Press 'Q' to quit")
    
    # Initial system settings
    tracker.set_system_volume(50)
    tracker.set_system_brightness(50)
    tracker.speak("Hand gesture control system activated")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.flip(frame, 1)
        processed_frame, gesture = tracker.process_frame(frame)
        
        # Add control instructions on screen
        cv2.putText(processed_frame, "Thumbs Up/Down: Volume Control", (10, 120),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(processed_frame, "Palm Up/Down: Brightness Control", (10, 140),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow('Advanced Hand Tracking with Voice Control', processed_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    tracker.speak("Hand gesture system closed")

if __name__ == "__main__":
    main()
