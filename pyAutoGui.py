
import cv2
import mediapipe as mp
import os
import threading
from deepface import DeepFace

# Initialize MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Start video capture
cap = cv2.VideoCapture(0)

# Global variables for storing results
analysis_result = {"dominant_emotion": "Detecting...", "emotion_percentages": {}, "age": "N/A", "gender": "N/A"}

def analyze_face(frame):
    """Runs DeepFace analysis in a separate thread to prevent lag."""
    global analysis_result
    try:
        analysis = DeepFace.analyze(frame, actions=['emotion', 'age', 'gender'], enforce_detection=False)
        if analysis:
            analysis_result["dominant_emotion"] = analysis[0].get('dominant_emotion', 'Unknown')
            analysis_result["emotion_percentages"] = analysis[0].get('emotion', {})
            analysis_result["age"] = analysis[0].get('age', 'N/A')
            analysis_result["gender"] = analysis[0].get('gender', 'N/A')
    except Exception as e:
        print(f"DeepFace error: {e}")
        analysis_result = {"dominant_emotion": "Error", "emotion_percentages": {}, "age": "N/A", "gender": "N/A"}

with mp_face_mesh.FaceMesh(min_detection_confidence=0.7, min_tracking_confidence=0.7) as face_mesh:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image_rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION,
                                          mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                                          mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=1, circle_radius=1))

            # Run DeepFace analysis in a separate thread
            threading.Thread(target=analyze_face, args=(frame.copy(),), daemon=True).start()

        # Display results
        cv2.putText(frame, f"Emotion: {analysis_result['dominant_emotion']}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"Age: {analysis_result['age']}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"Gender: {analysis_result['gender']}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Show emotion percentages
        y_offset = 120
        for emotion, percentage in analysis_result["emotion_percentages"].items():
            cv2.putText(frame, f"{emotion}: {percentage:.2f}%", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            y_offset += 20

        cv2.imshow('Face Analysis', frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

# Release resources
cap.release()
cv2.destroyAllWindows()

# Cleanup temporary DeepFace files
tmp_dirs = ['.deepface', 'representations_vgg_face.pkl']
for tmp in tmp_dirs:
    if os.path.exists(tmp):
        try:
            os.remove(tmp) if os.path.isfile(tmp) else os.rmdir(tmp)
        except Exception as e:
            print(f"Cleanup error: {e}")
