import cv2
import mediapipe as mp

#------------------Dat Extraction Experiments-----------
"""Function for getting xyz coordinates of all tracked bodypoints"""
def get_points_video(video_path):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, model_complexity=0)
    
    cap = cv2.VideoCapture(video_path)

    points = [[] for _ in range(32)]
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            for i in range(32):
                if i < len(landmarks):
                    lm = landmarks[i]
                    points[i].append((lm.x, lm.y, lm.z))
                else:
                    points[i].append(None)
        else:
            for i in range(32):
                points[i].append(None)

    cap.release()
    pose.close()
    return points, frame_count

"""Function for determining which lift it is"""
def determine_lift():
    pass

"""Function that calls correct pivot frame function"""

"""Function for determining pivot frames for squat (start, bottom, stop)"""

"""Function for determining pivot frames for bench (start, pause start, pause end, stop)"""

"""Function for determining pivot frames for deadlift (start, top)"""

"""Function for cropping point data to within relevant frames"""

#-------------------Media Pipe testing-------------------

def test_image_posing():
  mp_pose = mp.solutions.pose
  pose = mp_pose.Pose(static_image_mode=True, model_complexity=0)
  mp_drawing = mp.solutions.drawing_utils
  image_path = 'image.png'
  image = cv2.imread(image_path)

  image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  results = pose.process(image_rgb)
  if results.pose_landmarks:
      annotated_image = image.copy()
      mp_drawing.draw_landmarks(
          annotated_image, 
          results.pose_landmarks, 
          mp_pose.POSE_CONNECTIONS,
          mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
          mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2))
      cv2.imshow('Pose Landmarks', annotated_image)
      cv2.waitKey(0)
      cv2.destroyAllWindows()
  else:
      print("No pose landmarks found in the image.")
  pose.close()

def test_webcam_posing():
  mp_pose = mp.solutions.pose
  pose = mp_pose.Pose(static_image_mode=False, model_complexity=0)
  mp_drawing = mp.solutions.drawing_utils
  cap = cv2.VideoCapture(0)

  while cap.isOpened():
      ret, frame = cap.read()
      if not ret:
          break
      frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      results = pose.process(frame_rgb)
      if results.pose_landmarks:
          mp_drawing.draw_landmarks(
              frame, 
              results.pose_landmarks, 
              mp_pose.POSE_CONNECTIONS,
              mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
              mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2))
      cv2.imshow('Pose Estimation - Video Mode', frame)
      if cv2.waitKey(1) & 0xFF == ord('q'):
          break

  cap.release()
  cv2.destroyAllWindows()
  pose.close()

def test_saved_video_posing():
  video_path = "video.mp4"
  mp_pose = mp.solutions.pose
  pose = mp_pose.Pose(static_image_mode=False, model_complexity=0)
  mp_drawing = mp.solutions.drawing_utils
  cap = cv2.VideoCapture(video_path)

  while cap.isOpened():
      ret, frame = cap.read()
      if not ret:
          break
      frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      results = pose.process(frame_rgb)
      
      if results.pose_landmarks:
          mp_drawing.draw_landmarks(
              frame, 
              results.pose_landmarks, 
              mp_pose.POSE_CONNECTIONS,
              mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
              mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2))
      
      cv2.imshow('Pose Estimation - Video Mode', frame)
      if cv2.waitKey(1) & 0xFF == ord('q'):
          break

  cap.release()
  cv2.destroyAllWindows()
  pose.close()



if __name__ == "__main__":
    pass
