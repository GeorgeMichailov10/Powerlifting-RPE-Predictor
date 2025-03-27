import cv2
import mediapipe as mp
import json

#------------------Dat Extraction Experiments-----------

"""Manager function"""
def main(video_path):
    points = get_points_video(video_path)
    lift = determine_lift(points)
    cleaned_points = clean_point_data(lift, points)

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
def determine_lift(points):
    # Note: Hand Points: 15-22 inclusive; Shoulders: 11, 12; Hips: 23, 24;
    # If hands at shoulder level/in line with shoulders: Squat
    # If hands above shoulders/hips and shoulders pretty much horizontal: Bench
    # If hands at hip level/ straight down: Deadlift 
    pass

"""Function for cleaning/splitting frames into only what is relevant"""
def clean_point_data(lift, points):
    pivot_frames = get_pivot_frame_indices(lift, points)
    return clean_point_data(points, pivot_frames)

"""Function that calls correct pivot frame function"""
def get_pivot_frame_indices(lift, points):
    if lift == 'squat':
        return get_squat_pivot_frames(points)
    elif lift == 'bench':
        return get_bench_pivot_frames(points)
    elif lift == 'deadlift':
        return get_deadlift_pivot_frames(points)

"""Function for determining pivot frames for squat (start, bottom, stop)"""
def get_squat_pivot_frames(points):
    pass

"""Function for determining pivot frames for bench (start, pause start, pause end, stop)"""
def get_bench_pivot_frames(points):
    pass

"""Function for determining pivot frames for deadlift (start, top)"""
def get_deadlift_pivot_frames(points):
    pass

"""Function for cropping point data to within relevant frames"""
def crop_to_pivot_frames(points, pivot_frames):
    cleaned_points = []
    for pair in pivot_frames:
        curr_frame_range = []
        for point_data in points:
            curr_frame_range.append(point_data[pair[0]:pair[1]+1])
        cleaned_points.append(curr_frame_range)
    return cleaned_points
            

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
