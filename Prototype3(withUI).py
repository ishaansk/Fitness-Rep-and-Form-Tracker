import cv2
import mediapipe as mp
import numpy as np
import time

# Initialize variables
counter = 0
stage = None
target_reps = 12  # Total reps for one set
sets_completed = 0
display_set_completed = False
waiting_for_next_set = False

# Metrics tracking
set_start_time = None
rep_start_time = None
set_durations = []
rep_durations = []
arm_sync_errors = []
total_workout_start_time = None

# Initialize MediaPipe pose
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Video capture
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Camera not accessible.")
    exit()

def calculate_angle(a, b, c):
    a = np.array(a)  # First point
    b = np.array(b)  # Mid point
    c = np.array(c)  # End point

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

# Helper function to draw rounded rectangles
def draw_rounded_box(image, top_left, bottom_right, color, thickness):
    pts_outer = np.array([top_left, [bottom_right[0], top_left[1]], bottom_right, [top_left[0], bottom_right[1]]], np.int32)
    cv2.polylines(image, [pts_outer], isClosed=True, color=color, thickness=thickness, lineType=cv2.LINE_AA)

# Helper function to draw text with border
def draw_text_with_border(image, text, position, font, scale, color, thickness, border_color, border_thickness):
    cv2.putText(image, text, position, font, scale, border_color, border_thickness, cv2.LINE_AA)
    cv2.putText(image, text, position, font, scale, color, thickness, cv2.LINE_AA)

# MediaPipe Pose model setup
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            print("Error: Failed to capture image.")
            break

        # Convert the image to RGB and process with MediaPipe
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Start total workout timer and first set timer
        if total_workout_start_time is None:
            total_workout_start_time = time.time()
            set_start_time = total_workout_start_time  # Start time for the first set

        # Extract pose landmarks
        try:
            landmarks = results.pose_landmarks.landmark

            # Get coordinates for left shoulder, elbow, wrist
            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

            # Get coordinates for right shoulder, elbow, wrist
            right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                              landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

            # Calculate angles for both elbows
            left_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
            right_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)

            # Visualize the angles with borders
            draw_text_with_border(image, str(int(left_angle)),
                                  tuple(np.multiply(left_elbow, [frame.shape[1], frame.shape[0]]).astype(int)),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, (0, 0, 0), 3)
            draw_text_with_border(image, str(int(right_angle)),
                                  tuple(np.multiply(right_elbow, [frame.shape[1], frame.shape[0]]).astype(int)),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, (0, 0, 0), 3)

            # Logic for counting reps, only if not waiting for the next set
            if not waiting_for_next_set:
                if left_angle > 160 and right_angle > 160:
                    stage = "down"
                if left_angle < 30 and right_angle < 30 and stage == 'down':
                    stage = "up"
                    
                    # Track rep duration
                    if rep_start_time is not None:
                        rep_durations.append(time.time() - rep_start_time)
                    rep_start_time = time.time()

                    # Calculate arm synchronization error
                    sync_error = abs(left_angle - right_angle)
                    arm_sync_errors.append(sync_error)

                    counter += 1

        except Exception as e:
            print(f"Error: {e}")

        # Draw UI components
        draw_rounded_box(image, (0, 0), (225, 73), (50, 50, 50), 5)
        draw_rounded_box(image, (10, 10), (215, 63), (255, 255, 255), 2)
        draw_text_with_border(image, 'REPS', (30, 25), cv2.FONT_HERSHEY_DUPLEX, 0.4, (255, 255, 255), 1, (0, 0, 0), 2)
        draw_text_with_border(image, str(counter), (30, 55), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, (0, 0, 0), 2)
        draw_text_with_border(image, 'STAGE', (100, 25), cv2.FONT_HERSHEY_DUPLEX, 0.4, (255, 255, 255), 1, (0, 0, 0), 2)
        draw_text_with_border(image, stage, (100, 55), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, (0, 0, 0), 2)

        # Sets completed display
        draw_text_with_border(image, 'Sets Completed:', (10, 90), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1, (0, 0, 0), 2)
        draw_text_with_border(image, str(sets_completed), (160, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1, (0, 0, 0), 2)

        # Progress bar
        bar_top_left = (550, 50)
        bar_bottom_right = (590, 450)
        bar_height = 400
        fill_height = int(bar_height * (counter / target_reps))
        draw_rounded_box(image, bar_top_left, bar_bottom_right, (50, 50, 50), 5)
        cv2.rectangle(image, (bar_top_left[0], bar_bottom_right[1] - fill_height),
                      bar_bottom_right, (255, 255, 255), -1)
        progress_percentage = int((counter / target_reps) * 100)
        draw_text_with_border(image, f'Set progress {progress_percentage}%', (bar_top_left[0] - 60, bar_top_left[1] - 20),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, (0, 0, 0), 3)

        # Check if one set is completed
        if counter >= target_reps and not waiting_for_next_set:
            sets_completed += 1
            waiting_for_next_set = True
            display_set_completed = True

            # Calculate set duration and start next set timer
            if set_start_time is not None:
                set_duration = time.time() - set_start_time
                set_durations.append(set_duration)
                print(f"Set {sets_completed} duration: {set_duration:.2f} seconds")
            set_start_time = time.time()  # Reset set start time for next set

            counter = 0  # Reset reps for the next set

        # Display "Set Completed!" notification
        if display_set_completed:
            draw_text_with_border(image, 'Set Completed!', (30, 150), cv2.FONT_HERSHEY_DUPLEX, 0.7,
                                  (255, 255, 255), 2, (0, 0, 0), 3)

        # Show the video feed
        cv2.imshow('Fitness Form Corrector', image)

        # Key press handling
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):  # Start a new set
            waiting_for_next_set = False
            display_set_completed = False
            set_start_time = time.time()

    # Calculate final metrics after the workout
    total_workout_duration = time.time() - total_workout_start_time if total_workout_start_time else 0
    average_rep_duration = np.mean(rep_durations) if rep_durations else 0
    average_set_duration = np.mean(set_durations) if set_durations else 0
    average_sync_error = np.mean(arm_sync_errors) if arm_sync_errors else 0

    # Display final metrics
    print(f"\nWorkout Summary:")
    print(f"Total workout duration: {total_workout_duration:.2f} seconds")
    print(f"Total sets completed: {sets_completed}")
    print(f"Average time per set: {average_set_duration:.2f} seconds")
    print(f"Average time per rep: {average_rep_duration:.2f} seconds")
    print(f"Average arm sync error: {average_sync_error:.2f} degrees")

cap.release()
cv2.destroyAllWindows()
