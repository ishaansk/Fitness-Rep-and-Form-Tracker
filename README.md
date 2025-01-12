AI-Powered Fitness Form and Rep Tracker A Python-based fitness assistant that uses computer vision to track body movements, correct form, and count repetitions in real-time.
This tool is designed to help users improve their workout techniques while maintaining proper form to avoid injury and maximize results.

Features
Real-time pose detection: Utilizes OpenCV and MediaPipe to track body movements during exercises. 
Form correction: Detects improper form and provides feedback to improve accuracy and avoid injury. 
Repetition counter: Automatically counts the number of repetitions performed for each exercise. 
User interface: Simple UI that displays the current exercise, repetition count, and motivational feedback. 
Technologies Used 
Python 
OpenCV: For live video capture and image processing. 
MediaPipe: For body pose detection and keypoint tracking. 
NumPy: For handling numerical operations and calculations.

To incorporate different exercises:
For Shoulder Press :-
Landmarks : LEFT_HIP, LEFT_SHOULDER, LEFT_ELBOW, RIGHT_HIP, RIGHT_SHOULDER, RIGHT_ELBOW
Angles : 
  Down : >165
  Up : <80

For Lateral Raises :-
Landmarks : LEFT_HIP, LEFT_SHOULDER, LEFT_ELBOW, RIGHT_HIP, RIGHT_SHOULDER, RIGHT_ELBOW
Angles : 
  Down : >75
  Up : <20

For Bicep Curls :-
Landmarks : LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST, RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST
Angles :
  Down : 160
  Up : 30
