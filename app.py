from flask import Flask, render_template, Response
import cv2
import copy
from model import KeyPointClassifier
from app_files import calc_landmark_list, draw_info_text, draw_landmarks, pre_process_landmark, calc_bounding_rect, draw_sentence
import mediapipe as mp
import csv

app = Flask(__name__)

app.static_folder = 'static'
app.template_folder = 'templates'

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

cap = cv2.VideoCapture(0)  # Use the default camera (you may need to adjust this)

keypoint_classifier = KeyPointClassifier()

# Load keypoint classifier labels
with open('model/keypoint_classifier/keypoint_classifier_label.csv', encoding='utf-8-sig') as f:
    keypoint_classifier_labels = csv.reader(f)
    keypoint_classifier_labels = [row[0] for row in keypoint_classifier_labels]

result_rect = (0, 480, 640, 720)  # Adjust the coordinates as needed

detected_labels = []  # List to store detected labels

@app.route('/')
def index():
    if detected_labels:
        recent_label = detected_labels[-1]
    else:
        recent_label = "No label detected"  # Set a default value if no label is detected

    return render_template('index.html', result_rect=result_rect, recent_label=recent_label)

@app.route('/language')
def language():
    return render_template('language.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/learn')
def learn():
    return render_template('learn.html')

def generate_frames():
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        debug_frame = copy.deepcopy(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frame.flags.writeable = False
        results = hands.process(frame)
        frame.flags.writeable = True

        debug_frame = draw_sentence(debug_frame, [])

        # Reset detected_label for each frame
        detected_label = None

        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                landmark_list = calc_landmark_list(debug_frame, hand_landmarks)
                pre_processed_landmark_list = pre_process_landmark(landmark_list)
                brect = calc_bounding_rect(debug_frame, hand_landmarks)
                hand_sign_id, probab = keypoint_classifier(pre_processed_landmark_list, handedness.classification[0].label[0:])
                pr = float('%.3f' % float(probab))
                p = str(pr)
                debug_frame = draw_landmarks(debug_frame, landmark_list)
                flag = 1

                # Draw information with gesture label
                label = keypoint_classifier_labels[hand_sign_id]
                debug_frame = draw_info_text(brect, debug_frame, handedness, label, p, flag)

                # Store the detected label for displaying in the rectangle
                detected_label = label

        # Store the detected label in the list
        detected_labels.append(detected_label)

        # Display the most recent detected label on the frame
        if detected_labels:
            recent_label = detected_labels[-1]
        cv2.putText(debug_frame, f'Detected Object: {recent_label}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        
        _, buffer = cv2.imencode('.jpg', debug_frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

all_detected_labels = [] 

@app.route('/latest_label')
def latest_label():
    global all_detected_labels  # Declare the variable as global

    if detected_labels:
        recent_label = detected_labels[-1]
        all_detected_labels.append(recent_label)  # Append the recent label to the list
    else:
        recent_label = "No label detected"
        all_detected_labels = ["No label detected"]  # Initialize as a list if no label is detected

    return {'recent_label': recent_label, 'all_detected_labels': all_detected_labels}

if __name__ == '__main__':
    app.run(debug=True,port=5001)
