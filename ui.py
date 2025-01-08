import pyttsx3
import csv
import copy
import cv2 as cv
from model import KeyPointClassifier
from app_files import calc_landmark_list, draw_info_text, draw_landmarks, pre_process_landmark, calc_bounding_rect, \
    draw_sentence
import mediapipe as mp
import numpy as np
from app_files import get_args

def speak_text(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

def main():
    args = get_args() 
    cap_device = args.device
    cap_width = args.width
    cap_height = args.height
    use_static_image_mode = args.use_static_image_mode
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=use_static_image_mode,
        max_num_hands=1,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    keypoint_classifier = KeyPointClassifier()
    instruction = "Press ESC for exit, DEL(clear all) BACKSPACE (one-word removal)"
    
    with open('model/keypoint_classifier/keypoint_classifier_label.csv', encoding='utf-8-sig') as f:
        keypoint_classifier_labels = csv.reader(f)
        keypoint_classifier_labels = [row[0] for row in keypoint_classifier_labels]
    
    consecutive_word = list()
    to_display_sentence = list()
    spoken_signs = set()  # Set to track spoken signs

    # Define rectangles for detection and result display
    detection_rect = (0, 0, cap_width, cap_height)
    result_rect = (0, cap_height, cap_width, 2*cap_height)

    while True:
        key = cv.waitKey(10)
        if key == 27:  # ESC
            break
        if key == 8:
            if len(to_display_sentence) >= 1:
                to_display_sentence.pop()
                speak_text("Backspace pressed")
        if key == 0:
            while len(to_display_sentence) >= 1:
                to_display_sentence.pop()
            speak_text("Delete pressed")

        ret, image = cap.read()
        if not ret:
            break
        blackboard = np.zeros((480, 640, 3), dtype=np.uint8)
        image = cv.flip(image, 1)
        debug_image = copy.deepcopy(image)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True
        blackboard = draw_sentence(blackboard, to_display_sentence)
        
        # Draw detection rectangle
        cv.rectangle(debug_image, (detection_rect[0], detection_rect[1]), (detection_rect[2], detection_rect[3]), (0, 255, 0), 2)

        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)
                pre_processed_landmark_list = pre_process_landmark(landmark_list)
                brect = calc_bounding_rect(debug_image, hand_landmarks)
                hand_sign_id, probab = keypoint_classifier(pre_processed_landmark_list, handedness.classification[0].label[0:])
                pr = float('%.3f' % float(probab))
                p = str(pr)
                debug_image = draw_landmarks(debug_image, landmark_list)
                flag = 1
                debug_image = draw_info_text(brect,
                                             debug_image,
                                             handedness,
                                             keypoint_classifier_labels[hand_sign_id], p, flag)
                # Draw result rectangle
                cv.rectangle(debug_image, (result_rect[0], result_rect[1]), (result_rect[2], result_rect[3]), (0, 0, 255), 2)
                
                if probab > 0.5:
                    sign_label = keypoint_classifier_labels[hand_sign_id]
                    consecutive_word.append(sign_label)

                    if len(consecutive_word) > 20:
                        def chkList(lst):
                            return len(set(lst)) == 1

                        lst = consecutive_word[-20:]
                        if chkList(lst):
                            if len(to_display_sentence) == 0:
                                to_display_sentence.append(lst[-1])

                            if to_display_sentence[-1] != lst[-1]:
                                to_display_sentence.append(lst[-1])

                            if sign_label not in spoken_signs:
                                speak_text(sign_label)
                                spoken_signs.add(sign_label)
                        else:
                            lst = []

                    blackboard = draw_sentence(blackboard, to_display_sentence)

        # size of rect
        cv.rectangle(debug_image, (0, 430), (640, 480), (0, 0, 0), -1) 
        # thickness=-1 
        cv.putText(debug_image, instruction, (0, 460), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)
        
        # Display detection rectangle
        cv.rectangle(debug_image, (detection_rect[0], detection_rect[1]), (detection_rect[2], detection_rect[3]), (0, 255, 0), 2)
        cv.imshow('Detection Rectangle', debug_image)

        # Display result rectangle
        cv.imshow('Result Rectangle', blackboard)

    cap.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()

