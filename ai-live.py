import cv2
import mediapipe as mp

def count_fingers(landmarks):
    thumb_up = landmarks[4].x > landmarks[5].x
    index_up = landmarks[8].y < landmarks[6].y
    middle_up = landmarks[12].y < landmarks[10].y
    ring_up = landmarks[16].y < landmarks[14].y
    pinky_up = landmarks[20].y < landmarks[18].y

    fingers = [thumb_up, index_up, middle_up, ring_up, pinky_up]
    return sum(fingers)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 600)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 500)

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hand = mp_hands.Hands()

while True:
    success, frame = cap.read()
    if success:
        RGB_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hand.process(RGB_frame)
        if result.multi_hand_landmarks:
            for hand_landmark in result.multi_hand_landmarks:
                num_fingers = count_fingers(hand_landmark.landmark)

                # Draw the hand landmarks on the frame
                mp_drawing.draw_landmarks(frame, hand_landmark, mp_hands.HAND_CONNECTIONS)

                # Display the number of fingers on the screen
                cv2.putText(frame, str(num_fingers) + " fingers", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255))

        cv2.imshow("image", frame)
        if (cv2.waitKey(1) == 27):
            break

cv2.destroyAllWindows()