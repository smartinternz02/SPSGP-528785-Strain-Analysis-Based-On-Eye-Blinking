from scipy.spatial import distance as dist
from imutils.video import FileVideoStream, VideoStream
from imutils import face_utils
import argparse
import imutils
import time
import dlib
import datetime
import cv2
from gtts import gTTS
import tkinter as tk
from tkinter import ttk
from pygame import mixer


def palyaudio(text1):
    speech = gTTS(text=text1, lang='en', slow=False)
    print(type(speech))
    # speech.save("output1.mp3")
    mixer.init()
    mixer.music.load("output1.mp3")
    mixer.music.set_volume(0.7)
    mixer.music.play()
    return


LARGE_FONT = ("Verdana", 12)
NORM_FONT = ("Helvetica", 10)
SMALL_FONT = ("Helvetica", 8)


def popupmsg(msg):
    popup = tk.Tk()
    popup.wm_title("Urgent")
    style = ttk.Style(popup)
    style.theme_use('classic')
    style.configure('Test.TLabel', background='aqua')
    label = ttk.Label(popup, text=msg, style='Test.TLabel', width=50)
    label.pack(side='top', fill="x", pady=10,)
    B1 = ttk.Button(popup, text="Okay", command=popup.destroy)
    B1.pack()
    popup.mainloop()


def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A+B)/(C)
    return ear


def mark_eyeLandmark(img, eyes):
    for eye in eyes:
        pt1, pt2 = (eye[1], eye[5])
        pt3, pt4 = (eye[0], eye[3])
        cv2.line(img, pt1, pt2, (200, 00, 0), 2)
        cv2.line(img, pt3, pt4, (200, 0, 0), 2)
    return img


ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
                help="path to facial landmark predictor")
ap.add_argument("-v", "--video", type=str, default="",
                help="path to input video file")
args = vars(ap.parse_args())

EYE_AR_THRESH = 0.5
EYE_AR_CONSEC_FRAMES = 2
COUNTER = 0
TOTAL = 0

print("[INFO] loading facial landmark predictor")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])
print(type(predictor), predictor)

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

eye_thresh = 10
before = datetime.datetime.now().minute

if not args.get("video", False):
    print("[INFO] starting video stream...")
    vs = VideoStream(src=0).start()
    time.sleep(1.0)
else:
    print("[INFO] opening video file...")
    vs = cv2.VideoCapture(args["video"])
    time.sleep(1.0)

while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray)
    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        for lm in shape:
            cv2.circle(frame, (lm), 3, (10, 2, 200))
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        img = frame.copy()
        img = mark_eyeLandmark(img, [leftEye, rightEye])
        ear = (leftEAR+rightEAR)/2

    # leftEyeHull = cv2.convexHull(leftEye)
    # rightEyeHull = cv2.convexHull(rightEye)
    # cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
    # cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        if ear < EYE_AR_THRESH:
            COUNTER += 1
        else:
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                TOTAL += 1
            else:
                COUNTER = 0

        now = datetime.datetime.now().minute
        no_of_min = now - before
        print(no_of_min, before, now)
        blinks = no_of_min*eye_thresh

        if (TOTAL < blinks - eye_thresh):
            palyaudio(
                "Take rest for a While as your blink count is less than average blinks")
            popupmsg("Take rest for a while!!!:D")
            cv2.putText(frame, "Take rest for a while!!!:D", (70, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        elif (TOTAL > blinks+eye_thresh):
            palyaudio(
                "Take rest for a While as your blink count is less than average blinks")
            popupmsg("Take rest for a while!!!:D")
            cv2.putText(frame, "Take rest for a while!!!:D", (70, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.putText(frame, "Blinks:{}".format(TOTAL), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "EAR:{}".format(ear), (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
cv2.destroyAllWindows()
vs.stop()
