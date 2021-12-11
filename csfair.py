#!/usr/bin/python3

MIN_CONFIDENCE=0.95
CAM_INDEX=0

import sys

import tensorflow as tf
from matplotlib import pyplot
import numpy as np

import os
import json

import cv2
import time

if(len(sys.argv) > 1):
	CAM_INDEX=int(sys.argv[1])

TEMP_FILE=".csfair-"+str(CAM_INDEX)+".tmp.jpg"


models = {}
histories = {}
class_names = ["Wearing Mask a Correctly Over Mouth and Nose", "Wearing Mask a Over Mouth, but not Over Nose", "Not Wearing a Mask Over Mount and Nose", "Not Wearing a Mask"]
for name in ["CNN", "MLP"]:
    model_path = os.path.join("saved_models", name+".h5")
    history_path = os.path.join("saved_histories", name+".json")
    histories[name] = {}
    if(os.path.exists(model_path) and os.path.exists(history_path)):
        models[name] = tf.keras.models.load_model(model_path)
        histories[name]["history"] = json.load(open(history_path, 'r'))

def add_text(img, text, center_pos, text_color):

    font = cv2.FONT_HERSHEY_PLAIN
    font_scale=3
    font_thickness=2
    text_color_bg=(0, 0, 0)


    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_width, text_height = text_size
    center_x, center_y = center_pos
    pos = (int(center_x-text_width/2), int(center_y-text_height/2 - font_scale - 1))
    x, y = pos
    cv2.rectangle(img, pos, (x + text_width, y + text_height), text_color_bg, -1)
    cv2.putText(img, text, (x, y + text_height + font_scale - 1), font, font_scale, text_color, font_thickness)

def take_picture():
    cam = cv2.VideoCapture(CAM_INDEX)

    cv2.namedWindow("AI Mask Detection")

    while True:
        ret, frame = cam.read()
        if not ret:
            print("Failed to grab frame")
            break
        display_frame = cv2.flip(frame, 1)
        cv2.imshow("AI Mask Detection", display_frame)

        k = cv2.waitKey(1)
        if k%256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
        elif k%256 == 32:
            # SPACE pressed
            access = False
            confidence = 0.0

            while(confidence <= MIN_CONFIDENCE):
                ret, frame = cam.read()
                if not ret:
                    print("failed to grab frame")
                    break
                img = cv2.resize(frame, (150,150))
                cv2.imwrite(TEMP_FILE, img)

                access, confidence = predict()

                display_frame = cv2.flip(frame, 1)
                add_text(display_frame, "Scanning...", (int(cam.get(3)/2), int(cam.get(4)/2)), (255,255,255))
                cv2.imshow("AI Mask Detection", display_frame)
                cv2.waitKey(1)

            text = "Access Denied"
            text_color = (0,0,255)
            if(access):
                text = "Access Granted"
                text_color = (0,255,0)

            display_frame = cv2.flip(frame, 1)
            add_text(display_frame, text, (int(cam.get(3)/2), int(cam.get(4)/2)), text_color)
            cv2.imshow("AI Mask Detection", display_frame)
            print("%s with a %.2f confidence." % (text, confidence * 100))
            cv2.waitKey(0)


    cam.release()
    cv2.destroyAllWindows()
    if os.path.exists(TEMP_FILE):
        os.remove(TEMP_FILE)

def predict():
    out = ()
    access = False

    img = tf.keras.utils.load_img(TEMP_FILE, target_size=(150, 150))

    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    for name in models:
        model = models[name]
        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])

        confidence = np.max(score)

        if(name == "CNN"):
            print("{} predicts {} ({:.2f}% confidence)".format(name, class_names[np.argmax(score)], 100 * np.max(score)))
            access = (np.argmax(score) == 0)
            out = access, confidence

    return out

take_picture()
