#!/usr/bin/python3

import tensorflow as tf
from matplotlib import pyplot
import numpy as np

import os
import json

import cv2
import time


models = {}
histories = {}
class_names = ["Access Granted","Access Denied", "Access Denied","Access Denied"]
for name in ["CNN", "MLP"]:
    model_path = "saved_models/"+name+".h5"
    history_path = "saved_histories/"+name+".json"
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
    cam = cv2.VideoCapture(0)

    cv2.namedWindow("Take a Picture")

    while True:
        ret, frame = cam.read()
        if not ret:
            print("failed to grab frame")
            break
        cv2.imshow("Take a Picture", frame)

        k = cv2.waitKey(1)
        if k%256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
        elif k%256 == 32:
            # SPACE pressed
            cv2.imwrite("faces/temp.jpg", frame)
            access, text = predict()
            text_color = (0,0,255)
            if(access):
                text_color = (255,0,0)
            add_text(frame, text, (int(cam.get(3)/2), int(cam.get(4)/2)), text_color)
            cv2.imshow("Take a Picture", frame)
            cv2.waitKey(0)

    cam.release()
    cv2.destroyAllWindows()

def predict():
    out = ""
    access = False

    img = tf.keras.utils.load_img("faces/temp.jpg", target_size=(50, 50))

    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    for name in models:
        model = models[name]
        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])

        print("{} predicts {} ({:.2f}% confidence)\n".format(name, class_names[np.argmax(score)], 100 * np.max(score)))
        if(name == "CNN"):
            out = class_names[np.argmax(score)]
            access = (np.argmax(score) == 1)

    return access, out

take_picture()
