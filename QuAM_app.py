import numpy as np
import cv2
from skimage.color import rgb2gray

import time
from skimage.transform import resize

import keras



model = keras.models.load_model('./CNN_model')

cap = cv2.VideoCapture(0)
width  = cap.get(3)
height = cap.get(4)
left = int(width/2 - 128)
right = int(width/2 + 128)
top = int(height/2 - 128)
bot = int(height/2 + 128)

frame_rate = 10
prev = 0
class_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W'
                , 'X', 'Y', 'Z']

while True:
    ret, frame = cap.read()
    time_elapsed = time.time() - prev
    
    if time_elapsed > 1./frame_rate:
        prev = time.time()
        
        
        
        image = frame[top:bot, left:right]
        image = np.array(resize(rgb2gray(image), (128,128)).astype(np.float32))

        pred_probs = model.predict(np.array([image]))
        pred_class = np.argmax(pred_probs,axis=1)
        pred_prob = pred_probs[0, pred_class]

        
        
        cv2.putText(frame, class_labels[pred_class[0]], (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.rectangle(frame, (left, top), (right, bot), (0, 255, 0), 2)
        cv2.imshow('Video', frame)

    if cv2.waitKey(1) == ord('q'):
        cap.release()
        break