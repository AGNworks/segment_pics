from moduls.segmentation import *
import numpy as np 
import time
from PIL import Image 
import cv2

img_w = 240
img_h = 320
                
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
time.sleep(5)

success, frame = camera.read()
print(success)
img_test = cv2.resize(frame, (img_h,img_w))
cv2.imwrite('x_test/test.jpg', img_test)

print('Type of picture:', type(img_test)) 

predict_segments = segmentpics(img_test)

#save results to file
for i in range(len(predict_segments)):
    im = Image.fromarray(predict_segments[i])
    im.save(f'results/{i}.png')