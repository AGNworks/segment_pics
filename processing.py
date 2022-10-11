from moduls.segmentation import *
import numpy as np 
import os
from PIL import Image 

img_w = 240
img_h = 320
                
#importing test picture from folder x_test
filenames = sorted(os.listdir('x_test'))
im_test = image.load_img(os.path.join('x_test', filenames[0]), target_size=(img_w, img_h))

print('Type of picture:', type(im_test)) 

predict_segments = segmentpics(im_test)

#save results to file
for i in range(len(predict_segments)):
    im = Image.fromarray(predict_segments[i])
    im.save(f'results/{i}.png')