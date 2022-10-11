from tensorflow.keras.models import Model , load_model
from tensorflow.keras.preprocessing import image 
import numpy as np 
import time, random, os
from PIL import Image 

way = (255,255,255)
backg = (0,0,0)

class_labels = (way, backg)

def labels_to_rgb(image_list):
    result = []

    for y in image_list:
        temp = np.zeros((img_w, img_h, 3), dtype='uint8')

        for i, cl in enumerate(class_labels):
            temp[np.where(np.all(y==i, axis=-1))] = class_labels[i]

        result.append(temp)
  
    return np.array(result)

#parameters of pictures used for input
img_w = 240
img_h = 320

#loading model
my_model = load_model('model_lin1.h5')

print("Ready to make segmanted pictures")

#importing test pictures from folder x_test and save them to list
image_list = []
for filename in sorted(os.listdir('x_test')):
    image_list.append(image.load_img(os.path.join('x_test', filename), target_size=(img_w, img_h)))                

print('Number of pictures:', len(image_list)) 

#converting images to numpy array
x_test = []
for img in image_list:
    x = image.img_to_array(img)
    x_test.append(x)

x_test = np.array(x_test)
print(x_test.shape)

#getting results as array
predict = np.argmax(my_model.predict(x_test), axis=-1)
predict_segments = labels_to_rgb(predict[..., None])


#save results to file
for i in range(len(predict_segments)):
    im = Image.fromarray(predict_segments[i])
    im.save(f'results/{i}.png')