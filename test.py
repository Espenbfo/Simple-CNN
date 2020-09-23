from model import model
from keras.preprocessing import image
import os
import random
import cv2
import numpy as np

images = 10

in1 = "pizza"
in2 = "burgers"
weight = "weights/file.h5"

links = [os.path.join(in1,path) for path in os.listdir(in1)] + [os.path.join(in2,path) for path in os.listdir(in2)]
links = random.sample(links,images)

images = np.array([image.img_to_array(image.load_img(link))/127.5-1 for link in links])

m = model()
m.load_weights(weight)
predicts = m.predict(images)

for i,img in enumerate(images):
    print(predicts[i])
    cv2.imshow("1",cv2.resize(cv2.cvtColor(img/2+0.5,cv2.COLOR_RGB2BGR),(256,256)))
    cv2.waitKey(0)