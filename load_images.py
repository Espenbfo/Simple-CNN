import numpy as np
from keras.preprocessing import image
import cv2
import os
import random
def load_images(input_folders,flip=True):
    images = []
    labels = []
    for flipping in range(2 if flip else 1):
        for i,folder in enumerate(input_folders):
            for img in os.listdir(folder):
                path = os.path.join(folder,img)
                if flipping:
                    images.append(cv2.flip(image.img_to_array(image.load_img(path)) / 127.5 - 1,1))
                else:
                    images.append(image.img_to_array(image.load_img(path))/127.5-1)

                labels.append(i)


    temp = list(zip(images,labels))
    random.shuffle(temp)
    images,labels = zip(*temp)
    images = np.array(images)
    labels = np.array(labels)
    return images,labels
