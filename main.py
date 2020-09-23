import numpy as np
from keras.preprocessing import image
import cv2
import os
from model import model
from load_images import load_images

EPOCHS = 10
BATCH_SIZE=256
#Input folder 1
in1 = "pizza"

#Input folder 2
in2 = "burgers"


images,labels = load_images([in1,in2])

print("antall bilder", len(images))
print("antall labels", len(labels))
print("Fordeling av labels", labels.mean())


m = model(64)
m.fit(images,labels,BATCH_SIZE,EPOCHS,validation_split=0.1)

os.makedirs("weights",exist_ok=True)
m.save_weights(os.path.join("weights","file.h5"))