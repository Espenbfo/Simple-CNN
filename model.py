from keras.models import Sequential
from keras.layers import Conv2D, Dense, BatchNormalization, LeakyReLU,Flatten, Dropout
from keras.optimizers import Adam

def model(size = 64,lr = 0.0001):
    m = Sequential()


    m.add(Dense(1,activation="sigmoid"))
    
    opt = Adam(lr)

    m.compile(opt,"binary_crossentropy",["accuracy"])

    return m