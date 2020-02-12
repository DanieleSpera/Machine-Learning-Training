import keras
from keras import losses, models, optimizers
from keras.activations import relu, softmax
from keras.layers import (Convolution2D, BatchNormalization, Flatten, Dropout, MaxPool2D, Activation, Input, Dense)
class KerasCnnModel:
    ''' Create a standard deep 2D convolutional neural network'''
    def __init__(self):
        pass
    def load(self,n=30):
        nclass = 8
        inp = Input(shape=(n,345,1))  #2D matrix of 30 MFCC bands by 216 audio length.
        x = Convolution2D(32, (4,10), padding="same")(inp)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = MaxPool2D()(x)
        x = Dropout(rate=0.2)(x)
        
        x = Convolution2D(32, (4,10), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = MaxPool2D()(x)
        x = Dropout(rate=0.2)(x)
        
        x = Convolution2D(32, (4,10), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = MaxPool2D()(x)
        x = Dropout(rate=0.2)(x)
        
        x = Convolution2D(32, (4,10), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = MaxPool2D()(x)
        x = Dropout(rate=0.2)(x)
        
        x = Flatten()(x)
        x = Dense(64)(x)
        x = Dropout(rate=0.2)(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Dropout(rate=0.2)(x)
        
        out = Dense(nclass, activation=softmax)(x)
        model = models.Model(inputs=inp, outputs=out)
        
        opt = optimizers.Adam(0.001)
        model.compile(optimizer=opt, loss=losses.categorical_crossentropy, metrics=['acc'])
        return model