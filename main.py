from matplotlib import pyplot as plt
from matplotlib import image as mpimg
from PIL import Image
import numpy as np
import nibabel as nib
import tensorflow

from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
import skimage.io as io
import skimage.transform as trans

import os
path = "/home/charles/PycharmProjects/pythonProject/iseg"
os.chdir("/home/charles/PycharmProjects/pythonProject/iseg")

def fastdisplay(*img):
    plt.figure(figsize=(16,8))
    nbimg = len(img)
    cols = min(9, nbimg)
    rows = (nbimg // cols) + 1
    for ii, img2d in enumerate(img):
        plt.subplot(rows, cols, ii+1)
        plt.imshow(img2d)
    plt.show()

def printSlices(img):
    sx, sy, sz, _ = img.shape
    fastdisplay(img[sx//2, :, :, 0], img[:, sy//2, :, 0], img[:, :, sz//2, 0])


def normalisation(img):
    img_positives = img[img > 0]
    mean = np.mean(img_positives)
    std = np.std(img_positives)
    return (img - mean) / (5 * std)
def step1_2():
    for dirname, _, filenames in os.walk(path):
        for filename in filenames:
            if "label" in filename and "hdr" not in filename:
                print(filename)
                nomT1 = filename[:-10]+"-T1.img"
                T1 = nib.load(nomT1).get_fdata()
                print("t1 " + nomT1)
                T1 = normalisation(T1)
                printSlices(T1)
                nomT2 = filename[:-10]+"-T2.img"
                T2 = nib.load(nomT2).get_fdata()
                print("t2 " + nomT2)
                T2 = normalisation(T2)
                printSlices(T2)
                VT = nib.load(filename).get_fdata()
                print("VT " + filename)
                printSlices(VT)

def algo_step3(patients):
    SEUIL_AIRE = 100

    num_coupes = 0

    for numpatient in patients:
        VT = nib.load(f"subject-{numpatient + 1}-label.img").get_fdata()

        sx, sy, sz, _ = VT.shape
        for z in range(sz):
            aire = np.sum(np.where(VT[:, :, z] > 0, 1, 0))
            if aire > SEUIL_AIRE:
                num_coupes += 1

    X = np.zeros((num_coupes, sx, sy, 2))
    Y = np.zeros((num_coupes, sx, sy, 1))
    #note a moi meme je me suis trompé en reutilisant num coupe comme indice
    #j'ai aussi compris que mon algo de base était faux et que je devais repasser une nouvelle fois l'image afin de remplir xtrain
    num_valid_coupe = 0
    for numpatient in patients:
        VT = nib.load(f"subject-{numpatient + 1}-label.img").get_fdata()
        T1 = normalisation(nib.load(f"subject-{numpatient + 1}-T1.img").get_fdata())
        T2 = normalisation(nib.load(f"subject-{numpatient + 1}-T2.img").get_fdata())
        sx, sy, sz, _ = VT.shape
        for z in range(sz):
            aire = np.sum(np.where(VT[:, :, z] > 0, 1, 0))
            if aire > SEUIL_AIRE:
                X[num_valid_coupe, :, :, 0] = T1[:, :, z, 0]
                X[num_valid_coupe, :, :, 1] = T2[:, :, z, 0]
                Y[num_valid_coupe, :, :, 0] = VT[:, :, z, 0]
                num_valid_coupe += 1
    return X, Y, num_valid_coupe
def step3():

    train_patients = list(range(6))
    validation_patients = list(range(6, 8))
    test_patients = list(range(8, 10))


    X_train, Y_train, num_coupes = algo_step3(train_patients)
    print(num_coupes)
    X_validation, Y_validation, num_coupes = algo_step3(validation_patients)
    print(num_coupes)
    return X_train, Y_train, X_validation, Y_validation


def unet(pretrained_weights=None, input_size=(256, 256, 1)):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(drop5))
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

    model = Model(inputs=inputs, outputs=conv10)

    model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

    #model.summary()

    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    return model
def step4():
    return
if __name__ == '__main__':
    #step1_2()
    Xtrain, Ytrain, Xval, Yval = step3()

    #print(Xtrain.shape, Ytrain.shape)
    #Xtrain = Xtrain.reshape((-1, 144, 192, 1))
    #print(Xtrain.shape, Ytrain.shape)
    #print(Xtrain)

    # On double le nombre d'échantillon de Ytrain car leur valeur sont 0 dans tous les cas
    #Ytrain = np.repeat(Ytrain, 2, axis=0)
    #print(Ytrain)
    #model = unet(input_size=(144, 192, 1))
    #model.fit(Xtrain, Ytrain, batch_size=8, epochs=10, validation_data=(Xval, Yval))