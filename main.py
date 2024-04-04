from matplotlib import pyplot as plt
from matplotlib import image as mpimg
from PIL import Image
import numpy as np
import nibabel as nib

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

train_patients = list(range(6))
validation_patients = list(range(6, 8))
test_patients = list(range(8, 10))
def normalisation(img):
    img_positives = img[img > 0]
    mean = np.mean(img_positives)
    std = np.std(img_positives)
    return (img - mean) / (5 * std)

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

SEUIL_AIRE = 100

X_train = []
Y_train = []
num_coupes = 0

for numpatient in train_patients:
    VT = nib.load(f"subject-{numpatient + 1}-label.img").get_fdata()
    T1 = normalisation(nib.load(f"subject-{numpatient + 1}-T1.img").get_fdata())
    T2 = normalisation(nib.load(f"subject-{numpatient + 1}-T2.img").get_fdata())

    sx, sy, sz, _ = VT.shape
    for z in range(sz):
        aire = np.sum(np.where(VT[:, :, z] > 0, 1, 0))
        if aire > SEUIL_AIRE:
            X_train.append([T1[:, :, z], T2[:, :, z]])
            Y_train.append(np.where(VT[:, :, z] > 0, 1, 0))
            num_coupes += 1

X_train = np.array(X_train)
Y_train = np.array(Y_train)

X_validation = []
Y_validation = []
num_coupes = 0

for numpatient in validation_patients:
    VT = nib.load(f"subject-{numpatient + 1}-label.img").get_fdata()
    T1 = normalisation(nib.load(f"subject-{numpatient + 1}-T1.img").get_fdata())
    T2 = normalisation(nib.load(f"subject-{numpatient + 1}-T2.img").get_fdata())

    sx, sy, sz, _ = VT.shape
    for z in range(sz):
        aire = np.sum(np.where(VT[:, :, z] > 0, 1, 0))
        if aire > SEUIL_AIRE:
            X_validation.append([T1[:, :, z], T2[:, :, z]])
            Y_validation.append(np.where(VT[:, :, z] > 0, 1, 0))
            num_coupes += 1

X_validation = np.array(X_validation)
Y_validation = np.array(Y_validation)

print(num_coupes)