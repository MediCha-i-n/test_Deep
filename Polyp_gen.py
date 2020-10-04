import glob
import cv2
import glob
from sklearn.feature_extraction.image import extract_patches
from itertools import cycle
import numpy as np
import os

class Generator():
    def __init__(self, patch_size, batch_size, split = 0.1):
        self.files = glob.glob("./../polyps/data/*")
        self.length = len(self.files)
        self.train_len = int(self.length*(1-split))
        self.train_files = self.files[:self.train_len]
        self.val_files = self.files[self.train_len:]
        self.batch_size = batch_size
        self.patch_size = patch_size

    def train_gen(self):
        train_files = cycle(self.train_files)

        data = np.zeros((0, self.patch_size, self.patch_size, 3))
        label = np.zeros((0, self.patch_size, self.patch_size, 1))

        while 1:

            while data.shape[0] < self.batch_size:
                file = next(train_files)

                img, annot = self.patches(file)

                data = np.append(data, img, axis = 0)
                label = np.append(label, annot, axis = 0)

            x = data[:self.batch_size, :, :, :]
            y = label[:self.batch_size, :, :, :]

            data = data[self.batch_size:, :, :, :]
            label= label[self.batch_size:, :, :, :]

            yield x, y

    def val_gen(self):
        val_files = cycle(self.val_files)

        data = np.zeros((0, self.patch_size, self.patch_size, 3))
        label = np.zeros((0, self.patch_size, self.patch_size, 1))

        while 1:
            if data.shape[0] == 0:
                file = next(val_files)

                img, annot = self.patches(file)

                data = np.append(data, img, axis = 0)
                label = np.append(label, annot, axis = 0)
            x = data[0, :, :, :].reshape(-1, self.patch_size, self.patch_size, 3)
            y = data[0, :, :, :].reshape(-1, self.patch_size, self.patch_size, 1)

            data = data[1:, :, :, :]
            label = label[1:, :, :, :]

            yield x, y


    def patches(self, file):
        l_file = os.path.dirname(file).replace("data", "label/") + \
                 os.path.basename(file).replace("tif", "png")
        img = cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2RGB)/255.0
        annot = cv2.imread(l_file, cv2.IMREAD_GRAYSCALE)/255.0

        img = extract_patches(img, (self.patch_size, self.patch_size, 3), 256). \
            reshape(-1, self.patch_size, self.patch_size, 3)
        annot = extract_patches(annot, (self.patch_size, self.patch_size), 256). \
            reshape(-1, self.patch_size, self.patch_size, 1)
        return img, annot