import glob
import cv2
import glob
from sklearn.feature_extraction.image import extract_patches
from itertools import cycle
import numpy as np
import os
from Naked.toolshed.shell import muterun_js
import base64


class Generator():
    def __init__(self, patch_size, batch_size):
        self.files = muterun_js("trainDataQuery.js", '\"trainer1\"').stdout.decode('utf-8').split('\n')
        self.preprocess()
        self.length = len(self.files)//2
        self.train_img = self.files[:self.length]
        self.train_mask = self.files[self.length:]
        self.train = cycle(zip(self.train_img, self.train_mask))
        self.patch_size = patch_size
        self.batch_size = batch_size
    def preprocess(self):
        for i in range(len(self.files)):
            self.files[i] = base64.b64decode(self.files[i])
    def generator(self):

        data = np.zeros((0, self.patch_size, self.patch_size, 3))
        masks = np.zeros((0, self.patch_size, self.patch_size, 1))

        while 1:
            while data.shape[0] < self.batch_size:
                img, mask = next(self.train)
                img, mask = self.get_patches(img, mask)

                data = np.append(data, img, axis = 0)
                masks = np.append(masks, mask, axis = 0)

            x = data[:self.batch_size, :, :, :]
            y = masks[:self.batch_size, :, :, :]

            data [self.batch_size, :, :, :]
            masks = masks[self.batch_size, :, :, :]

            yield x, y

    def get_patches(self, img, mask):
        img = np.frombuffer(img, np.uint8)
        mask = np.frombuffer(mask, np.uint8)

        img = cv2.imdecode(img, cv2.IMREAD_COLOR)/255.0
        mask = cv2.imdecode(mask, cv2.IMREAD_GRAYSCALE)/255.0

        img = extract_patches(img, (self.patch_size, self.patch_size, 3), 256). \
            reshape(-1, self.patch_size, self.patch_size, 3)
        mask = extract_patches(mask, (self.patch_size, self.patch_size), 256). \
            reshape(-1, self.patch_size, self.patch_size, 1)

        return img, mask



        


class train_Generator():
    def __init__(self, patch_size, batch_size, path, split = 0.1):
        self.files = glob.glob(path+"/data/*")
        self.length = len(self.files)
        self.train_len = int(self.length*(1-split))
        self.train_files = self.files[:self.train_len]
        self.val_files = self.files[self.train_len:]
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.whole_size = 1575

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
                 os.path.basename(file)
        img = cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2RGB)/255.0

        annot = cv2.imread(l_file, cv2.IMREAD_GRAYSCALE)/255.0


        img = extract_patches(img, (self.patch_size, self.patch_size, 3), 256). \
            reshape(-1, self.patch_size, self.patch_size, 3)

        annot = extract_patches(annot, (self.patch_size, self.patch_size), 256). \
        reshape(-1, self.patch_size, self.patch_size, 1)

        return img, annot