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
    def __init__(self, patch_size, batch_size, identity):
        self.files = muterun_js("./application/dataQuery.js", identity).stdout.decode('utf-8').split("\n")[7:-2]
        self.whole_size = int(muterun_js("./application/getTotalEnrollCount.js", identity).stdout.decode('utf-8').split("\n")[-2])
        self.train_img = []
        self.train_mask = []
        self.preprocess()
        self.length = len(self.train_img)
        self.train = cycle(zip(self.train_img, self.train_mask))
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.patch_length = self.length

    def preprocess(self):
        for i in range(len(self.files)):
            if i % 2 == 0:
                self.train_img.append(base64.b64decode(self.files[i]))
            else:
                self.train_mask.append(base64.b64decode(self.files[i]))
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

            data = data[self.batch_size:, :, :, :]
            masks = masks[self.batch_size:, :, :, :]

            yield x, y

    def get_patches(self, img, mask):
        img = np.frombuffer(img, np.uint8)
        mask = np.frombuffer(mask, np.uint8)

        img = cv2.imdecode(img, cv2.IMREAD_COLOR)
        mask = cv2.imdecode(mask, cv2.IMREAD_GRAYSCALE)
        _, mask = cv2.threshold(mask, 100, 255, cv2.THRESH_BINARY)
        mask = (mask/255.0).astype('uint8')

        img = cv2.resize(img, (self.patch_size, self.patch_size))
        mask = cv2.resize(mask, (self.patch_size, self.patch_size))

        img = img.reshape(-1, img.shape[0], img.shape[1], img.shape[2])
        mask = mask.reshape(-1, mask.shape[0], mask.shape[1], 1)

        return img, mask



