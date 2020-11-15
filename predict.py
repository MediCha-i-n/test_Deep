import Unet
import numpy as np
import cv2
def predict(img):
    model = Unet.Unet()
    model.load_weights("best_model.hdf5")
    shape = img.shape

    img = cv2.resize(img, (256, 256)).reshape(1, 256, 256, 3)
    output = model.predict(img)
    output = np.around(output)

    output = output.reshape(256, 256)

    output = cv2.resize(output, (shape[1], shape[0]), interpolation=cv2.INTER_CUBIC)

    return output*255
