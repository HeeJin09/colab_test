import cv2
import os
import numpy as np
import keras
import tensorflow as tf
from tensorflow.python.client import device_lib
from keras import backend

def image_blending():
    original_path = './drive/MyDrive/test/datasets/test_image'
    segmentation_path = './drive/MyDrive/test/datasets/result'

    img1 = cv2.imread(os.path.join(original_path,'input__2.jpg'))
    img2 = cv2.imread(os.path.join(segmentation_path, '2_predict.png'))
    img2[:, :, 1] = 0
    img2[:, :, 2] = 0
    a=0.7
    b=0.3
    dst = cv2.addWeighted(img1, a, img2, b, 0)
    # cv2.imshow('dst',dst)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    cv2.imwrite(os.path.join(segmentation_path, 'image_blending_image0.png'), dst)

image_blending()
# print(256/420)
# print(2048*256/420)
# print(1248/256)
# print(256*5)