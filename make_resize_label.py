import cv2
import glob
import os
import numpy as np
import keras
import tensorflow as tf
from tensorflow.python.client import device_lib
from keras import backend


def sample_resize_label():
    path = './drive/MyDrive/test/datasets/label'
    save_path = './drive/MyDrive/test/datasets/label'
    imgs = os.listdir(path)
    imgs = sorted(imgs)
    for img in imgs:
        image = cv2.imread(os.path.join(path, img))
        image = cv2.resize(image,(256,256))
        cv2.imwrite(os.path.join(save_path,img),image)

sample_resize_label()