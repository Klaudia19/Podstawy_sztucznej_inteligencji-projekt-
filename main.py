import os
import cv2 as cv
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers


def preprocess(cat, split, label):
    train_images = []
    train_labels = []
    for i in os.listdir(cat):
        image = cv.imread(cat + '/' + i)
        res = cv.resize(image, dsize=(128, 128), interpolation=cv.INTER_CUBIC)
        gray = cv.cvtColor(res, cv.COLOR_BGR2GRAY)
        train_images.append(gray)
        train_labels.append(label)

    size = len(train_images)
    return train_images[int(split * size):], train_images[:int(split * size)], train_labels[
                                                                               int(split * size):], train_labels[
                                                                                                    :int(split * size)]