#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by LazzySquirrel on 2017/12/24
import csv
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sklearn.utils

from keras.layers import Flatten, Dense, Lambda
from keras.layers import Conv2D, AveragePooling2D
from keras.layers import Cropping2D
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from random import shuffle

import matplotlib.image as mping

train_data_path = './data'
train_img = train_data_path + '/IMG/'

epochs = 3
batch_size = 32

# image correction
correction = 0.08

lines_buffs = []
with open(os.path.join(train_data_path, 'driving_log.csv'), 'r') as csvfile:
    reader = csv.reader(csvfile)

    for _ in reader:
        lines_buffs.append(_)

    del lines_buffs[0]

output_steerings = []
for buff in lines_buffs:
    output_steerings.append(float(buff[3]))

lines = []
output_values_steering = []
center_counter1 = 0
for line in lines_buffs:
    center_value = float(line[3])
    if center_value < 0.0 or center_value > 0.2:
        output_values_steering.append(center_value)
        lines.append(line)
    if center_value >= 0.0 or center_value <= 0.2:
        if center_counter1 >= 4:
            output_values_steering.append(center_value)
            lines.append(line)
            center_counter1 = 0
        center_counter1 += 1
#
#
# def preprocess_image(img):
#     ret_img = cv2.GaussianBlur(img, (5, 5), 0)
#     return cv2.cvtColor(ret_img, cv2.COLOR_BGR2YUV)


def image_flip(image, steeing):
    image, steeing = np.fliplr(image), -steeing
    return image, steeing


class Model(object):
    def __init__(self, input_shape, keep_prob=0.5):
        self.model = Sequential()
        self.input_shape = input_shape
        self.keep_prob = keep_prob

    def normal(self):
        self.model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=self.input_shape))
        self.model.add(Cropping2D(cropping=((60, 20), (0, 0))))

    def lexnet(self):
        self.normal()
        self.model.add(Conv2D(6, 5, 5, activation='relu'))
        self.model.add(AveragePooling2D())
        self.model.add(Conv2D(16, 5, 5, activation='relu'))
        self.model.add(AveragePooling2D())
        self.model.add(Flatten())
        self.model.add(Dense(120))
        self.model.add(Dense(84))
        self.model.add(Dense(1))

        self.model.compile(optimizer='adam', loss='mse')
        return self.model

    def nvidial_net(self):
        self.normal()
        self.model.add(Conv2D(24, 5, 5, subsample=(2, 2), activation='relu'))
        self.model.add(Conv2D(36, 5, 5, subsample=(2, 2), activation='relu'))
        self.model.add(Conv2D(48, 5, 5, subsample=(2, 2), activation='relu'))
        self.model.add(Conv2D(64, 3, 3, activation='relu'))
        self.model.add(Conv2D(64, 3, 3, activation='relu'))
        self.model.add(Flatten())
        self.model.add(Dense(100))
        self.model.add(Dense(50))
        self.model.add(Dense(10))
        self.model.add(Dense(1))

        self.model.compile(optimizer='adam', loss='mse')
        return self.model


def generator(lines, batch_size=32):
    """
    生成器, 生成数据每批次的返回
    :param lines: 总的图片的数量
    :param batch_size: 批次的大小,
    :return:
    """
    num_line = len(lines)
    while True:
        shuffle(lines)
        for offset in range(0, num_line, batch_size):
            batch_lines = lines[offset:offset+batch_size]

            images = []
            angles = []
            for batch_line in batch_lines:
                for i in range(3):
                    source_path = batch_line[i]
                    file_name = source_path.split('/')[-1]
                    current_path = './data/IMG/' + file_name
                    image = mping.imread(current_path)
                    # image = preprocess_image(image)
                    if i == 2:
                        posion_num = -1.0
                    else:
                        posion_num = i

                    angle = float(batch_line[3]) + posion_num * correction
                    images.append(image)
                    angles.append(angle)
                    image, angle = image_flip(image, angle)
                    images.append(image)
                    angles.append(angle)

            x_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(x_train, y_train)


train_lines, validation_lines = train_test_split(lines, test_size=0.3)

train_generator = generator(train_lines, batch_size=batch_size)
validation_generator = generator(validation_lines, batch_size=batch_size)


# train
input_shape = (160, 320, 3)
model = Model(input_shape).nvidial_net()

history_object = model.fit_generator(train_generator,
                                     samples_per_epoch=6*len(train_lines),
                                     validation_data=validation_generator,
                                     nb_val_samples=6*len(validation_lines),
                                     nb_epoch=epochs,
                                     verbose=1
                                     )

plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['train_set', 'validation loss'], loc='upper right')
plt.show()

model.save('model.h5')
exit()


if __name__ == '__main__':
    pass
