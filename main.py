# -*- coding:utf-8 -*-
from PIL import Image, ImageDraw
from model import Yolo_v1
from absl import flags
from random import random, shuffle, randint

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import math
import cv2

OBJECT_NAMES = ["person", "bird", "cat", "cow", "dog", "horse", "sheep", "aeroplane", "boat", "bus",
                   "car", "motorbike", "train", "bottle", "chair", "dining", "table", "potted plane", "sofa", "tv/monitor"]
seed = [randint(1, 1000) for i in range(3)]

flags.DEFINE_string("tr_img_path", "D:/[1]DB/[3]detection_DB/PascalVoc2012/pascal_voc_2012/VOC2012/JPEGImages", "Training image path")

flags.DEFINE_string("tr_txt_path", "D:/[1]DB/[3]detection_DB/PascalVoc2012/pascal_voc_2012/VOC2012/Annotations_text", "Training text path")

flags.DEFINE_integer("img_size", 448, "model input size")

flags.DEFINE_integer("batch_size", 16, "batch size")

flags.DEFINE_integer("output_size", 7, "")

flags.DEFINE_float("mini_threshold", 0.2, "")

flags.DEFINE_float("coord_lambda", 5.0, "")

flags.DEFINE_float("noObject_lambda", 0.5, "")

flags.DEFINE_float("lr", 0.0001, "Leanring rate")

FLAGS = flags.FLAGS
FLAGS(sys.argv)

optim = tf.compat.v1.train.MomentumOptimizer(FLAGS.lr, momentum=0.9)

def func_(image, label):

    list_ = image
    image = tf.io.read_file(image)
    image = tf.image.decode_jpeg(image, 3)
    shape = tf.shape(image)
    image = tf.image.resize(image, [FLAGS.img_size, FLAGS.img_size])

    image = tf.image.convert_image_dtype(image, tf.float32) / 255.  # -1 ~ 1
    #image = tf.image.per_image_standardization(image)

    return image, label, shape, list_

def read_label(file, batch_size):

    cell_h = FLAGS.img_size // FLAGS.output_size
    cell_w = FLAGS.img_size // FLAGS.output_size

    label = []
    responsibleGrid = np.zeros([FLAGS.batch_size, 7, 7, 25])
    for b in range(batch_size):
        f = open(tf.compat.as_bytes(file[b].numpy()), 'r')
        cla = []
        while True:
            line = f.readline()
            if not line: break
            line = line.split('\n')[0]

            xmin = int(int(line.split(',')[0]))
            xmax = int(int(line.split(',')[2]))
            ymin = int(int(line.split(',')[1]))
            ymax = int(int(line.split(',')[3]))
            height = int(line.split(',')[4])
            width = int(line.split(',')[5])
            classes = int(line.split(',')[6])

            # xmin, xmax에 대해서는 이게 순서가 뒤죽박죽인것도 있는거같아서
            # https://github.com/ivder/LabelMeYoloConverter/blob/master/convert.py 참고


            x = (xmin + xmax) / 2.0
            y = (ymin + ymax) / 2.0
            w = xmax - xmin
            h = ymax - ymin

            x = x * (1/width)
            w = w * (1/width)
            y = y * (1/height)
            h = h * (1/height)

            i, j = int(FLAGS.output_size * y), int(FLAGS.output_size * x)
            x_cell, y_cell = FLAGS.output_size * x - j, FLAGS.output_size * y - i
            width_cell = w * FLAGS.output_size
            height_cell = h * FLAGS.output_size

            boxData = [x_cell, y_cell, width_cell, height_cell]

            responsibleGridX = i
            responsibleGridY = j


            # 이부분 순서를 고쳐야한다! 우선은 이 인덱스 순서는 loss를 작성하고 진행하자
            responsibleGrid[b][responsibleGridX][responsibleGridY][classes] = 1    # class
            responsibleGrid[b][responsibleGridX][responsibleGridY][21:25] = boxData    # box
            responsibleGrid[b][responsibleGridX][responsibleGridY][20] = 1 # confidence


    responsibleGrid = np.array(responsibleGrid, dtype=np.float32)

    return responsibleGrid

def train_IOU(predict_box, label_box):

    box1_x1 = predict_box[..., 0:1] - predict_box[..., 2:3] / 2
    box1_y1 = predict_box[..., 1:2] - predict_box[..., 3:4] / 2
    box1_x2 = predict_box[..., 0:1] + predict_box[..., 2:3] / 2
    box1_y2 = predict_box[..., 1:2] + predict_box[..., 3:4] / 2

    box2_x1 = label_box[..., 0:1] - label_box[..., 2:3] / 2
    box2_y1 = label_box[..., 1:2] - label_box[..., 3:4] / 2
    box2_x2 = label_box[..., 0:1] + label_box[..., 2:3] / 2
    box2_y2 = label_box[..., 1:2] + label_box[..., 3:4] / 2

    intersect_x1 = tf.maximum(box1_x1, box2_x1)
    intersect_y1 = tf.maximum(box1_y1, box2_y1)
    intersect_x2 = tf.minimum(box1_x2, box2_x2)
    intersect_y2 = tf.minimum(box1_y2, box2_y2)

    intersect = tf.clip_by_value(intersect_x2 - intersect_x1, clip_value_min=0, clip_value_max=7) * \
                tf.clip_by_value(intersect_y2 - intersect_y1, clip_value_min=0, clip_value_max=7)

    box1_area = tf.abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = tf.abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    iou = intersect / (box1_area + box2_area - intersect + 1e-7)

    return iou

def save_fig(model, images, original_height, original_width):

    return image

def cal_loss(model, images, labels):

    # 모델의 최종 output은 1460 이 된다.
    # https://www.youtube.com/watch?v=n9_XyCGr-MI
    with tf.GradientTape() as tape:
        predict = model(images, True)   # [:, 1470]
        predict = tf.reshape(predict, [-1, FLAGS.output_size, FLAGS.output_size, 20 + 2*5]) # [:, 7, 7, 30]

        output_b1 = train_IOU(predict[..., 21:25], labels[..., 21:25])    # [:, 7, 7, 1]
        output_b2 = train_IOU(predict[..., 26:30], labels[..., 21:25])    # [:, 7, 7, 1]
        ious = tf.concat([output_b1, output_b2], 3) # [:, 7, 7, 2]
        # torch 에서는 max를 따로 구했는데 왜 ???굳히??
        best_box = tf.expand_dims(tf.argmax(ious, axis=3), 3)    # [:, 7, 7, 1]
        best_box = tf.cast(best_box, tf.float32)
        iou_max = tf.expand_dims(np.max(ious.numpy(), 3), 3)    # [:, 7, 7, 1]
        exists_box = tf.expand_dims(labels[..., 20], 3) # [:, 7, 7, 1]

        # =================== #
        # Box coordinate loss #
        box_prediction = exists_box * (
            best_box * predict[..., 26:30] + (1 - best_box) * predict[..., 21:25]
            )

        box_target = exists_box * labels[..., 21:25]

        box_prediction = box_prediction.numpy()
        box_prediction[..., 2:4] = tf.math.sign(box_prediction[..., 2:4] * tf.sqrt(tf.abs(box_prediction[..., 2:4] + 1e-7)))    # sign 은 음수가 나오면 -1, 양수가나오면 1, 0이 나오면 0을 리턴한다
        
        box_target = box_target.numpy()
        box_target[..., 2:4] = tf.sqrt(box_target[..., 2:4])

        box_loss = tf.keras.losses.MeanSquaredError()(tf.reshape(box_target, [FLAGS.batch_size*FLAGS.output_size*FLAGS.output_size, 4]),
                                       tf.reshape(box_prediction, [FLAGS.batch_size*FLAGS.output_size*FLAGS.output_size, 4]))
        # =================== #

        # =================== #
        # Object loss #
        pred_box = (best_box * predict[..., 25:26] + (1 - best_box) * predict[..., 20:21])

        object_loss = tf.keras.losses.MeanSquaredError()(tf.reshape(exists_box * labels[..., 20:21], [FLAGS.batch_size*FLAGS.output_size*FLAGS.output_size, 1]),
                                          tf.reshape(exists_box * pred_box, [FLAGS.batch_size*FLAGS.output_size*FLAGS.output_size, 1]))
        # =================== #

        # =================== #
        # No Object loss #
        # --> 동영상 52:03 초 부터 다시보면서 하면된다. 이해는 다 된다.
        # =================== #

    return total_loss

def main():
    model = Yolo_v1((FLAGS.img_size, FLAGS.img_size, 3))
    model.summary()


    text_list = os.listdir(FLAGS.tr_txt_path)
    text_list = [FLAGS.tr_txt_path + '/' + data for data in text_list]

    image_list = os.listdir(FLAGS.tr_img_path)
    image_list = [FLAGS.tr_img_path + '/' + data for data in image_list]


    count = 0
    A = list(zip(image_list, text_list))
    shuffle(A)
    image_list, text_list = zip(*A)

    image_list = np.array(image_list)
    text_list = np.array(text_list)

    data = tf.data.Dataset.from_tensor_slices((image_list, text_list))
    data = data.shuffle(len(text_list))
    data = data.map(func_, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    data = data.batch(FLAGS.batch_size)
    data = data.prefetch(tf.data.experimental.AUTOTUNE)

    batch_idx = len(text_list) // FLAGS.batch_size
    it = iter(data)
    for step in range(batch_idx):
        image, label, shape, list_ = next(it)
        #print(list_)
        original_height, original_width = shape[:, 0], shape[:, 1]
        label = read_label(label, FLAGS.batch_size)

        loss = cal_loss(model, image, label)    # 이 부분은 쉽게 이해함

        print(loss, count)

        if count % 100 == 0:
            img = save_fig(model, image, original_height, original_width)
            cv2.imshow("ss", img)
            cv2.waitKey(0)

        count += 1

if __name__ == "__main__":
    main()
