# -*- coding:utf-8 -*-
from model import Yolo_v1
from absl import flags
from random import random, shuffle

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import math

flags.DEFINE_string("tr_img_path", "D:/[1]DB/[3]detection_DB/PascalVoc2012/pascal_voc_2012/VOC2012/JPEGImages", "Training image path")

flags.DEFINE_string("tr_txt_path", "D:/[1]DB/[3]detection_DB/PascalVoc2012/pascal_voc_2012/VOC2012/Annotations_text", "Training text path")

flags.DEFINE_integer("img_size", 448, "model input size")

flags.DEFINE_integer("batch_size", 2, "batch size")

flags.DEFINE_integer("output_size", 7, "")

FLAGS = flags.FLAGS
FLAGS(sys.argv)

def func_(image, label):

    image = tf.io.read_file(image)
    image = tf.image.decode_jpeg(image, 3)
    image = tf.image.resize(image, [FLAGS.img_size, FLAGS.img_size])

    image = tf.image.convert_image_dtype(image, tf.float32) / 255.
    #image = tf.image.per_image_standardization(image)

    return image, label

def read_label(file, batch_size):
    # https://github.com/lovish1234/YOLOv1/blob/master/preprocess.py

    cell_h = FLAGS.img_size // FLAGS.output_size
    cell_w = FLAGS.img_size // FLAGS.output_size

    label = []
    responsibleGrid = np.zeros([7, 7, 25])
    for b in range(batch_size):
        f = open(tf.compat.as_bytes(file[b].numpy()), 'r')
        cla = []
        while True:
            line = f.readline()
            if not line: break
            line = line.split('\n')[0]
            
            # resize 448 x 448 with roi
            #height_rate = (FLAGS.img_size / int(line.split(',')[4]))
            #width_rate = (FLAGS.img_size / int(line.split(',')[5]))

            xmin = int(int(line.split(',')[0]))
            xmax = int(int(line.split(',')[2]))
            ymin = int(int(line.split(',')[1]))
            ymax = int(int(line.split(',')[3]))
            height = int(line.split(',')[4])
            width = int(line.split(',')[5])
            classes = int(line.split(',')[6])

            # take the center point and scale according to new size

            C_x = ((xmin + xmax) * 1.0 / 2.0) * (FLAGS.img_size * 1.0 / width)
            C_y = ((ymin + ymax) * 1.0 / 2.0) * (FLAGS.img_size * 1.0 / height)
            b_w = (xmax - xmin) * 1.0
            b_h = (ymax - ymin) * 1.0

            offset_x = ((C_x % cell_w) * 1.0) / cell_w
            offset_y = ((C_y % cell_h) * 1.0) / cell_h

            offset_w = math.sqrt(b_w * 1.0) / width
            offset_h = math.sqrt(b_h * 1.0) / height

            boxData = [offset_x, offset_y, offset_w, offset_h]

            responsibleGridX = int(C_x / cell_w)    # confidence coordinate
            responsibleGridY = int(C_y / cell_h)

            responsibleGrid[responsibleGridX][responsibleGridY][classes] = 1    # class
            responsibleGrid[responsibleGridX][responsibleGridY][20:24] = boxData    # box
            responsibleGrid[responsibleGridX][responsibleGridY][24] = 1 # confidence

        label.append(responsibleGrid)

    label = np.array(label, dtype=np.float32)

    return label

def cal_loss(model, images, labels):
    # https://github.com/lovish1234/YOLOv1/blob/master/yolo.py
    """
    Calculate the total loss for gradient descent.
    For each ground truth object, loss needs to be calculated.
    It is assumed that each image consists of only one object.
    Predicted
    0-19 CLass prediction
    20-21 Confidence that objects exist in bbox1 or bbox2 of grid
    22-29 Coordinates for bbo1, followed by those of bbox2
    Real
    0-19 Class prediction (One-Hot Encoded)
    20-23 Ground truth coordinates for that box
    24-72 Cell has an object/no object (Only one can be is 1)
    """

    with tf.GradientTape() as tape:
        logits = model(images, training=True)
        pred_class = logits[:,:,:,:20]
        predictedObjectConfidence = logits[:, :, :, 20:22]
        predictedBoxes = logits[:, :, :, 22:]
        predictedFirstBoxes = predictedBoxes[:, :, :, :4]
        predictedSecondBoxes = predictedBoxes[:, :, :, 5:]

        groundTruthClasses = labels[:, :, :, :20]
        groundTruthBoxes = labels[:, :, :, 20:24]
        groundTruthGrid = labels[:, :, :, 24:]

        # 891 줄 부터 다음주 월요일에 다시하자!!!


    return total_loss

def main():
    model = Yolo_v1((FLAGS.img_size, FLAGS.img_size, 3))
    model.summary()


    text_list = os.listdir(FLAGS.tr_txt_path)
    text_list = [FLAGS.tr_txt_path + '/' + data for data in text_list]

    image_list = os.listdir(FLAGS.tr_img_path)
    image_list = [FLAGS.tr_img_path + '/' + data for data in image_list]

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
        image, label = next(it)   
        label = read_label(label, FLAGS.batch_size)

        loss = cal_loss(model, image, label)

if __name__ == "__main__":
    main()