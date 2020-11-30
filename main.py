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

flags.DEFINE_float("coord_lambda", 5.0, "")

flags.DEFINE_float("noObject_lambda", 0.5, "")

flags.DEFINE_float("lr", 0.001, "Leanring rate")

FLAGS = flags.FLAGS
FLAGS(sys.argv)

optim = tf.keras.optimizers.SGD(FLAGS.lr, momentum=0.9)

def func_(image, label):

    image = tf.io.read_file(image)
    image = tf.image.decode_jpeg(image, 3)
    original_height, original_width = image.shape[0], image.shape[1]
    image = tf.image.resize(image, [FLAGS.img_size, FLAGS.img_size])

    image = tf.image.convert_image_dtype(image, tf.float32) / 255.
    #image = tf.image.per_image_standardization(image)

    return image, label, original_height, original_width

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

            offset_w = math.sqrt(b_w * 1.0 / width)
            offset_h = math.sqrt(b_h * 1.0 / height)

            boxData = [offset_x, offset_y, offset_w, offset_h]

            responsibleGridX = int(C_x / cell_w)    # confidence coordinate
            responsibleGridY = int(C_y / cell_h)

            responsibleGrid[responsibleGridX][responsibleGridY][classes] = 1    # class
            responsibleGrid[responsibleGridX][responsibleGridY][20:24] = boxData    # box
            responsibleGrid[responsibleGridX][responsibleGridY][24] = 1 # confidence

        label.append(responsibleGrid)

    label = np.array(label, dtype=np.float32)

    return label

def cal_iou(predict_box1, predict_box2, label_box):
    def get_iou(label_box, predict_box):

        intersection_width = tf.minimum(predict_box[:, :, :, 0] + 0.5*predict_box[:, :, :, 2],
                                        label_box[:, :, :, 0] + 0.5*label_box[:, :, :, 2]) \
                            - tf.maximum(predict_box[:, :, :, 0] - 0.5*predict_box[:, :, :, 2],
                                            label_box[:, :, :, 0] - 0.5*label_box[:, :, :, 2])

        intersection_height = tf.minimum(predict_box[:, :, :, 1] + 0.5*predict_box[:, :, :, 3],
                                        label_box[:, :, :, 1] + 0.5*label_box[:, :, :, 3]) \
                            - tf.maximum(predict_box[:, :, :, 1] - 0.5*predict_box[:, :, :, 3],
                                            label_box[:, :, :, 1] - 0.5*label_box[:, :, :, 3])

        intersection = tf.multiply(tf.maximum(0, intersection_height), tf.maximum(0, intersection_width))

        union = tf.subtract(tf.multiply(predict_box[:, :, :, 2], predict_box[:, :, :, 3]) + tf.multiply(label_box[:, :, :, 2], label_box[:, :, :, 3]),intersection)

        iou = tf.divide(intersection, union)

        return iou

    iou1 = tf.reshape(get_iou(label_box, predict_box1), [-1, 7, 7, 1])
    iou2 = tf.reshape(get_iou(label_box, predict_box2), [-1, 7, 7, 1])
    return tf.concat([iou1, iou2], 3)

def save_fig(model, images, original_height, original_width):

    output = model(images, False)

    predictedObjects = []

    pred_class = (output[:,:,:,:20])
    predictedObjectConfidence = output[:, :, :, 20:22]
    class_confidences = tf.split(pred_class, num_or_size_splits=20, axis=-1)

    predictedBoxes = output[:, :, :, 22:]
    predictedFirstBoxes = predictedBoxes[:, :, :, :4]
    predictedFirstBoxes = tf.reshape(predictedFirstBoxes, [-1, 7, 7, 1, 4])
    predictedSecondBoxes = predictedBoxes[:, :, :, 4:]
    predictedSecondBoxes = tf.reshape(predictedSecondBoxes, [-1, 7, 7, 1, 4])
    result = tf.concat([predictedFirstBoxes, predictedSecondBoxes], 3)
    np_result = result.numpy()
    np_result = np.array(np_result)

    offset_Y = np.tile(np.arange(7)[:, np.newaxis, np.newaxis],(1, 7, 2))
    offset_X = np.transpose(offset_Y, (1, 0, 2))

    for i in range(FLAGS.batch_size):
        np_result_ = np_result[i]

        np_result_[:, :, :, 0] = 1. * (np_result_[:, :, :, 0]+offset_X) * original_width
        np_result_[:, :, :, 1] = 1. * (np_result_[:, :, :, 1]+offset_Y) * original_height

        np_result_[:, :, :, 2] = original_width * np.multiply(np_result_[:, :, :, 2],
                                                              np_result_[:, :, :, 2])
        np_result_[:, :, :, 3] = original_width * np.multiply(np_result_[:, :, :, 3],
                                                              np_result_[:, :, :, 3])

        for class_confi in class_confidences:
            scores = predictedObjectConfidence * class_confi
            scores = tf.reshape(scores, [-1])

        # [7,7,2,20]

    return img

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

    with tf.GradientTape(persistent=True) as tape:
        logits = model(images, training=True)
        pred_class = (logits[:,:,:,:20])
        predictedObjectConfidence = logits[:, :, :, 20:22]
        predictedBoxes = logits[:, :, :, 22:]
        predictedFirstBoxes = predictedBoxes[:, :, :, :4]
        predictedSecondBoxes = predictedBoxes[:, :, :, 4:]

        groundTruthClasses = labels[:, :, :, :20]
        groundTruthBoxes = labels[:, :, :, 20:24]
        groundTruthGrid = labels[:, :, :, 24:]

        firstBox_loss = tf.reduce_sum(tf.square(predictedFirstBoxes - groundTruthBoxes), 3)
        secondBox_loss = tf.reduce_sum(tf.square(predictedSecondBoxes - groundTruthBoxes), 3)
                
        IOU = cal_iou(predictedFirstBoxes, predictedSecondBoxes, groundTruthBoxes)
        responsibleBox = tf.greater(IOU[:, :, :, 0], IOU[:, :, :, 1])
        coordinateLoss = tf.where(responsibleBox, firstBox_loss, secondBox_loss)    # [-1, 7, 7]
        coordinateLoss = tf.reshape(coordinateLoss, [-1, 7, 7, 1])  # [-1, 7, 7, 1]

        coordinateLoss = FLAGS.coord_lambda * tf.multiply(groundTruthGrid, coordinateLoss)

        object_loss = tf.square(predictedObjectConfidence - groundTruthGrid)
        object_loss = tf.where(responsibleBox, object_loss[:, :, :, 0], object_loss[:, :, :, 1])
        object_loss = tf.reshape(object_loss, [-1, 7, 7, 1])

        noObject_loss = FLAGS.noObject_lambda * tf.multiply(1 - groundTruthGrid, object_loss)
        object_loss = tf.multiply(groundTruthGrid, object_loss)

        class_loss = tf.square(pred_class - groundTruthClasses)
        class_loss = tf.reduce_sum(tf.multiply(class_loss, groundTruthGrid), 3)
        class_loss = tf.reshape(class_loss, [-1, 7, 7, 1])

        total_loss = coordinateLoss + object_loss + noObject_loss + class_loss
        total_loss = tf.reduce_mean(tf.reduce_sum(total_loss, [1,2,3]), 0)

        # coordinateLoss need reshape??

    grads = tape.gradient(total_loss, model.trainable_variables)
    optim.apply_gradients(zip(grads, model.trainable_variables))
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
        image, label, original_height, original_width = next(it)   
        label = read_label(label, FLAGS.batch_size)

        loss = cal_loss(model, image, label)

        save_fig(model, image, original_height, original_width)

        print(loss)

if __name__ == "__main__":
    main()
