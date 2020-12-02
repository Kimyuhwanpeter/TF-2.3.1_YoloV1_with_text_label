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
    # https://github.com/lovish1234/YOLOv1/blob/master/preprocess.py

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

            C_x = ((xmin + xmax) * 1.0 / 2.0) * \
                    (FLAGS.img_size * 1.0 / width)
            C_y = ((ymin + ymax) * 1.0 / 2.0) * \
                    (FLAGS.img_size * 1.0 / height)
            b_w = (xmax - xmin) * 1.0
            b_h = (ymax - ymin) * 1.0

            offset_x = ((C_x % cell_w) * 1.0) / cell_w
            offset_y = ((C_y % cell_h) * 1.0) / cell_h

            offset_w = math.sqrt(b_w * 1.0) / width
            offset_h = math.sqrt(b_h * 1.0) / height

            boxData = [offset_x, offset_y, offset_w, offset_h]

            responsibleGridX = int(C_x / cell_w)    # confidence coordinate
            responsibleGridY = int(C_y / cell_h)

            responsibleGrid[b][responsibleGridX][responsibleGridY][5:classes + 6] = 1    # class
            responsibleGrid[b][responsibleGridX][responsibleGridY][0:4] = boxData    # box
            responsibleGrid[b][responsibleGridX][responsibleGridY][4] = 1 # confidence

        #    cla.append(responsibleGrid)

        #label.append(cla)

    responsibleGrid = np.array(responsibleGrid, dtype=np.float32)

    return responsibleGrid

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

def test_iou(box1, box2):

    intersection_width = max(0, min(box1[0] + box1[2]*0.5, box2[0] + box2[2]*0.5) \
                        - max(box1[0] - box1[2]*0.5, box2[0] - box2[2]*0.5))

    intersection_height = max(0, min(box1[1] + box1[3]*0.5, box2[1] + box2[3]*0.5) \
                        - max(box1[1] - box1[3]*0.5, box2[1] - box2[3]*0.5))

    intersection = intersection_width * intersection_height
    union = box1[2]*box1[3] + box2[2]*box2[3] - intersection

    return intersection / union

def save_fig(model, images, original_height, original_width):

    output = model(images, False)

    pred_class = tf.reshape(output[:, 0:980], [-1,7,7,20])
    pred_confidence = tf.reshape(output[:, 980:1078], [-1, 7, 7, 2])

    predictedBoxes = tf.reshape(output[:, 1078:], [-1, 7, 7, 2, 4])
    np_result = predictedBoxes.numpy()
    np_result = np.array(np_result)

    offset_Y = np.tile(np.arange(7, dtype=np.float32)[:, np.newaxis, np.newaxis],(1, 7, 2))
    offset_X = np.transpose(offset_Y, (1, 0, 2))

    for i in range(FLAGS.batch_size):

        original_height_ = original_height[i].numpy()
        original_width_ = original_width[i].numpy()

        np_result_ = np_result[i]

        np_result_[:, :, :, 0] = 1. * (np_result_[:, :, :, 0]+offset_X) * FLAGS.img_size / FLAGS.output_size
        np_result_[:, :, :, 1] = 1. * (np_result_[:, :, :, 1]+offset_Y) * FLAGS.img_size / FLAGS.output_size

        np_result_[:, :, :, 2] = FLAGS.img_size * np.multiply(np_result_[:, :, :, 2],
                                                              np_result_[:, :, :, 2])
        np_result_[:, :, :, 3] = FLAGS.img_size * np.multiply(np_result_[:, :, :, 3],
                                                              np_result_[:, :, :, 3])

        np_result_[:, :, :, 0] = np_result_[:, :, :, 0] - np_result_[:, :, :, 2] / 2
        np_result_[:, :, :, 1] = np_result_[:, :, :, 1] - np_result_[:, :, :, 3] / 2
        np_result_[:, :, :, 2] = np_result_[:, :, :, 0] + np_result_[:, :, :, 2] / 2
        np_result_[:, :, :, 3] = np_result_[:, :, :, 1] + np_result_[:, :, :, 3] / 2

        np_result_[:, :, :, 0] = np_result_[:, :, :, 1]
        np_result_[:, :, :, 1] = np_result_[:, :, :, 0]
        np_result_[:, :, :, 2] = np_result_[:, :, :, 3]
        np_result_[:, :, :, 3] = np_result_[:, :, :, 2]
        box_data = np_result_













        # 1. 각 바운딩 박스와 그 바운딩 박스의 중심좌료가 있는 그리데셀의 스코어 값들을 각각 곱함
        pred_confidence_ = pred_confidence[i]
        pred_confidence_ = tf.expand_dims(pred_confidence_, 0)  # [1, 7, 7, 2]
        pred_class_ = pred_class[i]
        pred_class_ = tf.expand_dims(pred_class_, 0)    # [1, 7, 7, 20]
        pred_class_ = tf.split(pred_class_, num_or_size_splits=20, axis=-1)

        each_class_score_buf = []
        for pred_cla in pred_class_:
            each_class_score = pred_confidence_ * pred_cla  # [1, 7, 7, 2]
            each_class_score = each_class_score.numpy() # [1, 7, 7, 2]

            # 2. threshold 값으로 0.2를 지정, 해당 0.2 값보다 작은 결과는 다 0으로 만듬
            for k in range(FLAGS.output_size):
                for m in range(FLAGS.output_size):
                    for b in range(2):
                        if each_class_score[0, k, m, b] < FLAGS.mini_threshold:
                            each_class_score[0, k, m, b] = 0.
                        else:
                            each_class_score[0, k, m, b] = each_class_score[0, k, m, b]

            each_class_score_buf.append(each_class_score)
        each_class_score_buf = tf.convert_to_tensor(each_class_score_buf)   # [20, 1, 7, 7, 2]
        each_class_score_buf = tf.transpose(each_class_score_buf, [1,2,3,4,0])  # [1, 7, 7, 2, 20]

        each_class_score_NPlist = each_class_score_buf.numpy()

        # 3. 각 박스에 있는값들 내림차순으로 정렬
        boxidx_class = []
        idx_class = []
        val_class = []
        for c in range(20):
            sort_append = []
            box = []
            for k in range(FLAGS.output_size):
                for m in range(FLAGS.output_size):
                    for b in range(2):
                        sort_append.append(each_class_score_buf[0, k, m, b, c].numpy())
                        box.append(box_data[k, m, b, :])

            sort_val = np.sort(sort_append)[::-1]   # 내림차순 정렬 (value)   [98]
            sort_idx = np.argsort(sort_append)[::-1]    # 내림차순 정렬 (index) [98]
            sort_box = []
            for ii in range(len(sort_append)):
                sort_box.append(box[sort_idx[ii]])
            sort_box = np.array(sort_box)   # 내림차순 정렬 (box) [98, 4]

            # NMS
            selected_indices = tf.image.non_max_suppression(sort_box, sort_val, 3)
            selected_boxes = tf.gather(sort_box, selected_indices)

            boxidx_class.append(selected_boxes[tf.newaxis, :])
            val_class.append(sort_val[tf.newaxis, :])
            idx_class.append(sort_idx[tf.newaxis, :])

        boxidx_class = tf.concat(boxidx_class, 0)   # [3]
        val_class = tf.concat(val_class, 0)     # []
        idx_class = tf.concat(idx_class, 0)



            # 박스 위치도 같이 움직여야 하는것 아닌가?




    return image

def cal_loss(model, images, labels):
    # https://github.com/lovish1234/YOLOv1/blob/master/yolo.py

    with tf.GradientTape(persistent=True) as tape:
        logits = model(images, training=True)
        pred_class = tf.reshape(logits[:, 0:980], [-1,7,7,20])
        predictedObjectConfidence = tf.reshape(logits[:, 980:1078], [-1, 7, 7, 2])
        predictedBoxes = tf.reshape(logits[:, 1078:], [-1, 7, 7, 2, 4])
        predictedFirstBoxes = predictedBoxes[:, :, :, 0:1, :]
        predictedFirstBoxes = tf.squeeze(predictedFirstBoxes, 3)
        predictedSecondBoxes = predictedBoxes[:, :, :, 1:2, :]
        predictedSecondBoxes = tf.squeeze(predictedSecondBoxes, 3)

        groundTruthClasses = labels[:, :, :, 5:]
        groundTruthBoxes = labels[:, :, :, 0:4]
        groundTruthGrid = labels[:, :, :, 4:5]

        firstBox_loss = tf.reduce_sum(tf.square(predictedFirstBoxes - groundTruthBoxes), 3)
        secondBox_loss = tf.reduce_sum(tf.square(predictedSecondBoxes - groundTruthBoxes), 3)
                
        IOU = cal_iou(predictedFirstBoxes, predictedSecondBoxes, groundTruthBoxes)
        responsibleBox = tf.greater(IOU[:, :, :, 0], IOU[:, :, :, 1])
        coordinateLoss = tf.where(responsibleBox, firstBox_loss, secondBox_loss)    # [-1, 7, 7]
        coordinateLoss = tf.reshape(coordinateLoss, [-1, 7, 7, 1])  # [-1, 7, 7, 1]

        coordinateLoss = FLAGS.coord_lambda * tf.multiply(groundTruthGrid, coordinateLoss)

        object_loss = tf.square(predictedObjectConfidence - groundTruthGrid)
        object_loss = tf.where(responsibleBox, object_loss[:, :, :, 0], object_loss[:, :, :, 1])
        tempObjectLoss = tf.reshape(object_loss, [-1, 7, 7, 1])

        noObject_loss = FLAGS.noObject_lambda * tf.multiply(1 - groundTruthGrid, tempObjectLoss)
        object_loss = tf.multiply(groundTruthGrid, tempObjectLoss)

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
