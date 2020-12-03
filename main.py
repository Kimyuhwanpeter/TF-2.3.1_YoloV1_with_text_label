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

    logits = model(images, False)

    pred_class = logits[:,:,:,:20]
    pred_confidence = logits[:, :, :, 20:22]

    predictedBoxes = tf.reshape(logits[:, :, :, 22:], [-1, 7, 7, 2, 4])
    np_result = predictedBoxes.numpy()
    np_result = np.array(np_result)

    offset = np.transpose(np.reshape(np.array([np.arange(7)]*14),(2,7,7)),(1,2,0))
    probs = np.zeros((7,7,2,20))

    for i in range(FLAGS.batch_size):

        original_height_ = original_height[i].numpy()
        original_width_ = original_width[i].numpy()
        img = images[i].numpy()
        np_result_ = np_result[i]

        np_result_[:, :, :, 0] = 1. * (np_result_[:, :, :, 0]+offset) * FLAGS.img_size / FLAGS.output_size
        np_result_[:, :, :, 1] = 1. * (np_result_[:, :, :, 1]+np.transpose(offset, (1,0,2))) * FLAGS.img_size / FLAGS.output_size

        np_result_[:, :, :, 2] = original_width_ * np.multiply(np_result_[:, :, :, 2],
                                                              np_result_[:, :, :, 2])
        np_result_[:, :, :, 3] = original_height_ * np.multiply(np_result_[:, :, :, 3],
                                                              np_result_[:, :, :, 3])
        box_data = np_result_
        # 박스 클립핑을 꼭 할 것!
        # 1. 각 바운딩 박스와 그 바운딩 박스의 중심좌료가 있는 그리데셀의 스코어 값들을 각각 곱함
        for k in range(2):
            for m in range(20):
                probs[:,:,k,m] = np.multiply(pred_class[i,:,:,m],pred_confidence[i,:,:,k])

        filter_mat_probs = np.array(probs>=FLAGS.mini_threshold,dtype='bool')
        filter_mat_boxes = np.nonzero(filter_mat_probs)
        boxes_filtered = box_data[filter_mat_boxes[0],filter_mat_boxes[1],filter_mat_boxes[2]]
        probs_filtered = probs[filter_mat_probs]
        classes_num_filtered = np.argmax(filter_mat_probs,axis=3)[filter_mat_boxes[0],filter_mat_boxes[1],filter_mat_boxes[2]]

        argsort = np.array(np.argsort(probs_filtered))[::-1]
        boxes_filtered = boxes_filtered[argsort]
        probs_filtered = probs_filtered[argsort]
        classes_num_filtered = classes_num_filtered[argsort]
    
        for i in range(len(boxes_filtered)):
            if probs_filtered[i] == 0 : continue
            for j in range(i+1,len(boxes_filtered)):
                if test_iou(boxes_filtered[i],boxes_filtered[j]) > 0.5:
                    probs_filtered[j] = 0.0

        filter_iou = np.array(probs_filtered>0.0,dtype='bool')
        boxes_filtered = boxes_filtered[filter_iou]
        probs_filtered = probs_filtered[filter_iou]
        classes_num_filtered = classes_num_filtered[filter_iou]

        result = []
        for i in range(len(boxes_filtered)):    # 고쳐ㅛ야되ㅏㅁ
            result.append([OBJECT_NAMES[classes_num_filtered[i]],boxes_filtered[i][0],boxes_filtered[i][1],boxes_filtered[i][2],boxes_filtered[i][3],probs_filtered[i]])

        for i in range(len(result)):
            x = int(result[i][1])
            y = int(result[i][2])
            w = int(result[i][3]) // 2
            h = int(result[i][4]) // 2

            xmin = max(x - w, 0)
            ymin = max(y - h, 0)
            xmax = max(x + w, 0)
            ymax = max(y + h, 0)

            cv2.rectangle(img,(xmin,ymin),(xmax,ymax),(0,255,0),2)
            cv2.putText(img,result[i][0] + ' : %.2f' % result[i][5],(xmin+5,ymin-7),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1)

        # https://github.com/gliese581gg/YOLO_tensorflow/blob/master/YOLO_small_tf.py  --> 219 줄!!!
        cv2.imshow("dd", img)
        cv2.waitKey(0)




    return image

def cal_loss(model, images, labels):
    # https://github.com/lovish1234/YOLOv1/blob/master/yolo.py

    with tf.GradientTape(persistent=True) as tape:
        logits = model(images, training=True)
        pred_class = logits[:,:,:,:20]
        predictedObjectConfidence = logits[:, :, :, 20:22]
        predictedBoxes = tf.reshape(logits[:, :, :, 22:], [-1, 7, 7, 2, 4])
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
