# -*- coding:utf-8 -*-
from PIL import Image, ImageDraw
from model import Yolo_v1
from absl import flags
from random import random, shuffle, randint

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import sys
import math
import cv2

OBJECT_NAMES = ["person", "bird", "cat", "cow", "dog", "horse", "sheep", "aeroplane", "boat", "bus",
                   "car", "motorbike", "train", "bottle", "chair", "dining", "table", "potted plane", "sofa", "tv/monitor"]

flags.DEFINE_string("tr_img_path", "D:/[1]DB/[3]detection_DB/PascalVoc2012/pascal_voc_2012/VOC2012/JPEGImages", "Training image path")

flags.DEFINE_string("tr_txt_path", "D:/[1]DB/[3]detection_DB/PascalVoc2012/pascal_voc_2012/VOC2012/Annotations_text", "Training text path")

flags.DEFINE_integer("img_size", 448, "model input size")

flags.DEFINE_integer("batch_size", 8, "batch size")

flags.DEFINE_integer("epochs", 50, "Total epochs")

flags.DEFINE_integer("output_size", 7, "")

flags.DEFINE_float("mini_threshold", 0.2, "")

flags.DEFINE_float("coord_lambda", 5.0, "")

flags.DEFINE_float("noObject_lambda", 0.5, "")

flags.DEFINE_float("lr", 0.0002, "Leanring rate")

flags.DEFINE_bool("pre_checkpoint", False, "True or False")

flags.DEFINE_string("pre_checkpoint_path", "", "Saved checkpoint path")

flags.DEFINE_bool("train", True, "True or False")

flags.DEFINE_string("save_checkpoint", "", "Saving checkpoint path")


FLAGS = flags.FLAGS
FLAGS(sys.argv)

#optim = tf.compat.v1.train.MomentumOptimizer(FLAGS.lr, momentum=0.9)
optim = tf.keras.optimizers.Adam(FLAGS.lr)

#############################################################################################
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
    full_target_grid = []
    for b in range(batch_size):
        f = open(tf.compat.as_bytes(file[b].numpy()), 'r')
        cla = []
        traget_grid = []
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
            #if xmax > xmin:
            #    xmax = xmax
            #    xmin = xmin
            #if xmax < xmin:
            #    xmax = xmin
            #    xmin = xmax
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


            if responsibleGrid[b][responsibleGridX, responsibleGridY, 20] == 0:
                responsibleGrid[b][responsibleGridX][responsibleGridY][classes] = 1    # class
                responsibleGrid[b][responsibleGridX][responsibleGridY][21:25] = boxData    # box
                responsibleGrid[b][responsibleGridX][responsibleGridY][20] = 1 # confidence

                traget_grid.append([classes, 1, x, y, w, h])

        full_target_grid.append(traget_grid)

    responsibleGrid = np.array(responsibleGrid, dtype=np.float32)
    traget_grid = np.array(traget_grid, dtype=np.float32)

    return responsibleGrid, full_target_grid
#############################################################################################

#############################################################################################
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

    intersect = tf.where(intersect_x2 - intersect_x1 < 0, 0, intersect_x2 - intersect_x1) * \
                tf.where(intersect_y2 - intersect_y1 < 0, 0, intersect_y2 - intersect_y1)

    box1_area = tf.abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = tf.abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    iou = intersect / (box1_area + box2_area - intersect + 1e-7)

    return iou

def cal_loss(model, images, labels):

    # 모델의 최종 output은 1460 이 된다.
    # https://www.youtube.com/watch?v=n9_XyCGr-MI
    with tf.GradientTape() as tape:
        predict = model(images, True)   # [:, 1470]
        predict = tf.reshape(predict, [-1, FLAGS.output_size, FLAGS.output_size, 20 + 2*5]) # [:, 7, 7, 30]
        a = tf.expand_dims(tf.argmax(predict[..., :20], -1), -1)
        output_b1 = train_IOU(predict[..., 21:25], labels[..., 21:25])    # [:, 7, 7, 1]
        output_b2 = train_IOU(predict[..., 26:30], labels[..., 21:25])    # [:, 7, 7, 1]
        ious = tf.concat([output_b1, output_b2], 3) # [:, 7, 7, 2]
        best_box = tf.expand_dims(tf.argmax(ious, axis=3), 3)    # [:, 7, 7, 1]
        best_box = tf.cast(best_box, tf.float32)
        iou_max = tf.expand_dims(np.max(ious.numpy(), 3), 3)    # [:, 7, 7, 1]
        exists_box = tf.expand_dims(labels[..., 20], 3) # [:, 7, 7, 1] --> I^obj

        # =================== #
        # Box coordinate loss #
        box_prediction = exists_box * (
            best_box * predict[..., 26:30] + (1 - best_box) * predict[..., 21:25]
            )

        box_target = exists_box * labels[..., 21:25]

        box_prediction = box_prediction.numpy()
        box_prediction[..., 2:4] = tf.math.sign(box_prediction[..., 2:4]) * tf.sqrt(tf.abs(box_prediction[..., 2:4] + 1e-7))    # sign 은 음수가 나오면 -1, 양수가나오면 1, 0이 나오면 0을 리턴한다
        
        box_target = box_target.numpy()
        box_target[..., 2:4] = tf.sqrt(box_target[..., 2:4])

        # 내가 느끼기에는 loss가 잘못되어서 box가 이상하게 나오는것같다 여기를 고쳐보자
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
        no_object_loss = tf.keras.losses.MeanSquaredError()(tf.reshape((1 - exists_box) * labels[..., 20:21], [-1, 7*7]),
                                                            tf.reshape((1 - exists_box) * predict[..., 20:21], [-1, 7*7]))
        no_object_loss += tf.keras.losses.MeanSquaredError()(tf.reshape((1 - exists_box) * labels[..., 20:21], [-1, 7*7]),
                                                            tf.reshape((1 - exists_box) * predict[..., 25:26], [-1, 7*7]))

        # =================== #

        # =================== #
        # Class loss #
        class_loss = tf.keras.losses.MeanSquaredError()(tf.reshape(exists_box * labels[..., :20], [FLAGS.batch_size*FLAGS.output_size*FLAGS.output_size, 20]),
                                                        tf.reshape(exists_box * predict[..., :20], [FLAGS.batch_size*FLAGS.output_size*FLAGS.output_size, 20]))

        # =================== #

        loss = FLAGS.coord_lambda * box_loss \
            + object_loss \
            + FLAGS.noObject_lambda * no_object_loss \
            + class_loss
        

    grads = tape.gradient(loss, model.trainable_variables)
    optim.apply_gradients(zip(grads, model.trainable_variables))

    return loss
#############################################################################################

#############################################################################################
def convert_cellboxes_box(output, S=FLAGS.output_size):

    # convert to cellbox
    predict = tf.reshape(output, [tf.shape(output)[0], 7, 7, 20 + 2*5])

    box1 = predict[..., 21:25]
    box2 = predict[..., 26:30]  # 학습할 때는 이 부분이 참이도록 학습하였음

    scores = tf.concat([predict[..., 20:21], predict[..., 25:26]], 3)
    best_box = tf.expand_dims(tf.argmax(scores, 3), 3)
    best_box = tf.cast(best_box, tf.float32)
    best_boxes = box1 * (1 - best_box) + best_box * box2    # 216줄로 인해 이렇게 수식   # [:, 7, 7, 4]
    cell_indices = np.tile(np.arange(0,7, dtype=np.float32), [tf.shape(output)[0], FLAGS.output_size, 1])    # [:, 7, 7]
    cell_indices = np.expand_dims(cell_indices, 3)  # [:, 7, 7, 1]

    x = 1 / S * (best_boxes[..., :1] + cell_indices)    # [:, 7, 7, 1]
    y = 1 / S * (best_boxes[..., 1:2] + np.transpose(cell_indices, [0, 2, 1, 3]))   # [:, 7, 7, 1]
    w_h = 1 / S * (best_boxes[..., 2:4])    # [:, 7, 7, 2]

    converted_bboxes = tf.concat([x, y, w_h], 3)    # [:, 7, 7, 4]
    predict_class = predict[..., :20]
    predict_class = tf.expand_dims(tf.argmax(predict_class, 3), 3) # [:, 7, 7, 1]
    predict_class = tf.cast(predict_class, tf.float32)
    best_confidence = tf.maximum(predict[..., 25:26], predict[..., 20:21])  # [:, 7, 7, 1]

    convert_box = tf.concat([predict_class, best_confidence, converted_bboxes], 3)

    # cellbox to box
    convert_box = tf.reshape(convert_box, [tf.shape(output)[0], FLAGS.output_size*FLAGS.output_size, -1])
    convert_box = convert_box.numpy()
    convert_box[..., 0] = tf.cast(convert_box[..., 0], tf.int64)
    all_boxes = []

    for idx in range(tf.shape(output)[0]):
        boxes = []

        for box_idx in range(FLAGS.output_size*FLAGS.output_size):
            boxes.append([x for x in convert_box[idx, box_idx, :]])
        all_boxes.append(boxes)

    return all_boxes

def convert_label_box(label, S=FLAGS.output_size):

    # cellbox to box
    convert_box = tf.reshape(label, [tf.shape(label)[0], FLAGS.output_size*FLAGS.output_size, -1])
    convert_box = convert_box.numpy()
    convert_box[..., 0] = tf.cast(convert_box[..., 0], tf.int64)
    all_boxes = []

    for idx in range(tf.shape(label)[0]):
        boxes = []

        for box_idx in range(FLAGS.output_size*FLAGS.output_size):
            boxes.append([x for x in convert_box[idx, box_idx, :]])
        all_boxes.append(boxes)

    return all_boxes

def nms(bboxes, iou_threshold, threshold):

    #for box in bboxes:
    #    if box[1] > threshold:
    #        print("@@@@@@@@")       # 학습이 되는지 안되는지 확인용

    bboxes = [box for box in bboxes if box[1] > threshold]
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    bboxes_after_nms = []

    while bboxes:
        chosen_box = bboxes.pop(0)

        bboxes = [
            box
            for box in bboxes
            if box[0] != chosen_box[0]
            or test_IOU(
                chosen_box[2:],
                box[2:]
            )
            < iou_threshold
        ]

        bboxes_after_nms.append(chosen_box)

    return bboxes_after_nms

def test_IOU(predict_box, label_box):

    box1_x1 = predict_box[0] - predict_box[2] / 2
    box1_y1 = predict_box[1] - predict_box[3] / 2
    box1_x2 = predict_box[0] + predict_box[2] / 2
    box1_y2 = predict_box[1] + predict_box[3] / 2

    box2_x1 = label_box[0] - label_box[2] / 2
    box2_y1 = label_box[1] - label_box[3] / 2
    box2_x2 = label_box[0] + label_box[2] / 2
    box2_y2 = label_box[1] + label_box[3] / 2

    intersect_x1 = tf.maximum(box1_x1, box2_x1)
    intersect_y1 = tf.maximum(box1_y1, box2_y1)
    intersect_x2 = tf.minimum(box1_x2, box2_x2)
    intersect_y2 = tf.minimum(box1_y2, box2_y2)

    intersect = tf.where(intersect_x2 - intersect_x1 < 0, 0, intersect_x2 - intersect_x1) * \
                tf.where(intersect_y2 - intersect_y1 < 0, 0, intersect_y2 - intersect_y1)

    #intersect = tf.clip_by_value(intersect_x2 - intersect_x1, clip_value_min=0, clip_value_max=7) * \
    #            tf.clip_by_value(intersect_y2 - intersect_y1, clip_value_min=0, clip_value_max=7)

    box1_area = tf.abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = tf.abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    iou = intersect / (box1_area + box2_area - intersect + 1e-7)

    return iou

#############################################################################################

#############################################################################################
def gener_sample(image, boxes):
    
    """Plots predicted bounding boxes on the image"""
    im = np.array(image)
    height, width, _ = im.shape

    # Create figure and axes
    fig, ax = plt.subplots(1)
    # Display the image
    ax.imshow(im)

    # box[0] is x midpoint, box[2] is width
    # box[1] is y midpoint, box[3] is height

    # Create a Rectangle potch
    for box in boxes:
        box = box[2:]
        assert len(box) == 4, "Got more values than in x, y, w, h, in a box!"
        upper_left_x = box[0] - box[2] / 2
        upper_left_y = box[1] - box[3] / 2
        print(box[2] * width)
        print(box[3] * height)
        rect = patches.Rectangle(
            (upper_left_x * width, upper_left_y * height),
            box[2] * width,
            box[3] * height,
            linewidth=1,
            edgecolor="r",
            facecolor="none",
        )
        # Add the patch to the Axes
        ax.add_patch(rect)


    plt.show()
#############################################################################################

def main():
    model = Yolo_v1((FLAGS.img_size, FLAGS.img_size, 3))
    model.summary()

    if FLAGS.pre_checkpoint:
        ckpt = tf.train.Checkpoint(model=model, optim=optim)
        ckpt_manager = tf.train.CheckpointManager(ckpt, FLAGS.pre_checkpoint_path, 5)
        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint)
            print("Restored the checkpoint!!")

    if FLAGS.train:
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

        for epoch in range(FLAGS.epochs):

            data = tf.data.Dataset.from_tensor_slices((image_list, text_list))
            data = data.shuffle(len(text_list))
            data = data.map(func_)
            data = data.batch(FLAGS.batch_size)
            data = data.prefetch(tf.data.experimental.AUTOTUNE)

            batch_idx = len(text_list) // FLAGS.batch_size
            it = iter(data)
            for step in range(batch_idx):
                image, label, shape, list_ = next(it)
                original_height, original_width = shape[:, 0], shape[:, 1]
                tr_label, target_label = read_label(label, FLAGS.batch_size)

                loss = cal_loss(model, image, tr_label)

                if count % 10 == 0:
                    print("Epoch: {} [{}/{}] loss = {}".format(epoch, step + 1, batch_idx, loss))

                if count % 1000 == 0:
                    if count != 0:
                        output = model(image, False)
                        for idx in range(2):
                            boxes = convert_cellboxes_box(output)
                            boxes = nms(boxes[idx], 0.5, 0.2)

                            im = image[idx].numpy()
                            gener_sample(im, boxes)
                            gener_sample(im, target_label[idx])

                if count % 1500 == 0:
                    num_ = int(count // 1500)
                    model_dir = "%s/%s" % (FLAGS.save_checkpoint, num_)

                    if not os.path.isdir(model_dir):
                        os.makedirs(model_dir)
                        print("Make {} files to save checkpoint".format(num_))
                    ckpt = tf.train.Checkpoint(model=model, optim=optim)
                    ckpt_dir = model_dir + "/" + "Yolo_v1_{}.ckpt".format(count)

                    ckpt.save(ckpt_dir)

                count += 1

if __name__ == "__main__":
     main()
