from collections import Counter

import tensorflow as tf

weight_decay = 0.0000001

architecture_config = [
    (7, 64, 2, 3),
    "M",
    (3, 192, 1, 1),
    "M",
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    "M",
    [(1, 256, 1, 0), (3, 512, 1, 1), 4],
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    "M",
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1),
]

class CNNBlock(tf.keras.layers.Layer):
    def __init__(self, filters, bias=False, **kwargs):
        super(CNNBlock, self).__init__()
        self.conv = tf.keras.layers.Conv2D(filters=filters,
                                            use_bias=bias,
                                            **kwargs)
        self.batchnorm = tf.keras.layers.BatchNormalization()
        self.leakrelu = tf.keras.layers.LeakyReLU(0.1)

    def call(self, inputs):
        return self.leakrelu(self.batchnorm(self.conv(inputs)))

def YoloV1():
    h = inputs = tf.keras.Input((448, 448, 3))
    for x in architecture_config:
        if type(x) == tuple:
            if x[0] // 2 == x[3]:
                padding="valid"
                h = tf.pad(h, [[0,0],[x[3], x[3]],[x[3], x[3]],[0,0]])
            h = CNNBlock(x[1],
                kernel_size=x[0],
                strides=x[2],
                padding=padding,)(h)
        elif type(x) == str:
            h = tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2), padding="same")(h)

        elif type(x) == list:
            conv1 = x[0]
            conv2 = x[1]
            num_repeats = x[2]

            for _ in range(num_repeats):
                h = CNNBlock(conv1[1],
                            kernel_size=conv1[0],
                            strides=conv1[2],
                            padding="same",)(h)
                h = CNNBlock(conv2[1],
                            kernel_size=conv2[0],
                            strides=conv2[2],
                            padding="same",)(h)

    h = tf.keras.layers.Flatten()(h)
    h = tf.keras.layers.Dense(496)(h)
    h = tf.keras.layers.LeakyReLU(0.1)(h)
    h = tf.keras.layers.Dense(7*7*(20+2*5))(h)

    return tf.keras.Model(inputs=inputs, outputs=h)

def cal_mAP(pred_boxes, 
            true_boxes, 
            iou_threshold=0.5,
            num_classes=20):

    average_precisions = []
    epsilon = 1e-6

    # True positive | False positive (positive - 검출 되어야할것, negative --> 검출하지 않아도되는것)
    # False negative | True negative
    
    # TP: positive한 데이터가 옳바르게 검출된 것 | FP: negative한 데이터가 potitive로 잘못 검출 한 것 (오검출 1)
    # FN: positive한 데이터가 negative로 잘못 검출된 것 (오검출 2) |  TN: negative한 데이터가 negative로 검출된 것

    # precision = TP / (TP + FP)
    # Recall = TP / (TP + FN)

    for c in range(20):
        detections = []
        ground_truths = []

        for pred_box in pred_boxes:
            if pred_box[1] == c:
                detections.append(pred_box)

        for true_box in true_boxes:
            if true_box[1] == c:
                ground_truths.append(true_box)

        amount_bboxes = Counter([gt[0] for gt in ground_truths])

        for key, val in amount_bboxes.items():
            amount_bboxes[key] = tf.zeros(val)

        detections.sort(key=lambda x: x[2], reverse=True)
        TP = tf.zeros(len(detections))
        FP = tf.zeros(len(detections))
        total_true_boxes = len(ground_truths)

        if total_true_boxes == 0:
            continue




    return sum
