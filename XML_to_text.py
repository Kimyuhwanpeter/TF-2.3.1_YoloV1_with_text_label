# -*- code: utf-8 -*-
# reference: https://seongkyun.github.io/papers/2019/01/06/Object_detection/
from absl import flags, app
from os import listdir
from os.path import isfile, join

import numpy as np
import sys
import os
import xml.etree.ElementTree as ET

img_mean = (123.675, 116.28, 103.53)
# img_std = (58.395, 57.12, 57.375)
img_std = (1., 1., 1.)

flags.DEFINE_string('txt_path', 'D:/[2]DB/celebA/list_bbox_celeba.txt', 'Training text path')

flags.DEFINE_string('img_path', 'D:/[2]DB/celebA/img_align_celeba/', 'Training image path')

FLAGS = flags.FLAGS
FLAGS(sys.argv)

# feature를 뽑고 앵커박스로 윈도우를 하며 윈도우에 대한 classification 을 진행-> 그리고 그 roi안에 존재하는 클래스를 classification

def main(argv=None):

    dataA = [f for f in listdir('D:/[1]DB/[3]detection_DB/PascalVoc2012/pascal_voc_2012/VOC2012/Annotations') if isfile(join('D:/[1]DB/[3]detection_DB/PascalVoc2012/pascal_voc_2012/VOC2012/Annotations', f))]
    dataA = ['D:/[1]DB/[3]detection_DB/PascalVoc2012/pascal_voc_2012/VOC2012/Annotations/' + dataA_ for dataA_ in dataA]

    #write_anno = open('D:/[2]DB/PascalVOC/VOCdevkit/VOC2012/Annotations_text/label.txt', 'w')
    for i in range(len(dataA)):
        text = (dataA[i].split('/')[7]).split('.')[0] + '.txt'
        filename = 'D:/[1]DB/[3]detection_DB/PascalVoc2012/pascal_voc_2012/VOC2012/JPEGImages/' + (dataA[i].split('/')[7]).split('.')[0] + '.jpg'
        write_anno = open('D:/[1]DB/[3]detection_DB/PascalVoc2012/pascal_voc_2012/VOC2012/Annotations_text/' + text, 'w')

        tree = ET.parse(dataA[i])
        root = tree.getroot()

        for element in root.findall('size'):
            height = element.find('height').text
            width = element.find('width').text
    
        #write_anno.write(filename)
        #write_anno.write(' ')

        for element in root.findall('object'):
            print('=================')
            #write_anno.write(filename)
            #write_anno.write(' ')
            name = element.find('name').text
            if name == 'person':
                cla = 0
            elif name == 'bird':
                cla = 1
            elif name == 'cat':
                cla = 2
            elif name == 'cow':
                cla = 3
            elif name == 'dog':
                cla = 4
            elif name == 'horse':
                cla = 5
            elif name == 'sheep':
                cla = 6
            elif name == 'aeroplane':
                cla = 7
            elif name == 'boat':
                cla = 8
            elif name == 'bus':
                cla = 9
            elif name == 'car':
                cla = 10
            elif name == 'motorbike':
                cla = 11
            elif name == 'train':
                cla = 12
            elif name == 'bottle':
                cla = 13
            elif name == 'chair':
                cla = 14
            elif name == 'dining':
                cla = 15
            elif name == 'table':
                cla = 16
            elif name == 'potted plane':
                cla = 17
            elif name == 'sofa':
                cla = 18
            elif name == 'tv/monitor':
                cla = 19
            print(name)
            print(cla)

            xmin = element.find('bndbox').find('xmin').text
            ymin = element.find('bndbox').find('ymin').text
            xmax = element.find('bndbox').find('xmax').text
            ymax = element.find('bndbox').find('ymax').text

            xmin, ymin, xmax, ymax = int(float(xmin)), int(float(ymin)), int(float(xmax)), int(float(ymax))
            

            #x = ( int(xmax) + int(xmin) ) // 2
            #y = ( int(ymax) + int(ymin) ) // 2

            #w = ( int(xmax) - int(xmin) ) / (width)
            #h = ( int(ymax) - int(ymin) ) / height

            #write_anno.write(filename)
            #write_anno.write(' ')
            #height_rate = (448 / int(height))
            #width_rate = (448 / int(width))

            #xmin = int(int(xmin) * width_rate)
            #xmax = int(int(xmax) * width_rate)
            #ymin = int(int(ymin) * height_rate)
            #ymax = int(int(ymax) * height_rate)

            write_anno.write(str(xmin))
            write_anno.write(',')
            write_anno.write(str(ymin))
            write_anno.write(',')
            write_anno.write(str(xmax))
            write_anno.write(',')
            write_anno.write(str(ymax))
            write_anno.write(',')

            write_anno.write(str(height))
            write_anno.write(',')
            write_anno.write(str(width))
            write_anno.write(',')
            write_anno.write(str(cla))
            write_anno.write('\n')
            print('=================')
        
        #write_anno.write('\n')
        #write_anno.close()

if __name__ == '__main__':
    app.run(main)
