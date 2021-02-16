import numpy as np
import os
import cv2

class BoundBox:
    def __init__(self, xmin, ymin, xmax, ymax, c = None, classes = None):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax

        self.c       = c
        self.classes = classes

        self.label = -1
        self.score = -1

    def get_label(self):
        if self.label == -1:
            self.label = np.argmax(self.classes)

        return self.label

    def get_score(self):
        if self.score == -1:
            self.score = self.classes[self.get_label()]

        return self.score

def _interval_overlap(interval_a, interval_b):
    x1, x2 = interval_a
    x3, x4 = interval_b

    if x3 < x1:
        if x4 < x1:
            return 0
        else:
            return min(x2,x4) - x1
    else:
        if x2 < x3:
             return 0
        else:
            return min(x2,x4) - x3

def bbox_iou(box1, box2):
    intersect_w = _interval_overlap([box1.xmin, box1.xmax], [box2.xmin, box2.xmax])
    intersect_h = _interval_overlap([box1.ymin, box1.ymax], [box2.ymin, box2.ymax])

    intersect = intersect_w * intersect_h

    w1, h1 = box1.xmax-box1.xmin, box1.ymax-box1.ymin
    w2, h2 = box2.xmax-box2.xmin, box2.ymax-box2.ymin

    union = w1*h1 + w2*h2 - intersect

    if (union <= 0):
        union = union + 1e-9

    return float(intersect) / union

def draw_boxes_2(image, boxes, obj_thresh = 0.5, quiet=True):
    for box in boxes:
        label_str = ''
        label = -1

        if box.classes[0] > obj_thresh:
            if label_str != '': label_str += ', '
            label_str += (str(round(box.get_score()*100, 2)) + '%')
            label = 0
        if not quiet: print(label_str)

        if label >= 0:
            text_size = cv2.getTextSize(label_str, cv2.FONT_HERSHEY_SIMPLEX, 0.5e-3 * image.shape[0], 5)
            width, height = text_size[0][0], text_size[0][1]
            region = np.array([[box.xmin-3,        box.ymin],
                               [box.xmin-3,        box.ymin-height-26],
                               [box.xmin+width+13, box.ymin-height-26],
                               [box.xmin+width+13, box.ymin]], dtype='int32')

            cv2.rectangle(img=image, pt1=(box.xmin,box.ymin), pt2=(box.xmax,box.ymax), color=(0, 255, 25), thickness=3)
            cv2.fillPoly(img=image, pts=[region], color=(0, 255, 25))
            cv2.putText(img=image,
                        text=label_str,
                        org=(box.xmin+13, box.ymin - 13),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5e-3 * image.shape[0],
                        color=(0,0,0),
                        thickness=2)

    return image

def draw_boxes(image, boxes, show_score = True, color = (0, 255, 0)):
    for box in boxes:
        cv2.rectangle(image, (box.xmin,box.ymin), (box.xmax,box.ymax), color, 3)

        if show_score:
            cv2.putText(image,
                        str(round(box.get_score()*100, 2)) + '%',
                        (box.xmin, box.ymin - 9),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8e-3 * image.shape[0],
                        color, 2)

    return image
