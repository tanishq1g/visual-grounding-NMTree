import random
import numpy as np

import torch
import pickle
from functools import partial
import sys
import time
from PIL import Image, ImageDraw
sys.path.append(sys.path[0] + "/..")
sys.path.append(sys.path[0] + "/../yolov3")
print(sys.path)
from yolov3.utils import *
from yolov3.image import letterbox_image, correct_yolo_boxes
from yolov3.darknet import Darknet
from ROI import *

class YOLOROI:

    def __init__(self):
        print("Loading Yolov3 model")
        self.cfgfile = "yolov3/cfg/yolo_v3.cfg"
        self.weightfile = "yolov3/yolov3.weights"
        self.yolo = Darknet(self.cfgfile)
        # # self.yolo.print_network()
        self.yolo.load_weights(self.weightfile)
        print('Loading weights from %s... Done!' % (self.weightfile))
        self.yolo.cuda()

        print("Loading ROI Pool")
        self.roi_pool = TorchROIPool(2, 1.0)

    # get reverse coordinates for yolov3 model
    def get_coordinates(self, boxes, im_w, im_h):

        # im_w, im_h = float(im_w), float(im_h)
        # net_w, net_h = float(net_w), float(net_h)
        # if net_w/im_w < net_h/im_h:
            # new_w = net_w
            # new_h = (im_h * net_w)/im_w
        # else:
            # new_w = (im_w * net_h)/im_h
            # new_h = net_h

        # xo, xs = (net_w - new_w)/(2*net_w), net_w/new_w
        # yo, ys = (net_h - new_h)/(2*net_h), net_h/new_h

        b = boxes
        # from x1, y1, w, h to x1, y1, x2, y2
        # b[2] = b[0] + b[2]
        # b[3] = b[1] + b[3]

        target_size = 13
        x_scale = target_size / im_w
        y_scale = target_size / im_h

        b[0] = int(np.round(b[0] * x_scale))
        b[1] = int(np.round(b[1] * y_scale))
        b[2] = int(np.round(b[2] * x_scale))
        b[3] = int(np.round(b[3] * y_scale))
        # from x1, y1, x2, y2 to correct yolo coord
        # b[0] = (b[2] + b[0]) / (im_w * 2.0)
        # b[2] = (b[2] - b[0]) / im_w
        # b[1] = (b[3] + b[1]) / (im_h * 2.0)
        # b[3] = (b[3] - b[1]) / im_h
        # from correct yolo coord to mode layers coord
        # b[0] = b[0] / xs + xo
        # b[1] = b[1] / ys + yo
        # b[2] /= xs
        # b[3] /= ys
        return b

    def run_yolo(self, img_path):
        img = Image.open(img_path).convert('RGB')
        sized = letterbox_image(img, self.yolo.width, self.yolo.height)
        use_cuda = True

        boxes, layer74 = do_detect(self.yolo,  sized, 0.1, 0.4, use_cuda)
        print("Layer 74", np.shape(layer74))
        correct_yolo_boxes(boxes, img.width, img.height, self.yolo.width, self.yolo.height)
        print("boxes ", boxes)

        width = img.width
        height = img.height
        for box in boxes:
            box[0], box[1], box[2], box[3] = (box[0] - box[2]/2.0) * width, (box[1] - box[3]/2.0) * height, (box[0] + box[2]/2.0) * width, (box[1] + box[3]/2.0) * height
            # print(x1, y1, x2, y2)
            # box = [x1, y1, x2, y2]
        # print(boxes)

        for box in boxes:
            box = self.get_coordinates(box, img.width, img.height)
            if box[0] == box[2]:
                if box[2] != 13:
                    box[2] += 1
                else:
                    box[0] -= 1
            if box[1] == box[3]:
                if box[3] != 13:
                    box[3] += 1
                else:
                    box[1] -= 1
        # print(boxes)

        ann_boxes = torch.Tensor(boxes)
        # print(ann_boxes, "********")
        v = self.roi_pool(layer74, ann_boxes)
        # print(np.shape(v))
        v = v.reshape(1, v.size(0), v.size(1) * v.size(2) * v.size(3))
        print(np.shape(v))
        vis = v.detach()
        # convert to tensor
        vis = torch.from_numpy(np.asarray(vis)).cuda()
        return vis
if __name__ == '__main__':
    yo = YOLOROI()
    img_path = "yolov3/data/person.jpg" 
    vis = yo.run_yolo(img_path)
    print(vis)
