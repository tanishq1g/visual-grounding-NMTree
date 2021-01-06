from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import numpy as np
from utils.loader import Loader

import torch
import pickle
from functools import partial
import sys
import time
from PIL import Image, ImageDraw
sys.path.append(sys.path[0] + "/../yolov3")
# print(sys.path)
from yolov3.utils import *
from yolov3.image import letterbox_image, correct_yolo_boxes
from yolov3.darknet import Darknet
from ROI import *

class GtLoader(Loader):

    def __init__(self, data_json, visual_feats_dir, opt, tree_pth=None):
        # parent loader instance, see loader.py
        Loader.__init__(self, data_json)
        # print('data_json, visual_feats_dir, opt, tree_pth', data_json, visual_feats_dir, opt, tree_pth)
        self.opt = opt
        self.batch_size = opt.batch_size
        # self.batch_size = opt["batch_size"]
        # self.vis_dim = 2048 + 512 + 512
        self.vis_dim = 4096 

        # img_iterators for each split
        self.split_ix = {}
        self.iterators = {}

        # load ann feats
        # print('loading visual feats ...')
        # pickle.load = partial(pickle.load, encoding="latin1")
        # pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")
        # self.visual_feats = torch.load(visual_feats_dir,
            # map_location=lambda storage, loc: storage, pickle_module=pickle)

        # for ke in self.visual_feats:
            # print(ke, np.shape(self.visual_feats[ke]))
        # print('visual feats loaded')
        if tree_pth:
            # self.trees = torch.load(tree_pth, 'r')
            ## changed
            with open(tree_pth, 'rb') as fp:
                self.trees = pickle.load(fp)
        else:
            self.trees = None
        
        for image_id, image in self.Images.items():
            split = self.Refs[image['ref_ids'][0]]['split']

            if split not in self.split_ix:
                self.split_ix[split] = []
                self.iterators[split] = 0

            # add sentences to each subsets
            sent_ids = []
            for ref_id in self.Images[image_id]['ref_ids']:
                sent_ids += self.Refs[ref_id]['sent_ids']
            self.split_ix[split].append(image_id)

        for k, v in self.split_ix.items():
            print('assigned %d images to split %s' % (len(v), k))

        print("Loading Yolov3 model")
        self.cfgfile = "yolov3/cfg/yolo_v3.cfg"
        self.weightfile = "yolov3/yolov3.weights"
        self.yolo = Darknet(self.cfgfile)
        # self.yolo.print_network()
        self.yolo.load_weights(self.weightfile)
        print('Loading weights from %s... Done!' % (self.weightfile))
        self.yolo.cuda()

        print("Loading ROI Pool")
        self.roi_pool = TorchROIPool(2, 1.0)

    # shuffle split
    def shuffle(self, split):
        random.shuffle(self.split_ix[split])

    # reset iterator
    def reset_iterator(self, split):
        self.iterators[split] = 0

    # get reverse coordinates for yolov3 model
    def get_coordinates(self, boxes, im_w, im_h, net_w = 416, net_h = 416):      
        
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
        
        b = boxes[:]
        # from x1, y1, w, h to x1, y1, x2, y2
        b[2] = b[0] + b[2]
        b[3] = b[1] + b[3]
        
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

    # get one of data
    def get_data(self, split):
        # options
        split_ix = self.split_ix[split]
        max_index = len(split_ix) - 1  # don't forget to -1
        wrapped = False

        # get information about this batch image
        image_id = split_ix[self.iterators[split]]
        ann_ids = self.Images[image_id]['ann_ids']

        # fetch sentences
        sent_ids = []
        gt_ixs = []
        sents = []
        trees = []
        vis = []

        # print("\n\nimage_id", image_id, self.Images[image_id])
        img = Image.open("data/images/train2014/" + self.Images[image_id]['file_name']).convert('RGB')
        # img = Image.open("yolov3/data/dog.jpg").convert('RGB')
        img = letterbox_image(img, self.yolo.width, self.yolo.height)
        img = image2torch(img)
        use_cuda = True
        img = img.to(torch.device("cuda" if use_cuda else "cpu"))

        out_boxes, layer74 = self.yolo(img)
        # print("Layer 74", np.shape(layer74))

        sent_count = 0
        for ref_id in self.Images[image_id]['ref_ids']:
            # print("ref_id", ref_id, self.Refs[ref_id],"***********")
            for sent_id in self.Refs[ref_id]['sent_ids']:
                sent_ids += [sent_id]
                gt_ixs += [ann_ids.index(self.Refs[ref_id]['ann_id'])]
                # vis += [self.visual_feats[sent_id]]

                # print(sent_id, np.shape(self.visual_feats[sent_id]), self.Sentences[sent_id])
                if self.trees:
                    trees += [self.trees[sent_id]['tree']]
                    sents += [self.trees[sent_id]]
                else:
                    sents += [self.Sentences[sent_id]]
                sent_count += 1

        ann_boxes = []
        for ann_id in self.Images[image_id]['ann_ids']:
            # print("ann_id", ann_id, self.Anns[ann_id])
            box = self.get_coordinates(self.Anns[ann_id]['box'], self.Images[image_id]['width'], self.Images[image_id]['height'])
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
            ann_boxes.append(box)
            # print(self.Anns[ann_id], box)
        ann_boxes = torch.Tensor(ann_boxes)
        # print(ann_boxes, "********")
        v = self.roi_pool(layer74, ann_boxes)
        # print(np.shape(v))
        v = v.reshape(1, v.size(0), v.size(1) * v.size(2) * v.size(3))
        # print(np.shape(v))
        v = torch.cat([v] * sent_count).cuda()
        vis = v.detach()
        # print(np.shape(v), v, v.requires_grad)
        # convert to tensor
        # vis = torch.from_numpy(np.asarray(vis)).cuda()
        gt_ixs = torch.from_numpy(np.asarray(gt_ixs)).cuda()

        # update iter status
        if self.iterators[split] + 1 > max_index:
            self.iterators[split] = 0
            wrapped = True
        else:
            self.iterators[split] += 1

        # return
        data = {}
        data['vis'] = vis.float()  # (num_bboxs, fc7_dim)
        data['sents'] = sents
        if self.trees:
            data['trees'] = trees

        data['gts'] = gt_ixs.long()   # (num_sents, )
        data['sent_ids'] = sent_ids
        data['ann_ids'] = ann_ids
        data['bounds'] = {'it_pos_now': self.iterators[split],
                          'it_max': max_index,
                          'wrapped': wrapped}
        print(image_id, np.shape(data['vis']), data.keys(), data)
        return data
