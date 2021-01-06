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

import h5py, tqdm, spacy, misc.verb, re
from misc.refer import REFER

class GtLoader(Loader):

    def __init__(self, data_json, opt, tree_pth=None):
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
        # if tree_pth:
            # # self.trees = torch.load(tree_pth, 'r')
            # ## changed
            # with open(tree_pth, 'rb') as fp:
                # self.trees = pickle.load(fp)
        # else:
            # self.trees = None
        
        # for image_id, image in self.Images.items():
            # split = self.Refs[image['ref_ids'][0]]['split']

            # if split not in self.split_ix:
                # self.split_ix[split] = []
                # self.iterators[split] = 0

            # # add sentences to each subsets
            # sent_ids = []
            # for ref_id in self.Images[image_id]['ref_ids']:
                # sent_ids += self.Refs[ref_id]['sent_ids']
            # self.split_ix[split].append(image_id)

        # for k, v in self.split_ix.items():
            # print('assigned %d images to split %s' % (len(v), k))

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

    # shuffle split
    def shuffle(self, split):
        random.shuffle(self.split_ix[split])

    # reset iterator
    def reset_iterator(self, split):
        self.iterators[split] = 0

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

    def doc_prune(self, doc):
        new = []
        old = []

        for token in doc:
            if token.pos_ in ['DET', 'SYM', 'SPACE']:
                continue
            else:
                old.append(token.lower_)
                new.append(token.lower_)

        new_sent = " ".join(new)
        new_sent = re.sub(' +',' ', new_sent)

        old_sent = " ".join(old)
        old_sent = re.sub(' +',' ', old_sent)

        return new_sent.rstrip(), new, old_sent.rstrip(), old

    def doc_to_tree(self, doc):
        def traversal(node, sent_list=[]):
            txt = {}
            txt[(node.i, node.text, node.pos_, node.tag_, node.dep_)] = []

            sent_list.append((node.i, node.text, node.pos_, node.tag_, node.dep_))
            for child in node.children:
                tree, sent_list = traversal(child, sent_list)
                txt[(node.i, node.text, node.pos_, node.tag_, node.dep_)].append(tree)
            return txt, sent_list

        # traversal root's every children
        tree = {}
        try:
            root = next(doc.sents).root
        except:
            tree[(0, "UNK", "UNK", "UNK", "UNK")] = []
            print("a NULL sentence: {}".format(doc))
            return tree, [(0, "UNK", "UNK", "UNK", "UNK")]  # cause there a blank sentence in data

        # start traversal
        tree[(root.i, root.text, root.pos_, root.tag_, root.dep_)] = []
        sent_list = []
        sent_list.append((root.i, root.text, root.pos_, root.tag_, root.dep_))
        for child in root.children:
            t, sent_list = traversal(child, sent_list)
            tree[(root.i, root.text, root.pos_, root.tag_, root.dep_)].append(t)

        return tree, sent_list

    def load_sent(self, sent):
        nlp = spacy.load('en_core_web_sm', disable=['ner'])
        print(nlp)

        new_sents = []
        for idx, doc in enumerate(nlp.pipe([sent], disable = ['parser'])):
            new_sent, new, old_sent, old = self.doc_prune(doc)
            new_sents.append(new_sent)

        tree_list = []
        token_list = []
        tag_list = []
        dep_list = []
        for idx, doc in enumerate(nlp.pipe(new_sents)):
            t, s = self.doc_to_tree(doc)
            tree_list.append(t)
            s = sorted(s, key = lambda x: x[0])
            token = [x[1] for x in s]
            tag = [x[3] for x in s]
            dep = [x[4] for x in s]
            token_list.append(token)
            tag_list.append(tag)
            dep_list.append(dep)

        tree = {'tree': tree_list[0], 'tokens': token_list[0], 'tags': tag_list[0], 'deps': dep_list[0]}
        return tree

    # get one of data
    def get_data(self, img_path, sent):
        # options
        # split_ix = self.split_ix[split]
        # max_index = len(split_ix) - 1  # don't forget to -1
        wrapped = False

        # get information about this batch image
        # image_id = split_ix[self.iterators[split]]
        # ann_ids = self.Images[image_id]['ann_ids']

        # fetch sentences
        sent_ids = []
        gt_ixs = []
        sents = []
        trees = []
        vis = []

        # print("\n\nimage_id", image_id, self.Images[image_id])
        # img = Image.open(img_path).convert('RGB')
        img = Image.open(img_path)
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
        print(boxes)

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
        print(boxes)
        ann_boxes = torch.Tensor(boxes)
        # print(ann_boxes, "********")
        v = self.roi_pool(layer74, ann_boxes)
        # print(np.shape(v))
        v = v.reshape(1, v.size(0), v.size(1) * v.size(2) * v.size(3))
        print(np.shape(v))
        vis = v.detach()
        # convert to tensor
        vis = torch.from_numpy(np.asarray(vis)).cuda()
        # gt_ixs = torch.from_numpy(np.asarray(gt_ixs)).cuda()

        # update iter status
        # if self.iterators[split] + 1 > max_index:
            # self.iterators[split] = 0
            # wrapped = True
        # else:
            # self.iterators[split] += 1

        tree = self.load_sent(sent)

        # # return
        data = {}
        data['vis'] = vis.float()  # (num_bboxs, fc7_dim)
        data['sents'] = [tree]
        data['trees'] = [tree['tree']]

        # data['gts'] = gt_ixs.long()   # (num_sents, )
        # data['sent_ids'] = sent_ids
        # data['ann_ids'] = ann_ids
        # data['bounds'] = {'it_pos_now': self.iterators[split],
                          # 'it_max': max_index,
                          # 'wrapped': wrapped}
        print(np.shape(data['vis']), data.keys(), data)
        return data
