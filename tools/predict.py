import os, sys, time, random, json, logging, numpy as np
import torch
this_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(this_dir, '../'))

import models
import utils.eval_utils as eval_utils
import utils.train_utils as train_utils
from utils.gt_loader_test import GtLoader
import tools.opts as opts
from misc.glove import load_glove

import h5py, tqdm, spacy, misc.verb, re
from misc.refer import REFER

from yolov3.utils import *
from yolov3.image import letterbox_image, correct_yolo_boxes
from yolov3.darknet import Darknet
from ROI import *
from PIL import Image, ImageDraw

def doc_prune(doc):
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

def doc_to_tree(doc):
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

def load_sent(sent):
    nlp = spacy.load('en_core_web_sm', disable=['ner'])
    print(nlp)
    
    new_sents = []
    for idx, doc in enumerate(nlp.pipe([sent], disable = ['parser'])):
        new_sent, new, old_sent, old = doc_prune(doc)
        new_sents.append(new_sent)
    
    tree_list = []
    token_list = []
    tag_list = []
    dep_list = []
    for idx, doc in enumerate(nlp.pipe(new_sents)):
        t, s = doc_to_tree(doc)
        tree_list.append(t)
        s = sorted(s, key = lambda x: x[0])
        token = [x[1] for x in s]
        tag = [x[3] for x in s]
        dep = [x[4] for x in s]
        token_list.append(token)
        tag_list.append(tag)
        dep_list.append(dep)
    
    tree = {'tree': tree_list[0], 'tokens': token_list[0], 'tags': tag_list[0], 'deps': dep_list[0]} 
    print(tree)

if __name__ == "__main__":
    opt = opts.parse_opt()

    torch.manual_seed(opt.seed)
    random.seed(opt.seed)

    opt.dataset_split_by = opt.dataset + '_' + opt.split_by
    data_json = os.path.join(opt.feats_path, opt.dataset_split_by, opt.data_file + '.json')
    opt.data_file = 'data_dep_dict'
    data_pth = os.path.join(opt.feats_path, opt.dataset_split_by, opt.data_file + '.pth')
    loader = GtLoader(data_json, opt, data_pth)

    opt.word_vocab_size = loader.word_vocab_size
    opt.vis_dim = loader.vis_dim
    opt.tag_vocab_size = loader.tag_vocab_size
    opt.dep_vocab_size = loader.dep_vocab_size

    img_path = "data/VG_100K_2/" + opt.img
    model_path = ""
    sent = opt.sent
    print(img_path, sent)
    
    opt.start_from = "log/refcocog_umd_det_nmtree_01"
    with open(os.path.join(opt.start_from, 'infos.json'), 'r') as f:
        infos = json.load(f)
    print('Model loaded succesfully..')

    model = models.setup(opt, loader).cuda()
    crit = torch.nn.NLLLoss()

    assert os.path.isfile(os.path.join(opt.start_from, "model.pth"))
    model.load_state_dict(torch.load(os.path.join(opt.start_from, 'model.pth')))
    print('Model Loaded successfully !')

    glove_weight = load_glove(glove=opt.glove, vocab=loader.word_to_ix, opt=opt)
    assert glove_weight.shape == model.word_embedding.weight.size()
    model.word_embedding.weight.data.set_(torch.cuda.FloatTensor(glove_weight))
    print("Load word vectors ...")

    tic = time.time()
    wrapped = False
    data = loader.get_data(img_path, sent)
    torch.cuda.synchronize()
    scores = model(data)
    print("scores", scores, torch.argmax(scores))
