from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import struct
import sys

import cv2
import numpy as np
import sklearn
import torch
from sklearn.preprocessing import normalize

from backbones import get_model
from utils.utils_config import get_config


def read_img(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    return img


def get_feature(imgs, nets):
    count = len(imgs)
    data = np.zeros(shape=(count * 2, 3, imgs[0].shape[0],
                           imgs[0].shape[1]))
    for idx, img in enumerate(imgs):
        img = img[:, :, ::-1]  # to rgb
        img = np.transpose(img, (2, 0, 1))
        for flipid in [0, 1]:
            _img = np.copy(img)
            if flipid == 1:
                _img = _img[:, :, ::-1]
            _img = np.array(_img)
            data[count * flipid + idx] = _img

    F = []
    data = torch.from_numpy(data).float()
    x = nets.forward(data)
    embedding = x[0:count, :] + x[count:, :]
    embedding = sklearn.preprocessing.normalize(embedding.cpu().detach().numpy())
    # print('emb', embedding.shape)
    F.append(embedding)
    F = np.concatenate(F, axis=1)
    F = sklearn.preprocessing.normalize(F)
    # print('F', F.shape)
    return F


def write_bin(path, feature):
    feature = list(feature)
    with open(path, 'wb') as f:
        f.write(struct.pack('4i', len(feature), 1, 4, 5))
        f.write(struct.pack("%df" % len(feature), *feature))


def get_and_write(buffer, nets):
    imgs = []
    for k in buffer:
        imgs.append(k[0])
    features = get_feature(imgs, nets)
    # print(np.linalg.norm(feature))
    assert features.shape[0] == len(buffer)
    for ik, k in enumerate(buffer):
        out_path = k[1]
        feature = features[ik].flatten()
        write_bin(out_path, feature)


def main(args):
    parser = argparse.ArgumentParser(description='Get configurations')
    parser.add_argument('--config', default="configs/MS1MV3", help='the name of config file')
    cfg_args = parser.parse_args()
    cfg = get_config(cfg_args.config)
    model_path = cfg.output + "/model5.pt"
    weight = torch.load(model_path)
    net = get_model("r50", dropout=0, fp16=False).cuda()
    net.load_state_dict(weight)

    model = torch.nn.DataParallel(net)
    model.eval()

    facescrub_out = os.path.join(args.output, 'facescrub')
    megaface_out = os.path.join(args.output, 'megaface')

    i = 0
    succ = 0
    buffer = []
    for line in open(args.facescrub_lst, 'r'):
        if i % 1000 == 0:
            print("writing fs", i, succ)
        i += 1
        image_path = line.strip()
        _path = image_path.split('/')
        a, b = _path[-2], _path[-1]
        out_dir = os.path.join(facescrub_out, a)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        image_path = os.path.join(args.facescrub_root, image_path)
        img = read_img(image_path)
        if img is None:
            print('read error:', image_path)
            continue
        out_path = os.path.join(out_dir, b + "_%s.bin" % (args.algo))
        item = (img, out_path)
        buffer.append(item)
        if len(buffer) == args.batch_size:
            get_and_write(buffer, model)
            buffer = []
        succ += 1
    if len(buffer) > 0:
        get_and_write(buffer, model)
        buffer = []
    print('fs stat', i, succ)

    i = 0
    succ = 0
    buffer = []
    for line in open(args.megaface_lst, 'r'):
        if i % 1000 == 0:
            print("writing mf", i, succ)
        i += 1
        image_path = line.strip()
        _path = image_path.split('/')
        a1, a2, b = _path[-3], _path[-2], _path[-1]
        out_dir = os.path.join(megaface_out, a1, a2)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
            # continue
        # print(landmark)
        image_path = os.path.join(args.megaface_root, image_path)
        img = read_img(image_path)
        if img is None:
            print('read error:', image_path)
            continue
        out_path = os.path.join(out_dir, b + "_%s.bin" % (args.algo))
        item = (img, out_path)
        buffer.append(item)
        if len(buffer) == args.batch_size:
            get_and_write(buffer, model)
            buffer = []
        succ += 1
    if len(buffer) > 0:
        get_and_write(buffer, model)
        buffer = []
    print('mf stat', i, succ)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', type=int, help='', default=8)
    parser.add_argument('--image_size', type=str, help='', default='3,112,112')
    parser.add_argument('--gpu', type=int, help='', default=0)
    parser.add_argument('--algo', type=str, help='', default='insightface')
    parser.add_argument('--facescrub-lst',
                        type=str,
                        help='',
                        default=r'E:\DATASET\megaface\data\facescrub_lst')
    parser.add_argument('--megaface-lst',
                        type=str,
                        help='',
                        default=r'E:\DATASET\megaface\data\megaface_lst')
    parser.add_argument('--facescrub-root',
                        type=str,
                        help='',
                        default=r'E:\DATASET\megaface\data\megaface_testpack_v1.0\facescrub_images')
    parser.add_argument('--megaface-root',
                        type=str,
                        help='',
                        default=r'E:\DATASET\megaface\data\megaface_testpack_v1.0\megaface_images')
    parser.add_argument('--output', type=str, help='', default=r'E:\DATASET\megaface\data\feature_out')
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
