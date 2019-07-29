# encoding: utf-8
"""
@author:  liaoxingyu
@contact: liaoxingyu2@jd.com
"""

import torchvision.transforms as T
from PIL import Image

from .transforms import RandomErasing


class PadShortSide(object):
    def __init__(self):
        pass

    def __call__(self, x):
        h, w = x.size[-2:]
        s = max(h, w)
        new_im = Image.new("RGB", (s, s), color=(124, 116, 104))
        new_im.paste(x, ((s-h)//2, (s-w)//2))
        return new_im


def build_transforms(cfg, is_train=True):
    normalize_transform = T.Normalize(
        mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    if is_train:
        transform = T.Compose([
            PadShortSide(),
            T.RandomAffine(
                degrees=5, translate=(0.1, 0.1), scale=(0.9, 1.1), fillcolor=(124, 116, 104)),
            T.Resize(cfg.INPUT.SIZE_TRAIN),
            T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
            T.ToTensor(),
            normalize_transform,
            RandomErasing(probability=cfg.INPUT.RE_PROB,
                          mean=cfg.INPUT.PIXEL_MEAN)
        ])
    else:
        transform = T.Compose([
            PadShortSide(),
            T.Resize(cfg.INPUT.SIZE_TEST),
            T.ToTensor(),
            normalize_transform
        ])

    return transform
