# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import os

import torch


def train_collate_fn(batch):
    imgs, pids, _, _, = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids


def val_collate_fn(batch):
    imgs, pids, camids, _ = zip(*batch)
    return torch.stack(imgs, dim=0), pids, camids

def test_collate_fn(batch):
    imgs, pids, camids, paths = zip(*batch)
    basenames = [os.path.basename(path) for path in paths]
    return torch.stack(imgs, dim=0), pids, camids, basenames
