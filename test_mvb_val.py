# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import argparse
import os
import sys
from os import mkdir

import numpy as np
import torch
from torch.backends import cudnn

from config import cfg
from data import make_test_data_loader
from engine.inference import inference
from modeling import build_model
from utils.logger import setup_logger
from utils.reid_metric import R1_mAP


def main():
    parser = argparse.ArgumentParser(description="ReID Baseline Inference")
    parser.add_argument(
        "--config_file", default="./configs/softmax_triplet.yml", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]
                   ) if "WORLD_SIZE" in os.environ else 1

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.DATASETS.NAMES = 'mvb'
    cfg.DATASETS.ROOT_DIR = './data'
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        mkdir(output_dir)

    logger = setup_logger("reid_baseline", output_dir, 0)
    logger.info("Using {} GPUS".format(num_gpus))
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    if cfg.MODEL.DEVICE == "cuda":
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID
    cudnn.benchmark = True

    # init dataloader
    gallery_loader, query_loader, num_query, num_classes = make_test_data_loader(
        cfg)
    # build model and load checkpoint param
    model = build_model(cfg, num_classes)
    model.load_param(cfg.TEST.WEIGHT)

    model.eval().cuda()
    feats = []
    g_pids = []
    g_camids = []
    g_names = []
    q_names = []
    # ===== extract feats =====
    with torch.no_grad():
        print('extract query feats...')
        for batch in query_loader:
            data, _, _, paths = batch
            feat = model(data.cuda())
            feats.append(feat.cpu())
            q_names.extend(paths)
        print('extract gallery feats...')
        for batch in gallery_loader:
            data, pids, camids, paths = batch
            g_pids.extend(pids)
            g_camids.extend(camids)
            feat = model(data.cuda())
            feats.append(feat.cpu())
            g_names.extend(paths)

    # ===== init vars =====
    feats = torch.cat(feats, dim=0) # cat feats because feats is batch-wised
    feats = torch.nn.functional.normalize(feats, dim=1, p=2) # normalize feats
    qf = feats[:num_query]          # query feats
    gf = feats[num_query:]          # gallery feats
    g_pids = np.array(g_pids)       # gallery pids
    g_camids = np.array(g_camids)   # gallery camids
    # ===== calc euclidean distance between gallery feat and query feat =====
    m, n = qf.shape[0], gf.shape[0]
    distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    distmat.addmm_(1, -2, qf, gf.t())
    distmat = distmat.cpu().numpy()

    # ===== find min distance of every query during id-wised =====
    num_q, num_g = distmat.shape
    indices = np.argsort(distmat, axis=1)   # sort distmat from low to high (distance)
    q_nameinds = np.zeros((num_q), dtype=int)  # init query ids array for sort because query files is not ordered
    distmat_id_wised = np.ones((num_q, num_classes), dtype=np.float32)*100
    def get_id(x):
        return int(x.split('_')[0])
    for q_idx in range(num_q):
        order = indices[q_idx]
        q_nameinds[q_idx] = int(q_names[q_idx].split('.')[0])  # get query id frome query img filename
        names = np.array(g_names)[order].tolist()
        pids = map(get_id, names)
        dists = distmat[q_idx][order]
        # find min distance of current query during id-wised
        for pid, dist in zip(pids, dists):
            if distmat_id_wised[q_idx, pid] > dist:
                distmat_id_wised[q_idx, pid] = dist
    
    # ===== sort query id from 0 to query nums =====
    orders = np.argsort(q_nameinds, axis=0)
    q_nameinds = q_nameinds[orders]
    distmat_id_wised = distmat_id_wised[orders]

    # ===== write result to csv =====
    with open('../024_bag_result.csv', 'w') as f:
        for q_idx in range(num_q):
            order = np.argsort(distmat_id_wised[q_idx])
            max_dist = distmat_id_wised[q_idx].max()
            buf = '%05d,' % q_nameinds[q_idx]
            for ind in order:
                score = (max_dist-distmat_id_wised[q_idx][ind])/max_dist
                buf += '%04d,%.6f,' % (ind, score)
            buf = buf[:-1]
            f.write(buf+'\n')


if __name__ == '__main__':
    main()
