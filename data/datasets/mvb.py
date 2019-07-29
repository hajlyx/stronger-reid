# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import glob
import re

import os.path as osp

from .bases import BaseImageDataset


class MVB(BaseImageDataset):

    dataset_dir = 'mvb'

    def __init__(self, root='/home/haoluo/data', verbose=True, **kwargs):
        super(MVB, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)

        self._check_before_run()

        query, gallery = self._process_dir(self.dataset_dir, relabel=False)

        self.query = query
        self.gallery = gallery

        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(
            self.gallery)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError(
                "'{}' is not available".format(self.dataset_dir))

    def _process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))

        pid_container = set()
        for img_path in img_paths:
            if not '_' in img_path:
                continue
            pid = int(img_path.split('/')[-1].split('_')[0])
            if pid == -1:
                continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        query = []
        gallery = []
        for img_path in img_paths:
            if not '_' in img_path:
                query.append(img_path)
                continue
            add = 0
            pid = int(img_path.split('/')[-1].split('_')[0])
            if img_path.split('/')[-1].split('.')[0].split('_')[1] == 'p':
                add += 10
            camid = int(img_path.split('/')[-1].split('.')[0].split('_')[2]) + add

            if pid == -1:
                continue  # junk images are just ignored
            camid -= 1  # index starts from 0
            if relabel:
                pid = pid2label[pid]
            gallery.append((img_path, pid, camid))

        return query, gallery
