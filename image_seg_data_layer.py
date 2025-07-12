# image_seg_data_layer.py

import caffe
import numpy as np
import yaml
import cv2
import random
import os

class ImageSegDataLayer(caffe.Layer):
    def setup(self, bottom, top):
        params = yaml.safe_load(self.param_str)
        self.batch_size = params['batch_size']
        self.data_list = self._load_list(params['data_list'])  # txt 檔，每行一張圖+label
        self.root = params['root']  # VOC 根路徑
        self.index = 0

        top[0].reshape(self.batch_size, 3, 500, 500)  # image
        top[1].reshape(self.batch_size, 1, 500, 500)  # label

    def _load_list(self, list_file):
        with open(list_file) as f:
            lines = f.readlines()
        return [line.strip().split() for line in lines]

    def reshape(self, bottom, top):
        pass

    def forward(self, bottom, top):
        images = []
        labels = []
        for _ in range(self.batch_size):
            if self.index >= len(self.data_list):
                random.shuffle(self.data_list)
                self.index = 0

            img_path, label_path = self.data_list[self.index]
            img = cv2.imread(os.path.join(self.root, img_path))
            label = cv2.imread(os.path.join(self.root, label_path), cv2.IMREAD_GRAYSCALE)

            img = cv2.resize(img, (500, 500))
            label = cv2.resize(label, (500, 500), interpolation=cv2.INTER_NEAREST)

            img = img.transpose(2, 0, 1)  # HWC -> CHW
            images.append(img)
            labels.append(label[np.newaxis, :, :])
            self.index += 1

        top[0].data[...] = np.asarray(images)
        top[1].data[...] = np.asarray(labels)

    def backward(self, top, propagate_down, bottom):
        pass
