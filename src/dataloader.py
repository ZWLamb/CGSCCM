import os

from torch.utils.data import Dataset as BaseDataset
import glob
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

import cv2


class Dataset(BaseDataset):

    def __init__(self,
                 dataset_dir,
                 augmentation=None,
                 preprocessing=None,
                 mode=None):
        ## Dataloader for Regression and Classification

        self.dataset_dir = dataset_dir
        self.samples_paths = glob.glob("{}*.npy".format(self.dataset_dir))

        if mode == 'test':
            self.samples_paths = sorted(self.samples_paths)

        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.mode = mode

    def __getitem__(self, i):

        sample = np.load(self.samples_paths[i])

        image = (sample[:, :, :3] * 255.0).astype(np.uint8)
        inst_map = sample[:, :, 3]
        nuclear_mask = sample[:, :, 4]
        #type_mask = sample[:, :, 6]
        scale_mask = log_with_condition(sample[:, :, 7])

        centroid_prob_mask = self.distance_transform(sample[:, :, 3])#每个像素点到边界的欧式距离

        if np.max(centroid_prob_mask) == 0:
            pass
        else:
            centroid_prob_mask = (centroid_prob_mask /
                                  np.max(centroid_prob_mask)) * 1.0  #每个像素到边界距离的归一化

        mask = np.zeros((nuclear_mask.shape[0], nuclear_mask.shape[0], 4))
        mask[:, :, 0] = nuclear_mask #bin_mask
        mask[:, :, 1] = centroid_prob_mask  #每个像素到边界距离的归一化
        mask[:, :, 2] = inst_map #实例索引
        mask[:, :, 3] = scale_mask # 尺度map
        if self.augmentation:
            try:
                sample = self.augmentation(image=image, mask=mask)
                image, mask = sample["image"], sample["mask"]
            except:
                pass

        # apply preprocessing
        if self.preprocessing:
            image = image / 255.0
            sample = self.preprocessing(image=image) #transpose 255*255*3 --> 3*255*255
            image = sample["image"]

        file_name = os.path.basename(self.samples_paths[i])
        return image, mask, file_name

    def __len__(self):
        return len(self.samples_paths)

    def distance_transform(self, inst_mask):
        heatmap = np.zeros_like(inst_mask).astype("uint8")
        for x in np.unique(inst_mask)[1:]:#去掉背景 所以是1：
            temp = inst_mask + 0
            temp = np.where(temp == x, 1, 0).astype("uint8")

            heatmap = heatmap + cv2.distanceTransform(temp, cv2.DIST_L2, 3)

        return heatmap #获得每个像素到边界距离


def log_with_condition(matrix):
    result = np.zeros_like(matrix, dtype=float)
    zero_indices = matrix == 0
    nonzero_indices = matrix != 0
    result[nonzero_indices] = np.log(matrix[nonzero_indices])
    return result