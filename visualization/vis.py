import os

import cv2
import numpy as np
import torch

import config
from src.model import SC_Net


# for id,label in enumerate(instance_labels):
#           num_instances = np.max(label[...,0])
#           if num_instances>180:
#                     print(id)

#并列显示GT和Pre
def vis(img,gt_inst,inst_map,filename,save_path = config.experiment,docut = False,crop_params=""):
    #1.pre
    # 获取实例索引
    indices = np.unique(inst_map)
    # 获取实例数量
    num_instances = len(indices) #含背景

    # 为每个实例生成随机颜色
    colors = [(np.random.randint(0,128), np.random.randint(0,128), np.random.randint(0,128)) for _ in range(num_instances)]

    # 在图像上绘制每个实例的边界
    result = img.copy()

    for ind in range(num_instances):
          i = int(indices[ind])
          if i != 0:
              mask = (inst_map == i)
              contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
              for cnt in contours:
                        cv2.drawContours(result, [cnt], 0, colors[ind-1], 2)
    #2.GT
    # 获取实例索引
    indices = np.unique(gt_inst)
    # 获取实例数量
    num_instances = len(indices)  # 含背景

    # 为每个实例生成随机颜色
    colors = [(np.random.randint(0, 128), np.random.randint(0, 128), np.random.randint(0, 128)) for _ in
              range(num_instances)]

    # 在图像上绘制每个实例的边界
    gt_result = img.copy()
    for ind in range(num_instances):
        i = int(indices[ind])
        if i != 0:
            mask = (gt_inst == i)
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                cv2.drawContours(gt_result, [cnt], 0, colors[ind - 1], 2)

    result = cv2.hconcat([result, gt_result])#右是GT
    if docut == True:
        top,left,bottom,right = crop_params
        result = result[top:bottom, left:right, ...]
    dir = os.path.join(save_path,'{}_sample.png'.format(filename))
    cv2.imwrite(dir,result)


def distance_transform(inst_mask):
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

if __name__ == "__main__":
    model = SC_Net(device=config.device)
    weight_dir = "../{}/{}".format(config.checkpoints_dir,
                                  config.inference_para_weights)
    print("test_dir:", config.test_dir)
    print("weight_dir:", weight_dir)
    model.load_state_dict(torch.load(weight_dir))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    filename = "test_5_8.npy"
    images_path  = os.path.join('../',config.test_dir, filename)

    sample = np.load(images_path)
    img = (sample[:, :, :3] * 255.0).astype(np.uint8)

    inst_map = sample[:, :, 3]
    nuclear_mask = sample[:, :, 4]
    # type_mask = sample[:, :, 6]
    scale_mask = log_with_condition(sample[:, :, 7])

    centroid_prob_mask = distance_transform(sample[:, :, 3])  # 每个像素点到边界的欧式距离

    if np.max(centroid_prob_mask) == 0:
        pass
    else:
        centroid_prob_mask = (centroid_prob_mask /
                              np.max(centroid_prob_mask)) * 1.0  # 每个像素到边界距离的归一化

    mask = np.zeros((nuclear_mask.shape[0], nuclear_mask.shape[0], 4))
    mask[:, :, 0] = nuclear_mask  # bin_mask
    mask[:, :, 1] = centroid_prob_mask  # 每个像素到边界距离的归一化
    mask[:, :, 2] = inst_map  # 实例索引
    mask[:, :, 3] = scale_mask  # 尺度map


    from verify import do_seg
    input = np.transpose(sample[:, :, :3], (2, 0, 1)).astype(np.float32)
    do_seg(input,mask,filename,model,is_save_experiment=False,docut=True,crop_params=(47,0+256,47+105,105+256))
    # do_seg(input, mask, filename, model, is_save_experiment=False, docut=False)

    #vis(img,gt_map,gt_map,'test_1_0','')
####################################
#以下无用
####################################
#制作data 随机剪切
# def random_crop(img, target_shape=(256,256),seed=None):
#
#           # seed = np.random.randint(10000)
#           # X_train[index] = random_crop(img, target_shape=(IMG_HEIGHT,IMG_WIDTH), seed=seed)
#           #y_train[index] = random_crop(img_type_map, target_shape=(IMG_HEIGHT,IMG_WIDTH), seed=seed)
#           n_rows = img.shape[0]
#           n_cols = img.shape[1]
#
#           row_size, col_size = target_shape
#
#           start_row = np.random.RandomState(seed).choice(range(n_rows-row_size))
#           start_col = np.random.RandomState(seed).choice(range(n_cols-col_size))
#
#           return img[start_row:start_row+row_size,start_col:start_col+col_size]






