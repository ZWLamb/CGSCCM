import torch
from torch.nn import functional as F


def dice_loss(pred, true, smooth=1e-3):

    true = (F.one_hot(true.to(torch.int64), num_classes=2)).type(torch.float32) #B 256 256 2
    pred = F.softmax(pred.permute(0, 2, 3, 1).contiguous(), dim=-1) #B 256 256 2

    inse = torch.sum(pred * true, (0, 1, 2))
    l = torch.sum(pred, (0, 1, 2))
    r = torch.sum(true, (0, 1, 2))
    loss = 1.0 - (2.0 * inse + smooth) / (l + r + smooth)
    loss = torch.sum(loss)
    return loss


def mse_loss(pred, true):

    loss = pred - true
    loss = (loss * loss).mean()
    return loss


def combined_loss(pred, true):
    #仅做尺度预测
    if pred.size(1) == 1:
        #质量预测损失  TODO:质量概率场 中间给予大损失，边缘给予小损失。这样让中心尽快学习到目标尺度，而让边缘在迭代中，逐渐学习到尺度。
        mse2 = mse_loss(pred[:, 0, :, :], true[:, :, :, 3])
        return mse2
    else:
        mse = mse_loss(pred[:, 2, :, :], true[:, :, :, 1])  # 质心概率图
        dice = dice_loss(pred[:, :2, :, :], true[:, :, :, 0])  # pred [32, 2, 256, 256]   true的二值图 32*256*256
        if pred.size(1)==3:
            return 1 * (mse) + 2 * (dice)  # + 1 * mse2
        else:#pred.size(1) == 4
            mse2 = mse_loss(pred[:, 3, :, :],true[:, :, :, 3])
            return 1 * (mse) + 2 * (dice) + 2 * mse2

"""
        mask[:, :, 0] = nuclear_mask #bin_mask
        mask[:, :, 1] = centroid_prob_mask  #每个像素到边界距离的归一化
        mask[:, :, 2] = inst_map #实例索引
        mask[:, :, 3] = scale_mask  # 尺度map
"""
