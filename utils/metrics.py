import numpy as np
import torch
import torch.nn as nn
from math import ceil
from scipy import ndimage


def confusion_matrix(pred, label, num_classes):
    """
        获得混淆矩阵，其中对角线上表示分类正确的像素点
    :param pred: 转化为一维数组的预测结果，shape: (h, w)
    :param label: 转化为一维数组的标签，shape：(h, w)
    :param num_classes: 类别数
    :return: shape：(num_classes, num_classes), 其中 (i, j) 表示 标签类别为i 预测结果为j的像素点的个数
    """
    mask = (label >= 0) & (label < num_classes)
    conf_mat = np.bincount(num_classes * label[mask].astype(int) + pred[mask], minlength=num_classes ** 2).reshape(
        num_classes, num_classes)
    return conf_mat


def per_class_iou(conf_mat):
    """计算每个类别的iou"""
    intersection = np.diag(conf_mat)
    union = np.sum(conf_mat, 1) + np.sum(conf_mat, 0) - np.diag(conf_mat)
    union_ = union.copy()
    union_[union == 0] = 1
    iou = np.true_divide(intersection, union_)
    iou[union == 0] = 1.0

    return iou


def per_class_recall(conf_mat):
    """
    计算每个类别的召回率: 某类别被预测正确的概率
    recall：tp / (tp + fn)
    """
    TP = np.diag(conf_mat)
    TPFN = conf_mat.sum(1)
    TPFN_ = TPFN.copy()
    TPFN_[TPFN == 0] = 1
    recall = np.true_divide(TP, TPFN_)
    recall[TPFN == 0] = 1.0
    recall[np.isnan(recall)] = 1.0

    return recall


def per_class_precision(conf_mat):
    """
    计算每个类别的精准率: 某类别预测正确的概率
    precision = tp / (tp + fp)
    """
    TP = np.diag(conf_mat)
    TPFP = conf_mat.sum(0)
    TPFP_ = TPFP.copy()
    TPFP_[TPFP == 0] = 1
    precision = np.true_divide(TP, TPFP_)
    precision[TPFP == 0] = 1.0
    precision[np.isnan(precision)] = 1.0

    return precision


def calc_accuracy(conf_mat):
    """
    计算图像中像素准确率
    accuracy: (tp + tn) / (tp + tn + fp + fn)
    """
    return np.sum(np.diag(conf_mat)) / np.maximum(np.sum(conf_mat), 1)


def calc_kappa(conf_mat):
    """
    计算图像中的Kappa系数
    """
    acc = np.sum(np.diag(conf_mat)) / np.maximum(np.sum(conf_mat), 1)
    pe = np.dot(np.sum(conf_mat, 1), np.sum(conf_mat, 0)) / np.maximum(conf_mat.sum() ** 2, 1)
    kappa = (acc - pe) / (1 - pe)

    return kappa


def calc_f_score(conf_mat, f_beta=1):
    """
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f-score = 2 * precision * recall / (precision + recall)
    """
    TP = np.diag(conf_mat)
    FP = conf_mat.sum(0) - TP
    FN = conf_mat.sum(1) - TP
    F_score_up = (1 + f_beta ** 2) * TP
    F_score_down = (1 + f_beta ** 2) * TP + (f_beta ** 2) * FN + FP

    F_score_down_ = F_score_down.copy()
    F_score_down_[F_score_down == 0] = 1
    F_score = np.true_divide(F_score_up, F_score_down_)
    F_score[F_score_down == 0] = 1.0
    F_score[np.isnan(F_score)] = 1.0

    return F_score


def evaluate(conf_mat, f_beta, with_background):
    acc = calc_accuracy(conf_mat)
    kappa = calc_kappa(conf_mat)
    IoU = per_class_iou(conf_mat)
    F_score = calc_f_score(conf_mat, f_beta)
    precision = per_class_precision(conf_mat)
    recall = per_class_recall(conf_mat)

    if with_background == 0:
        real_IoU = IoU[1:]
        real_F_score = F_score[1:]
        real_precision = precision[1:]
        real_recall = recall[1:]
    else:
        real_IoU = IoU
        real_F_score = F_score
        real_precision = precision
        real_recall = recall

    mean_IoU = np.nanmean(real_IoU)
    mean_precision = np.nanmean(real_precision)
    mean_recall = np.nanmean(real_recall)
    mean_f1 = np.nanmean(real_F_score)

    return acc, kappa, mean_IoU, mean_f1, mean_precision, mean_recall, real_IoU, real_F_score, real_precision, real_recall
