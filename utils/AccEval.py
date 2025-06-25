import numpy as np


class AccEval():

    def __init__(self, label_count, f_beta, with_background):
        self.label_count = label_count
        self.f_beta = f_beta
        self.with_background = with_background
        self.accumulator = {}
        self.accumulator['TP'] = np.zeros(label_count, dtype=np.float64)
        self.accumulator['FN'] = np.zeros(label_count, dtype=np.float64)
        self.accumulator['FP'] = np.zeros(label_count, dtype=np.float64)
        self.accumulator['num_pixels'] = 0.0

    def evaluate(self, gt, pred):
        TP = np.zeros(self.label_count)
        # TN = np.zeros(self.label_count)
        FP = np.zeros(self.label_count)
        FN = np.zeros(self.label_count)

        gt_img = gt
        pred_img = pred

        for i in range(self.label_count):
            gt_copy = gt_img.copy()
            pred_copy = pred_img.copy()

            gt_copy[gt_img != i] = 0
            gt_copy[gt_img == i] = 1
            pred_copy[pred_img != i] = 0
            pred_copy[pred_img == i] = 2
            gt_pred = gt_copy + pred_copy

            TP[i] = np.sum(gt_pred == 3)
            FN[i] = np.sum(gt_pred == 1)
            FP[i] = np.sum(gt_pred == 2)
            # TN[i] = np.sum(gt_pred == 0)

        elem_count = gt_img.shape[0] * gt_img.shape[1] * gt_img.shape[2]

        self.accumulator['TP'] += TP
        self.accumulator['FN'] += FN
        self.accumulator['FP'] += FP
        self.accumulator['num_pixels'] += elem_count

    def totally_evaluate(self):
        tps = 0.0
        total = 0.0
        tpfps = np.zeros(self.label_count)
        tpfns = np.zeros(self.label_count)
        for i in range(self.label_count):
            tps += self.accumulator['TP'][i]
            tpfps[i] = (self.accumulator['TP'][i] + self.accumulator['FP'][i])
            tpfns[i] = (self.accumulator['TP'][i] + self.accumulator['FN'][i])
            total += tpfps[i] * tpfns[i]

        pixel_acc = tps / self.accumulator['num_pixels']  # 即为OA，tps为混淆矩阵对角线元素和

        pe = total / (self.accumulator['num_pixels'] ** 2)
        kappa = (pixel_acc - pe) / (1 - pe)

        intersection = self.accumulator['TP']
        union = self.accumulator['TP'] + self.accumulator['FN'] + self.accumulator['FP']  # 并集
        union_ = union.copy()
        union_[union == 0] = 1
        IoU = np.true_divide(intersection, union_)
        IoU[union == 0] = 1.0
        mIoU = self.calc_average(IoU, self.label_count, self.with_background)

        F_score = self.f_measure(self.accumulator['TP'], self.accumulator['FP'], self.accumulator['FN'], self.f_beta)
        F_score[np.isnan(F_score)] = 1.0
        mF_score = self.calc_average(F_score, self.label_count, self.with_background)

        precision, recall = self.compute_precision_recall(self.accumulator['TP'], self.accumulator['FP'],
                                                          self.accumulator['FN'])
        precision[np.isnan(precision)] = 1.0
        recall[np.isnan(recall)] = 1.0
        mPrecision = self.calc_average(precision, self.label_count, self.with_background)
        mRecall = self.calc_average(recall, self.label_count, self.with_background)

        if self.with_background == 0:
            real_IoU = IoU[1:]
            real_F_score = F_score[1:]
            real_precision = precision[1:]
            real_recall = recall[1:]
        else:
            real_IoU = IoU
            real_F_score = F_score
            real_precision = precision
            real_recall = recall

        return dict(
            pixel_acc=pixel_acc,
            kappa=kappa,
            IoU=real_IoU,
            mIoU=mIoU,
            F_score=real_F_score,
            mF_score=mF_score,
            precision=real_precision,
            mPrecision=mPrecision,
            recall=real_recall,
            mRecall=mRecall
        )

    def f_measure(self, TP, FP, FN, f_beta):
        F_score_up = (1 + f_beta ** 2) * TP
        F_score_down = (1 + f_beta ** 2) * TP + (f_beta ** 2) * FN + FP

        F_score_down_ = F_score_down.copy()
        F_score_down_[F_score_down == 0] = 1
        F_score = np.true_divide(F_score_up, F_score_down_)
        F_score[F_score_down == 0] = 1.0

        return F_score

    def compute_precision_recall(self, TP, FP, FN):
        TPFP = TP + FP
        TPFP_ = TPFP.copy()
        TPFP_[TPFP == 0] = 1
        precision = np.true_divide(TP, TPFP_)
        precision[TPFP == 0] = 1.0

        TPFN = TP + FN
        TPFN_ = TPFN.copy()
        TPFN_[TPFN == 0] = 1
        recall = np.true_divide(TP, TPFN_)
        recall[TPFN == 0] = 1.0

        return precision, recall

    def calc_average(self, values, label_count, with_background):
        mean_value = 0.0
        index_start = 0
        _label_count = label_count
        if with_background == 0:
            index_start = 1
            _label_count -= 1
        for i in range(index_start, label_count):
            mean_value += values[i]
        mean_value /= _label_count

        return mean_value
