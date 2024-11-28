import torch


class CalculatePerformance:
    def __init__(self, threshold):
        self.threshold = threshold

    def caculate(self, pred, correct):
        pred[pred >= self.threshold] = 1
        pred[pred < self.threshold] = 0
        TP = (pred[correct == 1] == correct[correct == 1]).sum()
        FP = (pred == 1).sum() - TP
        TN = (pred[correct == 0] == correct[correct == 0]).sum()
        FN = (pred == 0).sum() - TN
        # # 计算准确率
        # acc = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
        # # 计算查准率（precision）
        # precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        # # 计算查全率（recall）
        # recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        # # 计算 F1 分数
        # f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        acc = (TP + TN) / (TP + TN + FP + FN)
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1 = 2 * precision * recall / (precision + recall)
        return acc, precision, recall, f1
