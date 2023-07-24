"""
Code from FastDepth 
    Diana Wofk et al, FastDepth: Fast Monocular Depth
    Estimation on Embedded Devices, International Conference on Robotics and 
    Automation (ICRA), 2019
    https://github.com/dwofk/fast-depth
"""
import torch
import numpy as np
from torchmetrics.classification import MulticlassJaccardIndex, MulticlassAccuracy

import math


def log10(x):
      """Convert a new tensor with the base-10 logarithm of the elements of x. """
      return torch.log(x) / math.log(10)

class Result(object):
    def __init__(self):
        # Depth metrics
        self.irmse, self.imae = 0, 0
        self.mse, self.rmse, self.mae = 0, 0, 0
        self.absrel, self.lg10 = 0, 0
        self.delta1, self.delta2, self.delta3 = 0, 0, 0
        self.data_time, self.gpu_time = 0, 0
        self.rmse_log = 0

        # Segmentation metrics
        self.mIoU = 0
        self.IoU = 0
        self.mMAE = 0 # Mean accuracy
        self.class_px_acc = 0
        self.px_acc = 0

    def set_to_worst(self):
        self.irmse, self.imae = np.inf, np.inf
        self.mse, self.rmse, self.mae = np.inf, np.inf, np.inf
        self.rmse_log = np.inf
        self.absrel, self.lg10 = np.inf, np.inf
        self.delta1, self.delta2, self.delta3 = 0, 0, 0
        self.data_time, self.gpu_time = 0, 0

        self.mIoU = 0
        self.IoU = 0
        self.mMAE = 0
        self.class_px_acc = 0
        self.px_acc = 0

    def update(self, irmse, imae, mse, rmse, rmse_log, mae, absrel, lg10, delta1, delta2, delta3, mIoU, IoU, mMAE, class_px_acc, px_acc, gpu_time, data_time):
        self.irmse, self.imae = irmse, imae
        self.mse, self.rmse, self.mae = mse, rmse, mae
        self.rmse_log = rmse_log
        self.absrel, self.lg10 = absrel, lg10
        self.delta1, self.delta2, self.delta3 = delta1, delta2, delta3

        self.mIoU = mIoU
        self.IoU = IoU
        self.mMAE = mMAE
        self.class_px_acc = class_px_acc
        self.px_acc = px_acc

        self.data_time, self.gpu_time = data_time, gpu_time

    def evaluate_depth(self, output, target):
        abs_diff = (output - target).abs()

        self.mse = float((torch.pow(abs_diff, 2)).mean())
        self.rmse = math.sqrt(self.mse)
        self.mae = float(abs_diff.mean())
        self.lg10 = float((log10(output) - log10(target)).abs().mean())
        self.rmse_log = math.sqrt(torch.pow(log10(output) - log10(target), 2).mean())
        self.absrel = float((abs_diff / target).mean())

        maxRatio = torch.max(output / target, target / output)
        self.delta1 = float((maxRatio < 1.25).float().mean())
        self.delta2 = float((maxRatio < 1.25 ** 2).float().mean())
        self.delta3 = float((maxRatio < 1.25 ** 3).float().mean())
        self.data_time = 0
        self.gpu_time = 0

        inv_output = 1 / output
        inv_target = 1 / target
        abs_inv_diff = (inv_output - inv_target).abs()
        self.irmse = math.sqrt((torch.pow(abs_inv_diff, 2)).mean())
        self.imae = float(abs_inv_diff.mean())
    
    def evaluate_segmentation(self, output, target):
        """ 
            Description: 
                Update the evaluation metrics for segmentation task. Said metrics are: mean IoU and class IoU; pixel accuracy
                and mean accuracy (mMAE). All results are average per batch.

            Inputs:
                output <Tensor>: segmentation prediction with probabilities. Shape [batch_size, num_classes, w, h]
                target <Tensor>: segmentation target tensor. Each value goes from 0 to num_classes. Shape [batch_size, w, h]
        """
        
        num_classes = output.shape[1]
        pred = torch.argmax(output, dim=1)

        # Calculate IoU metrics
        # print('Prediction shape: ', pred.shape)
        # print('Target shape: ', target.shape)
        # print('Unique values')
        # print(torch.unique(pred))
        # print(torch.unique(target))
        self.IoU = iou_metric(output, target) # Output list of lenght num_classes
        self.mIoU = self.IoU.mean()

        # Calculate pixel accuracy over a batch
        correct = (pred == target).sum().item()
        total = len(target.flatten())
        self.px_acc = correct / total

        # Calculate mean accuracy
        # Calculate pixel accuracy per class and divide by number of classes
        accuracy = MulticlassAccuracy(num_classes=num_classes, average=None)
        self.class_px_acc = accuracy(pred, target) # Pixel accuracy over each class
        self.mMAE = self.class_px_acc.mean() # Mean pixel accuracy over all classes


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.count = 0.0

        self.sum_irmse, self.sum_imae = 0, 0
        self.sum_mse, self.sum_rmse, self.sum_mae = 0, 0, 0
        self.sum_rmse_log = 0
        self.sum_absrel, self.sum_lg10 = 0, 0
        self.sum_delta1, self.sum_delta2, self.sum_delta3 = 0, 0, 0
        self.sum_data_time, self.sum_gpu_time = 0, 0

        self.sum_mIoU = 0
        self.sum_IoU = 0
        self.sum_mMAE = 0 # Mean accuracy
        self.sum_class_px_acc = 0
        self.sum_px_acc = 0

    def update(self, result, gpu_time, data_time, n=1):
        self.count += n

        self.sum_irmse += n*result.irmse
        self.sum_imae += n*result.imae
        self.sum_mse += n*result.mse
        self.sum_rmse += n*result.rmse
        self.sum_rmse_log += n*result.rmse_log
        self.sum_mae += n*result.mae
        self.sum_absrel += n*result.absrel
        self.sum_lg10 += n*result.lg10
        self.sum_delta1 += n*result.delta1
        self.sum_delta2 += n*result.delta2
        self.sum_delta3 += n*result.delta3

        self.sum_mIoU += n*result.mIoU
        self.sum_IoU += n*result.IoU
        self.sum_mMAE += n*result.mMAE
        self.sum_class_px_acc += n*result.class_px_acc
        self.sum_px_acc += n*result.px_acc

        self.sum_data_time += n*data_time
        self.sum_gpu_time += n*gpu_time

    def average(self):
        avg = Result()
        avg.update(
            self.sum_irmse / self.count, 
            self.sum_imae / self.count,
            self.sum_mse / self.count, 
            self.sum_rmse / self.count, 
            self.sum_mae / self.count,
            self.sum_rmse_log / self.count, 
            self.sum_absrel / self.count, 
            self.sum_lg10 / self.count,
            self.sum_delta1 / self.count, 
            self.sum_delta2 / self.count, 
            self.sum_delta3 / self.count,
            self.sum_mIoU / self.count, 
            self.sum_IoU / self.count, 
            self.sum_mMAE / self.count, 
            self.sum_class_px_acc / self.count, 
            self.sum_px_acc / self.count,
            self.sum_gpu_time / self.count, 
            self.sum_data_time / self.count)
        return avg


def iou_metric(pred: torch.Tensor, target: torch.Tensor, average: str='mean') -> torch.Tensor:
    """
        Description: 
            Calculate the IoU metric over the prediction tensor and the target for each category

        Input:
            pred <Tensor>: prediction tensor. Shape [batch_size, num_classes, w, h]
            target <Tensor>: target tensor. Shape [batch_size, w, h]

        Output:
            ious <Tensor>: IoU metrics for each class
    """

    num_classes = pred.size(1)
    ious = []

    pred = torch.argmax(pred, dim=1) # Calculate the most probable classe over all

    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)

    # Calculate the intersection and union over each class
    for c in range(num_classes+1):
        pred_idx = pred == c
        target_idx = target == c

        intersection = (pred_idx[target_idx]).long().sum().cpu()
        union = pred_idx.long().sum().cpu() + target_idx.long().sum().cpu() - intersection

        ious.append(intersection / float(max(union,1)))

    ious = torch.Tensor(ious)
    if average == 'mean':
        return ious.mean()

    return ious