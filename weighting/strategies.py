import torch
import torch.nn.functional as F

from weighting import EW, UW, RW, STW


def equivalent_weighting(losses: list):
    """ 
        Average weighting if T is the number of task each gets 1/T weighting
        Input:
            losses <list>: list of Tensors with each task loss

        Return:
            loss_value <Tensor>: computed loss value over all task losses
    """
    weight = 1 / len(losses)
    loss_value = sum(weight * loss for loss in losses)

    return loss_value


def random_weighting(losses: list):
    """
        Assigning random values following a normal distribution where all add up to 1
        Paper link: https://openreview.net/pdf?id=OdnNBNIdFul
        Input:
            losses <list>: list of Tensors with each task loss

        Return:
            loss_value <Tensor>: computed loss value over all task losses
    """
    task_num = len(losses)
    random_weights = F.softmax(torch.randn(task_num), dim=-1)

    loss_value = sum(losses[i] * random_weights[i] for i in range(len(losses)))

    return loss_value


def uncertainty_weighting(losses: list):
    """
        Uncertainty weighting strategy TODO: better explain it
        Paper link: https://arxiv.org/pdf/1705.07115.pdf
        Input:
            losses <list>: list of Tensors with each task loss

        Return:
            loss_value <Tensor>: computed loss value over all task losses
    """

    return 


def choose_strategy(strategy: str, task_num: int, mode: int):

    if strategy == 'EW':
        return EW.EW(task_num)
    elif strategy == 'RW':
        return RW.RW(task_num)
    elif strategy == 'UW':
        return UW.UW(task_num)
    elif strategy == 'STW':
        return STW.STW(task_num, mode)
    else:
        print('[ERROR] Weighting Strategy not implemented or incorrect')
        exit(0)