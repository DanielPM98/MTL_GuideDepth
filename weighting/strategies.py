import torch
import torch.nn.functional as F


def equivalent_weighting(losses):
    """ Average weighting if T is the number of task each gets 1/T weighting"""
    task_num = losses.dim()
    loss_value = (torch.sum(losses) * task_num)
    return loss_value


def random_weighting(losses):
    """
        Assigning random values following a normal distribution where all add up to 1
        Paper link: https://openreview.net/pdf?id=OdnNBNIdFul
    """
    task_num = losses.dim()
    random_weights = F.softmax(torch.randn(task_num), dim=-1).cuda()
    loss_value = torch.sum(losses*random_weights)

    return loss_value

def choose_strategy(strategy: str):

    if strategy == 'EW':
        return equivalent_weighting
    elif strategy == 'RW':
        return random_weighting
    else:
        print('[ERROR] Weighting Strategy not implemented or incorrect')
        exit(0)