import torch

from weighting.AbsWeighting import AbsWeighting

class EW(AbsWeighting):
    def __init__(self, task_num):
        super(EW, self).__init__(task_num)

        # self.init_param() TODO: CHECK IF NEEDED... 

    def init_param(self): # TODO: include how to set parameters
        self.weight = torch.tensor([1 / self.task_num] * self.task_num).cuda()

    def evaluate(self, losses: torch.Tensor):
        loss_value = (self.weight * losses).sum()

        return loss_value
