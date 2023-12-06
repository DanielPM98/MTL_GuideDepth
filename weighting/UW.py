import torch

from weighting.AbsWeighting import AbsWeighting

class UW(AbsWeighting):
    def __init__(self, task_num):
        super(UW, self).__init__(task_num)

        # self.init_param() TODO: CHECK IF NEEDED... 

    def init_param(self): # TODO: include how to set parameters
        # self.scale = torch.nn.Parameter(torch.tensor([-0.5] * self.task_num))
        self.log_sigma = torch.tensor([0.0] * self.task_num, requires_grad=True)

    def evaluate(self, losses: torch.Tensor):
        # loss_value = (losses / (2*self.scale.exp()) + self.scale/2).sum()
        loss_value = sum([0.5 * (torch.exp(-log) * loss + log) for loss, log in zip(losses, self.log_sigma)])

        return loss_value