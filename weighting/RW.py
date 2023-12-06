import torch
import torch.nn.functional as F

from weighting.AbsWeighting import AbsWeighting

class RW(AbsWeighting):
    def __init__(self, task_num):
        super(RW, self).__init__(task_num)

        # self.init_param() TODO: CHECK IF NEEDED... 

    def evaluate(self, losses: torch.Tensor):
        weights = (F.softmax(torch.randn(self.task_num), dim=-1)).cuda()
        loss_value = torch.sum(weights*losses)

        return loss_value