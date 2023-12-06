import torch
import torch.nn.functional as F

from weighting.AbsWeighting import AbsWeighting

class STW(AbsWeighting):
    def __init__(self, task_num, main_task):
        super(STW, self).__init__(task_num)

        self.main_task = main_task
        # self.weights = torch.zeros(task_num)
        # self.weights[main_task] = 1.0
        # self.init_param() TODO: CHECK IF NEEDED... 

    def evaluate(self, losses: torch.Tensor):
        assert len(losses) == self.task_num
        loss_value = losses[self.main_task]

        return loss_value