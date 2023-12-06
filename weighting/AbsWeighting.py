

class AbsWeighting:
    def __init__(self, task_num):
        self.task_num = task_num

        self.init_param()

    def init_param(self):
        """ Implement initialization for weighting parameters"""
        pass

    @property
    def evaluate(self, losses, **kwargs):
        """ Implement weighting evaluation method"""
        pass