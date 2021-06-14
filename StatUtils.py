
import numpy as np
from CheckUtils import CheckUtils

class StatUtils:

    def samplewise_mean(self, images):
        avg = [np.mean(img) for img in images]
        CheckUtils.check_float_range(avg, lower_bound=0.0, upper_bound=1.0)

    def samplewise_var(self, images):
        variance = [np.var(img) for img in images]
        CheckUtils.check_gt_threshold(variance, threshold=0.0)
