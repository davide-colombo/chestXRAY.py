
import numpy as np

class StatUtils:

    def samplewise_mean(self, images):
        mean = []
        for image in images:
            mean.append(np.mean(image))
        return mean

    def samplewise_var(self, images):
        var = []
        for image in images:
            var.append(np.var(image))
        return var
