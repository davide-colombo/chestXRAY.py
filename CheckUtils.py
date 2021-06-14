from typing import Tuple
import numpy as np

class CheckUtils:

    @staticmethod
    def check_classes(class_names, target_class, target_len):
        names = [name for name in class_names if name == target_class]
        CheckUtils.check_len(len(names), target_len)

    @staticmethod
    def check_file_consistency(files, wrong_classes):
        for c in wrong_classes:
            intersection = [f for f in files if c in f]
            CheckUtils.check_len(current_len=len(intersection), target_len=0)

    @staticmethod
    def check_len(current_len, target_len):
        if current_len != target_len:
            raise Exception('{} is the current length. {} is the target length'.format(current_len, target_len))

    @staticmethod
    def check_shape(images, target_shape: Tuple[int, int, int] = (256, 256, 1)):
        for img in images:
            if img.shape != target_shape:
                raise Exception('{} is the current shape. {} is the target shape'.format(img.shape, target_shape))

    @staticmethod
    def check_nparray_range(images, lower_bound: float = 0.0, upper_bound: float = 1.0):
        for img in images:
            if np.amin(img) < lower_bound or np.amax(img) > upper_bound:
                raise Exception('{:.3f} is the current min value. {:.3f} is the lower bound;\n'
                                '{:.3f} is the current max value. {:.3f} is the upper bound'.format(np.amin(img), lower_bound,
                                                                                                    np.amax(img), upper_bound))

    @staticmethod
    def check_float_range(values, lower_bound: float = 0.0, upper_bound: float = 1.0):
        for v in values:
            if v < lower_bound or v > upper_bound:
                raise Exception('{:.3f} is the current value. '
                                '{:.3f} is the lower bound and {:.3f} is the upper bound'.format(v,
                                                                                                 lower_bound,
                                                                                                 upper_bound))

    @staticmethod
    def check_gt_threshold(values, threshold: float = 0.0):
        for v in values:
            if v < threshold:
                raise Exception('{} is the current value. {} is the threshold'.format(v, threshold))
