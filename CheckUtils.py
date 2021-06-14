from typing import Tuple, List
import numpy as np

class CheckUtils:

    @staticmethod
    def check_classes(class_names: List[str], target_class: str, target_len: int):
        names = [name for name in class_names if name == target_class]
        CheckUtils.check_len(len(names), target_len)

    @staticmethod
    def check_file_consistency(files: List[str], wrong_classes: List[str]):
        for c in wrong_classes:
            intersection = [f for f in files if c in f]
            CheckUtils.check_len(current_len=len(intersection), target_len=0)

    @staticmethod
    def check_len(current_len: int, target_len: int):
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
    def check_float_range(values: List[float], lower_bound: float = 0.0, upper_bound: float = 1.0):
        for v in values:
            if v < lower_bound or v > upper_bound:
                raise Exception('{:.3f} is the current value. '
                                '{:.3f} is the lower bound and {:.3f} is the upper bound'.format(v,
                                                                                                 lower_bound,
                                                                                                 upper_bound))

    @staticmethod
    def check_gt_threshold(values: List[float], threshold: float = 0.0):
        for v in values:
            if v < threshold:
                raise Exception('{} is the current value. {} is the threshold'.format(v, threshold))

    @staticmethod
    def check_unique_partition(partition1: List[str], partition2: List[str]):
        intersection = [name for name in partition1 if name in partition2]
        CheckUtils.check_len(len(intersection), target_len = 0)

    @staticmethod
    def check_partition_consistency(files: List[str],
                                    classes: List[str],
                                    class_names: List[str] = ('bacteria', 'normal', 'virus')):

        for c in class_names:
            f = [file for file in files if c in file]
            c = [cl for cl in classes if cl == c]
            CheckUtils.check_len(len(f), target_len = len(c))



