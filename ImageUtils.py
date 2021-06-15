
import cv2
import numpy as np
from CheckUtils import CheckUtils
from typing import List, Tuple

class ImageUtils:

    GRAYSCALE = cv2.IMREAD_GRAYSCALE
    RGB       = cv2.IMREAD_COLOR

    def import_images(self, filepath,
                      color_flag = GRAYSCALE,
                      scaling = True, scale_factor = 255,
                      resizing = False, new_dim = (256, 256),
                      reshaping = False, new_shape = (256, 256, 1),
                      list2nparray = False):

        images = self.__read_images(filepath, color_flag)
        CheckUtils.check_len(len(images), len(filepath))
        if resizing:
            images = self.__resize_images(images, new_dim)
        if scaling:
            images = self.__scale_images(images, scale_factor = scale_factor)
            CheckUtils.check_nparray_range(images, lower_bound=0.0, upper_bound=1.0)
        if reshaping:
            images = self.__reshape_images(images, new_shape)
            CheckUtils.check_shape(images, target_shape=new_shape)
        if list2nparray:
            images = self.list2nparray(images)
        return images

    def export_images(self, images: np.ndarray,
                      save_dir: str,
                      save_prefix: str,
                      save_format: str = '.jpeg') -> None:
        for i, image in enumerate(images):
            filepath = save_dir + save_prefix + str(i) + save_format
            cv2.imwrite(filepath, img = image)

# =============================================================================
    def __read_images(self, files: List[str], color_flag: int) -> List[np.ndarray]:
        return [cv2.imread(name, flags = color_flag) for name in files]

    def __resize_images(self, images: List[np.ndarray], new_dim: Tuple[int, int]) -> List[np.ndarray]:
        return [cv2.resize(img, new_dim) for img in images]

    def __scale_images(self, images: List[np.ndarray], scale_factor: int) -> List[np.ndarray]:
        return [np.divide(img, float(scale_factor)) for img in images]

    def __reshape_images(self, images: List[np.ndarray], new_shape: Tuple[int, int, int]) -> List[np.ndarray]:
        return [np.reshape(img, new_shape) for img in images]

    def list2nparray(self, images: List[np.ndarray]) -> np.ndarray:
        return np.array(images)
