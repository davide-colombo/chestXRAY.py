
from PIL import Image
import numpy as np

class MyImageReader:

    @staticmethod
    def import_from_path(path, img_width, img_height):
        images = np.ndarray(shape= (len(path), img_height, img_width))
        for num, name in enumerate(path):
            img = Image.open(name)
            if not img.size[0] == img_height:
                raise AssertionError("Image height is not equal to the expected value {}".format(img_height))
            if not img.size[1] == img_width:
                raise AssertionError("Image width is not equal to the expected value {}".format(img_width))
            # if it is all ok then add to the array
            images[num, :, :] = img
        return images


