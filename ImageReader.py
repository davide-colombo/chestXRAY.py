
import cv2      # OpenCV for images

class ImageReader:

    def import_images_from_pathlist(self, pathlist, color_flag):
        return [cv2.imread(name, flags = color_flag) for name in pathlist]

    def resize_array_of_images(self, array, d):
        return [cv2.resize(img, d) for img in array]

    def scale_array_of_images(self, array, scale_factor):
        return [img / float(scale_factor) for img in array]

    def get_preprocessed_images(self, pathlist, color_flag, d, s):
        array = self.import_images_from_pathlist(pathlist, color_flag)
        array = self.resize_array_of_images(array, d)
        array = self.scale_array_of_images(array, scale_factor = s)
        return array
