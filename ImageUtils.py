
import cv2              # OpenCV for images
import numpy as np      # array reshape

class ImageUtils:

    GRAYSCALE = cv2.IMREAD_GRAYSCALE
    RGB       = cv2.IMREAD_COLOR

    def import_class_images(self, filepath,
                            color_flag = GRAYSCALE,
                            scaling = True, scale_factor = 255,
                            resizing = False, new_dim = (256, 256),
                            reshaping = False, new_shape = (256, 256, 1),
                            list2nparray = False):

        images = self.__read_images(filepath, color_flag)
        if resizing:
            images = self.__resize_images(images, new_dim)
        if scaling:
            images = self.__scale_images(images, scale_factor = scale_factor)
        if reshaping:
            images = self.__reshape_images(images, new_shape)
        if list2nparray:
            images = self.__list2nparray(images)
        return images

    def export_images(self, images, save_dir, save_prefix):
        i = 0
        for image in images:
            filepath = save_dir + save_prefix + str(i) + '.jpeg'
            cv2.imwrite(filepath, img=image)
            i += 1

# =============================================================================
    def __read_images(self, file_path, color_flag):
        return [cv2.imread(name, flags = color_flag) for name in file_path]

    def __resize_images(self, images, new_dim):
        return [cv2.resize(img, new_dim) for img in images]

    def __scale_images(self, images, scale_factor):
        return [np.divide(img, float(scale_factor)) for img in images]

    def __reshape_images(self, images, new_shape):
        return [np.reshape(img, new_shape) for img in images]

    def __list2nparray(self, images):
        return np.array(images)

# =============================================================================

#           CHECK METHODS

# =============================================================================


