# @Author: Davide Colombo
# @Date: 2021, 14th June

# @Description: a class for coping with dataset
from typing import List

from sklearn.model_selection import train_test_split
from CheckUtils import CheckUtils
import pandas as pd
import random
import os

class DatasetUtils:
    LABEL_2_NUM = {'bacteria': 0, 'normal': 1, 'virus': 2}
    NUM_2_LABEL = {0: 'bacteria', 1: 'normal', 2: 'virus'}
    CLASS_COUNT = {'bacteria': 2780, 'normal': 1583, 'virus': 1493}

    def label_to_num(self, labels: List[str] = ('bacteria', 'normal', 'virus')):
        return [self.LABEL_2_NUM[label] for label in labels]

    def num_to_label(self, nums: List[int]):
        return [self.NUM_2_LABEL[num] for num in nums]

    def list_files_from_directory(self, dataset_folder_path: str):
        all_files = [os.path.join(root, file)
                     for root, dirs, files in os.walk(dataset_folder_path)
                     for file in files
                     if not file.startswith('.')]
        CheckUtils.check_len(len(all_files), 5856)
        return all_files

    def random_oversampling(self, files: List[str],
                            major_class: str = 'bacteria',
                            minor_classes: List[str] = ('normal', 'virus')):

        major_class_files = self.class2filepath(files, class_name=major_class)
        minor_classes_files = []
        for c in minor_classes:
            minor_classes_files.append(self.class2filepath(files, class_name = c))

        n_extra = [len(major_class_files) - len(minor_class_files) for minor_class_files in minor_classes_files]
        minor_classes_files = self.__oversample_minor_class(minor_classes_files, n_extra)

        train_files_oversampled   = self.__concatenate_path(major_class_files, minor_classes_files)
        train_classes_oversampled = self.__concatenate_classes(major_class, len(major_class_files),
                                                               minor_classes,
                                                               [len(minor_classes_files[0]), len(minor_classes_files[1])])
        CheckUtils.check_partition_consistency(train_files_oversampled, train_classes_oversampled)
        return train_files_oversampled, train_classes_oversampled

# =============================================================================

    def training_validation_split(self, files: List[str], class_name: List[str], val_size: float = 0.2):
        CheckUtils.check_len(len(files), len(class_name))
        train_files, val_files, train_classes, val_classes = train_test_split(files, class_name,
                                                                              test_size = val_size,
                                                                              stratify  = class_name,
                                                                              random_state = 1234)
        CheckUtils.check_unique_partition(train_files, val_files)
        CheckUtils.check_partition_consistency(train_files, train_classes)
        CheckUtils.check_partition_consistency(val_files, val_classes)
        return train_files, train_classes, val_files, val_classes

    def training_test_split(self, files: List[str], class_name: List[str], test_size: float = 0.2):
        CheckUtils.check_len(len(files), len(class_name))
        train_files, test_files, train_classes, test_classes = train_test_split(files, class_name,
                                                                                test_size = test_size,
                                                                                stratify  = class_name,
                                                                                random_state=1234)
        CheckUtils.check_unique_partition(train_files, test_files)
        CheckUtils.check_partition_consistency(train_files, train_classes)
        CheckUtils.check_partition_consistency(test_files, test_classes)
        return train_files, train_classes, test_files, test_classes

# =============================================================================

    def get_multiclass_indices(self, y, class_list):
        tmp = []
        for i in range(0, len(class_list)):
            tmp.append(self.__get_class_indices(y, class_list[i]))
        return tmp

    def __get_class_indices(self, y, class_name):
        return [i for i, x in enumerate(y) if x == class_name]

# =============================================================================

    def filepath2class(self, files: List[str], path_separator: str = '/'):
        all_names = [name.split(path_separator)[-1] for name in files]
        class_names = ['bacteria'
                       if 'bacteria' in name
                       else 'virus'
                       if 'virus' in name
                       else 'normal'
                       for name in all_names]
        CheckUtils.check_classes(class_names, 'bacteria', 2780)
        CheckUtils.check_classes(class_names, 'virus', 1493)
        CheckUtils.check_classes(class_names, 'normal', 1583)
        return class_names

    def class2filepath(self, files: List[str], class_name: str):
        tmp = [f for f in files if class_name in f]
        # CheckUtils.check_len(current_len=len(tmp), target_len=self.CLASS_COUNT.get(class_name))
        CheckUtils.check_file_consistency(tmp, wrong_classes= [k for k in self.CLASS_COUNT.keys() if k != class_name])
        return tmp

# =============================================================================

    def __oversample_minor_class(self, minor_classes_files: List[List[str]], n_extra: List[int]):
        tmp = []
        for i, files in enumerate(minor_classes_files):
            selected_files = random.choices(files, k = n_extra[i])                 # sampling with replacement
            files.extend(selected_files)
            tmp.append(files)
        return tmp

    # concatenate all major classes and all minor classes on after another
    def __concatenate_path(self, major_class_files: List[str], minor_classes_files: List[List[str]]):
        tmp = major_class_files.copy()
        for minor_files in minor_classes_files:
            tmp.extend(minor_files)
        CheckUtils.check_len(len(tmp), len(major_class_files) * 3)
        return tmp

    def __concatenate_classes(self, major_class_name: str, n_major: int,
                              minor_classes: List[str], n_minor: List[int]):
        tmp = [major_class_name] * n_major
        for i in range(0, len(minor_classes)):
            tmp.extend([minor_classes[i]] * n_minor[i])
        return tmp

# =============================================================================
    def shuffle_indices(self, indices):
        return random.sample(indices, len(indices))

