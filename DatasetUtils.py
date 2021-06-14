# @Author: Davide Colombo
# @Date: 2021, 14th June

# @Description: a class for coping with dataset

import pandas as pd
import random
import os


class DatasetUtils:
    LABEL_2_NUM = {'bacteria': 0, 'normal': 1, 'virus': 2}
    NUM_2_LABEL = {0: 'bacteria', 1: 'normal', 2: 'virus'}
    CLASS_COUNT = {'bacteria': 2780, 'normal': 1583, 'virus': 1493}

    def __init__(self, major_class, minor_class):
        self.major_class = major_class
        self.minor_class = minor_class

    def label_to_num(self, labels):
        return [self.LABEL_2_NUM[label] for label in labels]

    def num_to_label(self, nums):
        return [self.NUM_2_LABEL[num] for num in nums]

    def list_files_from_directory(self, dataset_folder_path):
        all_files = [os.path.join(root, file)
                     for root, dirs, files in os.walk(dataset_folder_path)
                     for file in files
                     if not file.startswith('.')]
        self.__check_len(len(all_files), 5856)
        return all_files

    def get_filepath_from_classname(self, files, class_name):
        tmp = [f for f in files if class_name in f]
        self.__check_len(current_len=len(tmp), target_len=self.CLASS_COUNT.get(class_name))
        self.__check_file_consistency(tmp, wrong_classes= [k for k in self.CLASS_COUNT.keys() if k != class_name])
        return tmp

    # major_class: string, the major class name
    # minor_classes: list of strings, the minor classes name
    # path_list: list of strings, the complete path to a file
    def random_oversampling(self, file_path, major_class, minor_classes):
        df = self.__create_dataframe(file_path)
        major_list_path = self.__get_filepath_from_classname(df, major_class)
        class_count = self.__get_class_count(df, minor_classes)
        n_extra = [len(major_list_path) - c for c in class_count]
        minor_list_path = self.__oversample_minor_class(df, minor_classes, n_extra)
        all_path = self.__concatenate_path(major_list_path, minor_list_path)
        all_classes = self.__concatenate_classes(major_class, len(major_list_path),
                                                 minor_classes, [len(minor_list_path[0]), len(minor_list_path[1])])
        return all_path, all_classes

    def get_multiclass_indices(self, y, class_list):
        tmp = []
        for i in range(0, len(class_list)):
            tmp.append(self.__get_class_indices(y, class_list[i]))
        return tmp

    def __get_class_indices(self, y, class_name):
        return [i for i, x in enumerate(y) if x == class_name]

    # =============================================================================

    def __create_dataframe(self, file_path):
        classes = self.get_classname_from_path(file_path)
        d = {'file_path': file_path, 'class': classes}
        return self.dict_to_dataframe(d)

    def dict_to_dataframe(self, dictionary):
        return pd.DataFrame(
            dictionary,
            columns=list(dictionary.keys())
        )

    def get_classname_from_path(self, file_path, path_separator):
        all_names = [name.split(path_separator)[-1] for name in file_path]
        class_names = ['bacteria'
                       if 'bacteria' in name
                       else 'virus'
                       if 'virus' in name
                       else 'normal'
                       for name in all_names]
        self.__check_classes(class_names, 'bacteria', 2780)
        self.__check_classes(class_names, 'virus', 1493)
        self.__check_classes(class_names, 'normal', 1583)
        return class_names

    # =============================================================================

    def __get_filepath_from_classname(self, df, class_name):
        return df.loc[df['class'] == class_name]['file_path']

    def __get_class_count(self, df, minor_classes):
        gb = df.groupby('class')['file_path'].count()
        return [gb.loc[c] for c in minor_classes]

    def __oversample_minor_class(self, df, minor_classes, n_extra):
        path = []
        for i in range(0, len(minor_classes)):
            tmp = self.__get_filepath_from_classname(df, minor_classes[i])
            rnd = random.choices(list(tmp.index), k=n_extra[i])  # sampling with replacement
            # extra + tot
            extra = list(df['file_path'].iloc[rnd])
            tmp = list(tmp)
            tmp.extend(extra)
            path.append(tmp)
        return path

    # concatenate all major classes and all minor classes on after another
    def __concatenate_path(self, major_path, minor_path):
        tmp = list(major_path).copy()
        for i in range(0, len(minor_path)):
            tmp.extend(minor_path[i])
        return tmp

    def __concatenate_classes(self, major_classes, n_major, minor_classes, n_minor):
        tmp = [major_classes] * n_major
        for i in range(0, len(minor_classes)):
            label = [minor_classes[i]] * n_minor[i]
            tmp.extend(label)
        return tmp

    def shuffle_indices(self, indices):
        return random.sample(indices, len(indices))

# =============================================================================

#           CHECK METHODS

# =============================================================================

    def __check_len(self, current_len, target_len):
        if current_len != target_len:
            raise Exception('{} is the current length. {} is the target length'.format(current_len, target_len))

    def __check_classes(self, class_names, target_class, target_len):
        names = [name for name in class_names if name == target_class]
        self.__check_len(len(names), target_len)

    def __check_file_consistency(self, files, wrong_classes):
        for c in wrong_classes:
            intersection = [f for f in files if c in f]
            self.__check_len(current_len=len(intersection), target_len=0)

