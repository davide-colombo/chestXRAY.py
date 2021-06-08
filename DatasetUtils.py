
import pandas as pd
import random

class DatasetUtils:

    LABEL_2_NUM = {'bacteria': 0, 'normal': 1, 'virus': 2}
    NUM_2_LABEL = {0: 'bacteria', 1: 'normal', 2: 'virus'}

    def __init__(self, path_separator, major_class, minor_class):
        self.path_separator = path_separator
        self.major_class = major_class
        self.minor_class = minor_class

    def label_to_num(self, labels):
        return [self.LABEL_2_NUM[label] for label in labels]

    def num_to_label(self, nums):
        return [self.NUM_2_LABEL[num] for num in nums]

    # major_classes: list of strings
    # minor_classes: list of strings
    # path_list: list of path to file
    def random_oversampling(self, path_list, major_class, minor_classes):
        df = self.__create_dataframe(path_list)
        major_list_path = self.__get_filepath_from_classname(df, major_class)
        class_count = self.__get_class_count(df, minor_classes)
        n_extra = [len(major_list_path) - c for c in class_count]
        minor_list_path = self.__oversample_minor_class(df, minor_classes, n_extra)
        all_path    = self.__concatenate_path(major_list_path, minor_list_path)
        all_classes = self.__concatenate_classes(major_class, len(major_list_path),
                                                 minor_classes, [len(minor_list_path[0]), len(minor_list_path[1])])
        return all_path, all_classes

# =============================================================================

    def __create_dataframe(self, file_path):
        classes = self.__get_class_from_path(file_path)
        d = {'file_path': file_path, 'class': classes}
        return self.__dict_to_dataframe(d)

    def __dict_to_dataframe(self, dictionary):
        return pd.DataFrame(
            dictionary,
            columns = list(dictionary.keys())
        )

    def __get_class_from_path(self, file_path):
        all_names = [name.split(self.path_separator)[-1] for name in file_path]
        return ['bacteria'
                if 'bacteria' in name
                else 'virus'
                if 'virus' in name
                else 'normal'
                for name in all_names]

# =============================================================================

    def __get_filepath_from_classname(self, df, class_name):
        return df.loc[df['class'] == class_name]['file_path']

    def __get_class_count(self, df, minor_classes):
        count = []
        for c in minor_classes:
            count.append(df.groupby('class')['file_path'].count().loc[c])
        return count

    def __oversample_minor_class(self, df, minor_classes, n_extra):
        path = []
        for i in range(0, len(minor_classes)):
            tmp = self.__get_filepath_from_classname(df, minor_classes[i])
            rnd = random.choices(list(tmp.index), k = n_extra[i])        # sampling with replacement
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
