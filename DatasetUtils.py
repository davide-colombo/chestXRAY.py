
import pandas as pd
import random

class DatasetUtils:

    LABEL_2_NUM = {'bacteria': 0, 'normal': 1, 'virus': 2}
    NUM_2_LABEL = {0: 'bacteria', 1: 'normal', 2: 'virus'}

    def label_to_num(self, labels):
        return [self.LABEL_2_NUM[label] for label in labels]

    def num_to_label(self, nums):
        return [self.NUM_2_LABEL[num] for num in nums]

    def __get_patient_from_path(self, path_list):
        return [name.split('/')[-1].split('_')[0]
                if name.split('/')[-1].startswith('person')
                else 'unknown'
                for name in path_list]

    def __get_class_from_path(self, path_list):
        all_names = [name.split('/')[-1] for name in path_list]
        return ['bacteria'
                if 'bacteria' in name
                else 'virus'
                if 'virus' in name
                else 'normal'
                for name in all_names]

    def __get_dataframe_from_dict(self, dictionary):
        return pd.DataFrame(
            dictionary,
            columns = list(dictionary.keys())
        )

    # gb is a 'GroupBy' object in which the first key must be the 'class' value
    def __undersample_class_gb_patient(self, gb, class_name):
        samples_in  = []
        samples_out = []
        for keys in gb.indices:
            if keys[0] == class_name:
                idx = list(gb.indices[keys])
                if len(idx) > 2:
                    selected = random.sample(idx, 2)
                    not_selected = [i for i in idx if i not in selected]
                    samples_in.extend(selected)
                    samples_out.extend(not_selected)
                else:
                    samples_in.extend(idx)
        return samples_in, samples_out

    # gb is a 'GroupBy' object in which the first key must be the 'class' value
    def __oversample_class_gb_patient(self, gb, class_name):
        single_values = []
        for keys in gb.indices:
            if keys[0] == class_name:
                idx = list(gb.indices[keys])
                if len(idx) == 1:
                    single_values.extend(idx)
        return single_values

    def __create_dataframe(self, path_list):
        patient = self.__get_patient_from_path(path_list)
        classes = self.__get_class_from_path(path_list)
        d = {'path': path_list, 'patient': patient, 'class': classes}
        return self.__get_dataframe_from_dict(d)

    # major_classes: list of strings
    # minor_classes: list of strings
    # path_list: list of path to file
    def balance_dataset(self, path_list, major_classes, minor_classes):
        df = self.__create_dataframe(path_list)
        gb = df.groupby(['class', 'patient'])
        major_in, major_out = self.__undersample_class_gb_patient(gb, class_name = major_classes)
        final_major = list(df['path'].iloc[major_in])
        class_count = self.__get_n_class_examples(df, minor_classes)
        n_extra = [len(final_major) - c for c in class_count]
        minor_list_path = self.__minor_class_oversampling(df, minor_classes, n_extra)
        all_path = self.__contat_path(final_major, minor_list_path)
        all_classes = self.__concat_classes('bacteria', len(final_major), minor_classes, [len(minor_list_path[0]), len(minor_list_path[1])])
        return all_path, all_classes

    def __get_n_class_examples(self, df, minor_classes):
        count = []
        for c in minor_classes:
            count.append(df.groupby('class')['path'].count().loc[c])
        return count

    def __minor_class_oversampling(self, df, minor_classes, n_extra):
        path = []
        for i in range(0, len(minor_classes)):
            tmp = df.loc[df['class'] == minor_classes[i]]['path']
            if minor_classes[i] == 'normal':
                rnd = random.sample(list(tmp.index), n_extra[i])
            else:
                gb = df.groupby(['class', 'patient'])
                single_patient = self.__oversample_class_gb_patient(gb, minor_classes[i])
                rnd = random.sample(single_patient, n_extra[i])
            # extra + tot
            extra = list(df['path'].iloc[rnd])
            tmp = list(tmp)
            tmp.extend(extra)
            path.append(tmp)

        return path

    # concatenate all major classes and all minor classes on after another
    def __contat_path(self, major_path, minor_path):
        tmp = major_path.copy()
        for i in range(0, len(minor_path)):
            tmp.extend(minor_path[i])
        return tmp

    def __concat_classes(self, major_classes, n_major, minor_classes, n_minor):
        tmp = [major_classes] * n_major
        for i in range(0, len(minor_classes)):
            label = [minor_classes[i]] * n_minor[i]
            tmp.extend(label)
        return tmp

    def shuffle_indices(self, indices):
        return random.sample(indices, len(indices))
