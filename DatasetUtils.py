
import pandas as pd
import random

class DatasetUtils:

    @staticmethod
    def get_patient_from_path(path_list):
        return [name.split('/')[-1].split('_')[0]
                if name.split('/')[-1].startswith('person')
                else 'unknown'
                for name in path_list]

    @staticmethod
    def get_dataframe_from_dict(dictionary):
        return pd.DataFrame(
            dictionary,
            columns = list(dictionary.keys())
        )

    # gb is a 'GroupBy' object in which the first key must be the 'class' value
    @staticmethod
    def undersample_class_gb_patient(gb, class_name):
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
    @staticmethod
    def oversample_class_gb_patient(gb, class_name):
        single_values = []
        for keys in gb.indices:
            if keys[0] == class_name:
                idx = list(gb.indices[keys])
                if len(idx) == 1:
                    single_values.extend(idx)
        return single_values
