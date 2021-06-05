
import random
import pandas as pd
import cv2
import os

from sklearn.model_selection import train_test_split
from DatasetUtils import DatasetUtils

# ROOT DIRECTORY
root_dir_path = '/Users/davidecolombo/Desktop/dataset'
dataset_dir   = '/chest_xray'

def list_files_from_directory(path):
    return [os.path.join(root, file) for root, dirs, files in os.walk(path) for file in files if not file.startswith('.')]

all_files = list_files_from_directory(root_dir_path + dataset_dir)
# len(all_files)              # 5855 because one was removed
# len(set(all_files))         # no duplicate names

# EXTRACT PATIENT ID
patient_id = [name.split('/')[-1].split('_')[0]
              if name.split('/')[-1].startswith('person')
              else 'unknown'
              for name in all_files]

# EXTRACT CLASS NAME
class_name = [name.split('/')[-1] for name in all_files]
class_name = ['bacteria' if 'bacteria' in name else 'virus' if 'virus' in name else 'normal' for name in class_name]

# TRAIN / TEST SPLIT
X_train, X_val, y_train, y_val = train_test_split(all_files, class_name,
                                                  test_size = 0.2,
                                                  stratify = class_name,
                                                  random_state = 1234)
# TRAIN / VAL SPLIT
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train,
                                                    test_size = 0.2,
                                                    stratify = y_train,
                                                    random_state = 1234)

# CHECK ON THE NUMBER OF EXAMPLES
# len(X_train)
# len(X_test)
# len(X_val)

# CHECK ON THE PROPORTION OF EXAMPLES
# len([name for name in y_train if name == 'bacteria'])
# len([name for name in y_train if name == 'normal'])
# len([name for name in y_train if name == 'virus'])
#
# len([name for name in y_test if name == 'bacteria'])
# len([name for name in y_test if name == 'normal'])
# len([name for name in y_test if name == 'virus'])
#
# len([name for name in y_val if name == 'bacteria'])
# len([name for name in y_val if name == 'normal'])
# len([name for name in y_val if name == 'virus'])

# GET TRAIN PATIENT OR UNKNOWN
train_patient = DatasetUtils.get_patient_from_path(X_train)
train_classes = DatasetUtils.get_class_from_path(X_train)
train_dict = {'path': X_train, 'patient': train_patient, 'class': train_classes}
train_df = DatasetUtils.get_dataframe_from_dict(train_dict)


# TRAINING DF GROUPED BY CLASS AND PATIENT
train_gb = train_df.groupby(by = ['class', 'patient'])

# COUNT HOW MUCH IMAGES THERE ARE FOR EACH COMBINATION OF PATIENT AND CLASS
n_by_patient_class = train_gb['path'].count()
n_by_patient_class.head(10)

train_bacteria_in  = []
train_bacteria_out = []

for keys in train_gb.indices:
    if keys[0] == 'bacteria':               # if is the majority class...
        if len(train_gb.indices[keys]) > 2:
            selected = random.sample(list(train_gb.indices[keys]), 2)
            not_selected = [idx for idx in list(train_gb.indices[keys]) if idx not in selected]
            train_bacteria_in.extend(selected)
            train_bacteria_out.extend(not_selected)
        else:
            train_bacteria_in.extend(list(train_gb.indices[keys]))

# now we have obtained 1394 bacteria images in the training set
undersampled_train_bacteria_path = train_df['path'].iloc[train_bacteria_in]
not_selected_train_bacteria_path = train_df['path'].iloc[train_bacteria_out]

# COMPUTE THE NUMBER OF VIRUS IMAGES TO OVERSAMPLE
n_virus_to_oversample = len(undersampled_train_bacteria_path) - train_df.groupby('class')['path'].count().loc['virus']

# COMPUTE THE NUMBER OF NORMAL IMAGES TO OVERSAMPLE
n_normal_to_oversample = len(undersampled_train_bacteria_path) - train_df.groupby('class')['path'].count().loc['normal']


# NOW IT'S TIME TO OVERSAMPLE THE MINORITY CLASSES!!!

train_virus_single_occurrence = []
for keys in train_gb.indices:
    if keys[0] == 'virus':
        if len(train_gb.indices[keys]) == 1:
            train_virus_single_occurrence.extend(list(train_gb.indices[keys]))

# PERSON VIRUS SINGLE OCCURRENCES
train_virus_index_to_oversample = random.sample(train_virus_single_occurrence, n_virus_to_oversample)
train_virus_oversampled_path = list(train_df['path'].iloc[train_virus_index_to_oversample])

# train_df.loc[train_df['class'] == 'virus']            # this produces a DataFrame object

# select all the virus path from the training set
train_virus_total = [path for path in train_df.loc[train_df['class'] == 'virus']['path']]
# len(train_virus_total)
train_virus_total.extend(train_virus_oversampled_path)
# len(train_virus_total)
# print('Number of virus images is equal to number of bacteria images? {}'.format(
#                                                   len(train_virus_total) == len(undersampled_train_bacteria_path)
#                                                   ))


# OVERSAMPLE THE NORMAL IMAGES FROM THE TRAINING SET: FOR THE NORMAL CLASS THERE ARE NO PATIENT!!

train_normal_total = list(train_df.loc[train_df['class'] == 'normal']['path'])
len(train_normal_total)
train_normal_oversample_indices = random.sample(list(train_df.loc[train_df['class'] == 'normal'].index),
                                                n_normal_to_oversample)
train_normal_oversample_path = list(train_df['path'].iloc[train_normal_oversample_indices])

train_normal_total.extend(train_normal_oversample_path)
# len(train_normal_total)
# print('Number of normal images is equal to the number of bacteria images? {}'.format(len(train_normal_total) == len(undersampled_train_bacteria_path)))


# Here, we can see that there are some patient that appear in both virus and bacteria classes
# The goal here is to 'under sampling' the majority class and to do so, we can begin by reducing the number of
# patient with the more than more than 2 images in the majority class

# i = 0
# j = 0
# z = 0
#
# for name in y_train:
#     if name == 'bacteria':
#         i += 1
#     elif name == 'virus':
#         j += 1
#     else:
#         z += 1
#
# print('Number of bacteria images is {}; number of normal images is {}; number of virus images is {}'.format(i, z, j))


def read_image_from_name_list(name_list, flag):
    return [cv2.imread(name, flags=flag) for name in name_list]

# ALL NORMAL IMAGES
normal_images   = read_image_from_name_list(name_list = normal_image_name, flag   = cv2.IMREAD_GRAYSCALE)

# ALL BACTERIA IMAGES
bacteria_images = read_image_from_name_list(name_list = bacteria_image_name, flag = cv2.IMREAD_GRAYSCALE)

# ALL VIRUS IMAGES
virus_images   = read_image_from_name_list(name_list = virus_image_name, flag     = cv2.IMREAD_GRAYSCALE)

def resize_image_from_array(images, dims):
    return [cv2.resize(image, dims) for image in images]

new_dim = (256, 256)

# ALL NORMAL RESIZED
normal_resized = resize_image_from_array(images = normal_images, dims = new_dim)

# ALL BACTERIA RESIZED
bacteria_resized = resize_image_from_array(images = bacteria_images, dims = new_dim)

# ALL VIRUS RESIZED
virus_resized = resize_image_from_array(images = virus_images, dims = new_dim)

def scale_images_from_array(images, factor):
    return [image / float(factor) for image in images]

# ALL NORMAL RESCALED
normal_rescaled = scale_images_from_array(images = normal_resized, factor = 255)

# ALL BACTERIA RESCALED
bacteria_rescaled = scale_images_from_array(images = bacteria_resized, factor = 255)

# ALL VIRUS RESCALE
virus_rescaled = scale_images_from_array(images = virus_resized, factor = 255)


def flatten_array(images):
    return [image.flatten() for image in images]

# ALL NORMAL FLATTEN
normal_flatten = flatten_array(normal_rescaled)

# ALL BACTERIA FLATTEN
bacteria_flatten = flatten_array(bacteria_rescaled)

# ALL VIRUS FLATTEN
virus_flatten = flatten_array(virus_rescaled)

# row and column names
index_values = ['img' + str(number) for number in range(0, len(normal_flatten))]
column_values = ['pixel' + str(number) for number in range(0, len(normal_flatten[0]))]

# NORMAL DATAFRAME
normal_dataframe = pd.DataFrame(normal_flatten, index_values, column_values)
normal_output    = ['normal'] * len(normal_dataframe.index)

# BACTERIA DATAFRAME


# VIRUS DATAFRAME


# PCA ANALYSIS

