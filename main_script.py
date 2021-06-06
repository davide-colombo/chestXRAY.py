
###################### IMPORTING LIBRARIES ######################

import tensorflow as tf                 # deep learning
import matplotlib.pyplot as plt         # plot
import os                               # operating system io
from sklearn.model_selection import train_test_split

from DatasetUtils import DatasetUtils
from ImageUtils import ImageUtils
from MyCustomMetrics import MyCustomMetrics
from ModelFactory import ModelFactory

# set the seed for reproducibility
tf.random.set_seed(1234)

###################### LIST ALL FILES IN THE DIRECTORY ######################

root_dir_path = '/Users/davidecolombo/Desktop/dataset'
dataset_dir   = '/chest_xray'

def list_files_from_directory(path):
    return [os.path.join(root, file) for root, dirs, files in os.walk(path) for file in files if not file.startswith('.')]

all_files = list_files_from_directory(root_dir_path + dataset_dir)
# len(all_files)              # 5855 because one was removed
# len(set(all_files))         # no duplicate names

###################### EXTRACT CLASS NAMES ######################

class_name = [name.split('/')[-1] for name in all_files]
class_name = ['bacteria' if 'bacteria' in name else 'virus' if 'virus' in name else 'normal' for name in class_name]

# len([name for name in class_name if name == 'normal'])

###################### TRAIN AND VALIDATION SPLIT ######################

X_train, X_val, y_train, y_val = train_test_split(all_files, class_name,
                                                  test_size = 0.2,
                                                  stratify = class_name,
                                                  random_state = 1234)

# len([name for name in y_train if name == 'normal'])
# len([name for name in y_val if name == 'virus'])

###################### TRAIN AND TEST SPLIT ######################

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train,
                                                    test_size = 0.2,
                                                    stratify = y_train,
                                                    random_state = 1234)

# len([name for name in y_train if name == 'bacteria'])
# len([name for name in y_test if name == 'bacteria'])

# len([path for path in X_train for test in X_test if path == test])
# len([path for path in X_train for val in X_val if path == val])
# len([path for path in X_test for val in X_val if path == val])

###################### CREATE A BALANCED VERSION OF THE TRAINING SET ######################

major_classes = 'bacteria'
minor_classes = ['normal', 'virus']

my_data_utils = DatasetUtils()

# TRAINING SET
train_path, train_classes = my_data_utils.balance_dataset(X_train, major_classes, minor_classes)

# len(train_path)
# len(train_classes)
# len([name for name in train_classes if name == 'bacteria'])

###################### READ ALL IMAGES ######################

my_img_utils = ImageUtils()
train_images = my_img_utils.import_images_from_pathlist(train_path, color_flag = ImageUtils.GRAYSCALE)
train_images = my_img_utils.resize_array_of_images(train_images, d = (256, 256))
train_images = my_img_utils.scale_array_of_images(train_images, scale_factor = 255)
train_images = my_img_utils.reshape_array_of_images(train_images, (256, 256, 1))

###################### SHUFFLE TRAINING IMAGES ######################

rnd = my_data_utils.shuffle_indices(list(range(0, len(train_images))))
train_images_shuffle  = [train_images[i] for i in rnd]
train_classes_shuffle = [train_classes[i] for i in rnd]

###################### READ VALIDATION AND TEST IMAGES ######################

validation_images = my_img_utils.get_preprocessed_images(X_val, ImageUtils.GRAYSCALE,
                                                         d = (256, 256), s = 255, shape = (256, 256, 1))

test_images = my_img_utils.get_preprocessed_images(X_test, ImageUtils.GRAYSCALE,
                                                   d = (256, 256), s = 255, shape = (256, 256, 1))

###################### RESHAPE IMAGE ARRAY ######################

train_images       = my_img_utils.convert_list_to_nparray(train_images_shuffle)
validation_images  = my_img_utils.convert_list_to_nparray(validation_images)
test_images        = my_img_utils.convert_list_to_nparray(test_images)

###################### CONVERT CLASS LABELS INTO NUMERIC ######################

train_classes = my_data_utils.label_to_num(train_classes_shuffle)
validation_classes = my_data_utils.label_to_num(y_val)
test_classes = my_data_utils.label_to_num(y_test)

###################### ONE-HOT ENCODE LABELS ######################

train_one_hot = tf.keras.utils.to_categorical(train_classes, num_classes = 3)
validation_one_hot = tf.keras.utils.to_categorical(validation_classes, num_classes = 3)
test_one_hot = tf.keras.utils.to_categorical(test_classes, num_classes = 3)

###################### PLOT SOME IMAGES ######################

plt.rcParams["figure.figsize"] = (10, 10)

def plot_images(images, labels):
    plt.subplots_adjust(hspace=0.8)
    for n, img in enumerate(images):
        plt.subplot(3, 3, n+1)
        plt.imshow(img, cmap = 'gray')
        plt.title(labels[n])

plot_images(train_images_shuffle[:9], train_classes_shuffle[:9])

###################### DEFINE HYPER PARAMETERS ######################

train_batch   = 32
test_batch    = 256
epochs        = 50

###################### DEFINE CUSTOM METRICS ######################

my_custom_metric = MyCustomMetrics()

custom_metrics = [
    my_custom_metric.balanced_accuracy,
    my_custom_metric.macro_f1score,
    my_custom_metric.macro_precision,
    my_custom_metric.macro_recall,
    my_custom_metric.macro_bacteria_precision,             # bacteria precision
    my_custom_metric.macro_bacteria_recall,                # bacteria recall
    my_custom_metric.macro_normal_precision,               # normal precision
    my_custom_metric.macro_normal_recall,                  # normal recall
    my_custom_metric.macro_virus_precision,                # virus precision
    my_custom_metric.macro_virus_recall                    # virus recall
]

###################### DEFINE CALLBACKS ######################

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor  = 'val_macro_f1score',
    verbose  = 1,
    patience = 10,
    mode     ='max',
    restore_best_weights = True
)

model_ckpt = tf.keras.callbacks.ModelCheckpoint(
    filepath = os.getcwd() + '/checkpoint/',
    monitor  = 'val_macro_f1score',
    mode     = 'max',
    verbose  = 1,
    save_best_only = True
)

###################### DEFINE THE MODEL ######################

my_model = ModelFactory.make_model(custom_metrics)

###################### TRAIN THE MODEL ######################

history = my_model.fit(
    train_images_shuffle,
    train_one_hot,
    validation_data  = (validation_images, validation_one_hot),
    epochs           = epochs,
    batch_size       = train_batch,
    validation_batch_size = test_batch,
    verbose          = 1,
    callbacks        = [early_stopping, model_ckpt]
)
