
###################### IMPORTING LIBRARIES ######################

import tensorflow as tf                 # deep learning
import matplotlib.pyplot as plt         # plot
import os                               # operating system io
from sklearn.model_selection import train_test_split

from DatasetUtils import DatasetUtils
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

###################### CREATE BALANCE TRAINING SET ######################

major_classes = 'bacteria'
minor_classes = ['normal', 'virus']

my_utils = DatasetUtils()

# training set
train_path, train_classes = my_utils.balance_dataset(X_train, major_classes, minor_classes)

# len(train_path)
# len(train_classes)
# len([name for name in train_classes if name == 'virus'])

###################### READ ALL IMAGES ######################

# from my_image_reader import MyImageReader

# Read all images
# normal_images   = MyImageReader.import_from_path(path= all_files[0], img_width=256, img_height=256)
# bacteria_images = MyImageReader.import_from_path(path= all_files[1], img_width=256, img_height=256)
# virus_images    = MyImageReader.import_from_path(path= all_files[2], img_width=256, img_height=256)

###################### PLOT SOME IMAGES ######################

# plt.imshow(normal_images[0])
# plt.imshow(bacteria_images[0])
# plt.imshow(virus_images[0])

###################### DEFINE DATA GENERATOR ######################

# Create a data generator for VGG16 architecture
vgg16_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function= tf.keras.applications.vgg16.preprocess_input,
    validation_split= 0.3
)

###################### DEFINE HYPER PARAMETERS ######################

# define vgg16 parameters
data_dir      = "/Users/davidecolombo/Desktop/dataset/chest_xray_keras/"
vgg16_imgsize = (224, 224)
batch_size    = 128
epochs        = 50

###################### DEFINE TRAINING SET ######################

training = vgg16_datagen.flow_from_directory(
    directory   = data_dir,
    target_size = vgg16_imgsize,
    class_mode  = 'categorical',
    batch_size  = batch_size,
    shuffle     = True,
    seed        = 1234,
    subset      = 'training'
)

###################### DEFINE VALIDATION SET ######################

validation = vgg16_datagen.flow_from_directory(
    directory   = data_dir,
    target_size = vgg16_imgsize,
    class_mode  = 'categorical',
    batch_size  = batch_size,
    shuffle     = True,
    seed        = 1234,
    subset      = 'validation'
)

###################### DEFINE CUSTOM METRICS ######################

my_custom_metric = MyCustomMetrics()

custom_metrics = [
    tf.keras.metrics.CategoricalAccuracy(name = 'acc'),
    my_custom_metric.categorical_f1_score,
    my_custom_metric.categorical_precision,
    my_custom_metric.categorical_recall,
    my_custom_metric.categorical_true_positives
]

###################### DEFINE CALLBACKS ######################

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor  = 'val_categorical_f1_score',
    verbose  = 1,
    patience = 10,
    mode     ='max',
    restore_best_weights = True
)

model_ckpt = tf.keras.callbacks.ModelCheckpoint(
    filepath = os.getcwd() + '/checkpoint/',
    monitor  = 'val_categorical_f1_score',
    mode     = 'max',
    verbose  = 1,
    save_best_only = True
)

###################### DEFINE VGG16 MODEL ######################

# must have 3 channels
vgg16_model = ModelFactory.make_vgg16(metrics = custom_metrics, img_size = vgg16_imgsize, channels = 3)

###################### TRAIN VGG16 MODEL ######################

vgg16_history = vgg16_model.fit_generator(
    training,
    validation_data  = validation,
    epochs           = epochs,
    batch_size       = batch_size,
    steps_per_epoch  = 4101 // batch_size,
    validation_steps = 1755 // batch_size,
    verbose   = 1,
    callbacks = [early_stopping, model_ckpt]
)
