
###################### IMPORTING LIBRARIES ######################

import tensorflow as tf                 # deep learning
import matplotlib.pyplot as plt         # plot
import os                               # operating system io

from MyCustomMetrics import MyCustomMetrics
from ModelFactory import ModelFactory

# set the seed for reproducibility
tf.random.set_seed(1234)

###################### LIST ALL FILES IN THE DIRECTORY ######################

# data_dir     = "/Users/davidecolombo/Desktop/dataset/chest_xray_keras/"
# class_names  = os.listdir(data_dir)
# class_names  = [c for c in class_names if not c.startswith('.')]     # remove the hidden folders

# create the complete path to the folder
# path2folder = [data_dir + dir for dir in class_names]

# create a list of three that contains the path to each image of the folder
# 0: normal | 1: bacteria | 2: virus
# all_files = [[os.path.join(path, f) for f in os.listdir(path)] for path in path2folder]

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
