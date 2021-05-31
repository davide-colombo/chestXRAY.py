
# @Author: Davide Colombo
# @Date: May 31st, 2021

# @Description: a script that uses Tensorflow and Keras libraries to train, test and validate
#               a CNN based deep learning model for detecting bacterial and viral pneumonia conditions
#               from children chest X-ray images.


###################### IMPORT LIBRARIES ######################

import tensorflow as tf
import matplotlib.pyplot as plt
import os

###################### IMPORT MODULES ######################

from MyCustomMetrics import MyCustomMetrics
from ModelFactory import ModelFactory

###################### SET RANDOM SEED ######################

tf.random.set_seed(1234)

###################### DEFINE VARIABLES ######################

train_dir   = "/Users/davidecolombo/Desktop/dataset/chest_xray_final/train/"
test_dir    = "/Users/davidecolombo/Desktop/dataset/chest_xray_final/test/"
val_dir     = "/Users/davidecolombo/Desktop/dataset/chest_xray_final/val/"

train_batch = 128
test_batch  = 64
epochs      = 50

###################### IMPORT IMAGES ######################

# image data generator with rescaling only
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale = 1./255
)

# training dataset
training = datagen.flow_from_directory(
    directory   = train_dir,
    batch_size  = train_batch,
    color_mode  = 'grayscale',
    class_mode  = 'categorical',
    shuffle     = True,
    seed        = 1234
)

# test set
testing = datagen.flow_from_directory(
    directory   = test_dir,
    batch_size  = test_batch,
    color_mode  = 'grayscale',
    class_mode  = 'categorical',
    shuffle     = True,
    seed        = 1234
)

# validation set
validation = datagen.flow_from_directory(
    directory   = val_dir,
    batch_size  = test_batch,
    color_mode  = 'grayscale',
    class_mode  = 'categorical',
    shuffle     = True,
    seed        = 1234
)

###################### CLASS WEIGHTS ######################

n_bacteria  = 2780
n_virus     = 1493
n_normal    = 1583
total       = n_bacteria + n_normal + n_virus
num_classes = 3
bacteria_weight = total / (num_classes * n_bacteria)
normal_weight   = total / (num_classes * n_normal)
virus_weight    = total / (num_classes * n_virus)

# Dictionary with class indices
class_weight = {
    0: bacteria_weight,
    1: normal_weight,
    2: virus_weight
}

###################### EVALUATION METRICS ######################

my_custom_metric = MyCustomMetrics()

custom_metrics = [
    my_custom_metric.categorical_weighted_accuracy,
    my_custom_metric.categorical_balanced_accuracy,
    my_custom_metric.macro_weighted_f1score,
    my_custom_metric.macro_weighted_precision,
    my_custom_metric.macro_weighted_recall,
    my_custom_metric.categorical_true_positives
]

###################### MAKE MODEL ######################

chestX_model = ModelFactory.make_model(custom_metrics)

###################### TRAIN THE MODEL ######################

chestX_history = chestX_model.fit(
    training,
    validation_data  = validation,
    steps_per_epoch  = 4686 // train_batch,
    validation_steps = 585 // test_batch,
    epochs = epochs,
    class_weight = class_weight,
    # callbacks = [],
    verbose = 1
)

###################### VISUALIZE TRAINING INSIGHTS ######################


###################### TEST THE MODEL ######################



