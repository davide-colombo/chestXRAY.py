
import tensorflow as tf
import matplotlib.pyplot as plt
import os

from MyCustomMetrics import MyCustomMetrics
from ModelFactory import ModelFactory

tf.random.set_seed(1234)

train_dir    = "/Users/davidecolombo/Desktop/dataset/chest_xray_final/train/"
test_dir    = "/Users/davidecolombo/Desktop/dataset/chest_xray_final/test/"
val_dir     = "/Users/davidecolombo/Desktop/dataset/chest_xray_final/val/"
train_batch = 128
test_batch  = 16
epochs      = 50

datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale = 1./255
)

training = datagen.flow_from_directory(
    directory   = train_dir,
    batch_size  = train_batch,
    color_mode  = "grayscale",
    class_mode  = 'categorical',
    shuffle     = True,
    seed        = 1234
)

testing = datagen.flow_from_directory(
    directory   = test_dir,
    batch_size  = test_batch,
    color_mode  = "grayscale",
    class_mode  = 'categorical',
    shuffle     = True,
    seed        = 1234
)

validation = datagen.flow_from_directory(
    directory   = val_dir,
    batch_size  = test_batch,
    color_mode  = "grayscale",
    class_mode  = 'categorical',
    shuffle     = True,
    seed        = 1234
)

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

my_custom_metric = MyCustomMetrics()

custom_metrics = [
    my_custom_metric.categorical_weighted_accuracy,
    my_custom_metric.categorical_balanced_accuracy,
    my_custom_metric.categorical_f1_score,
    my_custom_metric.categorical_precision,
    my_custom_metric.categorical_recall,
    my_custom_metric.categorical_true_positives
]

chestX_model = ModelFactory.make_model(custom_metrics)

chestX_history = chestX_model.fit(
    training,
    validation_data = validation,
    steps_per_epoch = 4686 // train_batch,
    validation_steps = 585 // test_batch,
    epochs = epochs,
    class_weight = class_weight,
    verbose = 1
)









