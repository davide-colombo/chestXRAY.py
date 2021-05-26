
###################### IMPORTING LIBRARIES ######################

import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from ModelFactory import ModelFactory

###################### SET RANDOM SEED FOR REPRODUCIBILITY ######################

tf.random.set_seed(1234)

###################### PARAMETERS DEFINITION ######################

data_dir   = "/Users/davidecolombo/Desktop/dataset/chest_xray_keras/"
batch_size = 64
image_size = (224, 224)

###################### TRAINING DATASET ######################

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split = 0.3,
    subset = 'training',
    batch_size = batch_size,
    image_size = image_size,
    shuffle = True,
    seed = 333
)

###################### VALIDATON DATASET ######################

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split = 0.3,
    subset = 'validation',
    batch_size = batch_size,
    image_size = image_size,
    shuffle = True,
    seed = 333
)

###################### CLASS NAMES ######################

class_names = train_ds.class_names


###################### PLOT IMAGES IN THE FIRST BATCH ######################

plt.figure(figsize = (10, 10))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i+1)
        plt.imshow(images[i].numpy().astype('uint8'))
        plt.title(class_names[labels[i]])
        plt.axis('off')

###################### DEFINE MODEL METRICS ######################

metrics = [
    tf.keras.metrics.CategoricalAccuracy(name = 'accuracy'),
    tf.keras.metrics.Precision(),
    tf.keras.metrics.Recall(),
    tf.keras.metrics.AUC(name='auc'),
    tf.keras.metrics.AUC(name='prc', curve='PR')
]

###################### DEFINE CALLBACKS ######################

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_prc',
    verbose=1,
    patience=10,
    mode='max',
    restore_best_weights=True
)

save_best = tf.keras.callbacks.ModelCheckpoint(
    filepath= os.getcwd() + '/checkpoint/',
    monitor= 'val_prc',
    mode='max',
    verbose=1,
    save_best_only= True
)

###################### DEFINE THE VGG16 MODEL ######################

vgg16_model = ModelFactory.make_vgg16(metrics= metrics, img_size= image_size[0], channels=3)
vgg16_model.save_weights(filepath=os.getcwd() + '/vgg16/')

###################### TRAIN THE MODEL ######################

vgg16_history = vgg16_model.fit_generator(
    train_ds,
    validation_data= val_ds,
    epochs= 50,
    steps_per_epoch= 4100 // 64,
    verbose= 1,
    callbacks= [early_stopping, save_best]
)



