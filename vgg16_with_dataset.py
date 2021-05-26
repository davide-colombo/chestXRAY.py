
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

###################### VALIDATION DATASET ######################

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split = 0.3,
    subset = 'validation',
    batch_size = batch_size,
    image_size = image_size,
    shuffle = True,
    seed = 333
)

###################### INSPECTING THE TRAINING DATASET ######################

# we have a tuple (image, label)
print(train_ds.element_spec)
print(train_ds)

for image, label in train_ds.take(1):
    for i in range(1):
        print(image[i])

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

###################### INSPECT TRAIN BATCHES ######################

for image_batch, labels_batch in train_ds:
    print(image_batch.shape)
    print(labels_batch.shape)
    break

###################### STANDARDIZE THE DATA ######################

normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)

normalized_train = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_train))
first_image = image_batch[0]
print(np.min(first_image), np.max(first_image))

normalized_val = val_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_val))
first_image = image_batch[0]
print(np.min(first_image), np.max(first_image))

###################### CONVERT THE LABELS INTO CATEGORICAL ######################

# USE 'tf.one_hot' and NOT 'tf.keras.utils.to_categorical'

categorical_train = normalized_train.map(lambda x, y: (x, tf.one_hot(y, len(class_names))))
image_batch, label_batch = next(iter(categorical_train))
print(label_batch.shape)

categorical_val = normalized_val.map(lambda x, y: (x, tf.one_hot(y, len(class_names))))
image_batch, label_batch = next(iter(categorical_val))
print(label_batch.shape)

###################### CONFIGURE THE DATASET FOR PERFORMANCE ######################

AUTOTUNE = tf.data.AUTOTUNE

normalized_train = normalized_train.cache().prefetch(buffer_size = AUTOTUNE)
normalized_val   = normalized_val.cache().prefetch(buffer_size = AUTOTUNE)

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



