
###################### IMPORTING LIBRARIES ######################

import tensorflow as tf                 # deep learning
import numpy as np                      # multi-dimensional array
import matplotlib.pyplot as plt         # plot
import matplotlib.image as pltimg       # alternative image processing
from PIL import Image                   # image processing
import os                               # operating system io

###################### LIST ALL FILES IN THE DIRECTORY ######################

data_dir = "/Users/davidecolombo/Desktop/dataset/chest_xray_keras/"
class_names  = os.listdir(data_dir)
class_names = [c for c in class_names if not c.startswith('.')]     # remove the hidden folders
class_names

# create the complete path to the folder
path2folder = [data_dir + dir for dir in class_names]

# create a list of three that contains the path to each image of the folder
# 0: normal | 1: bacteria | 2: virus
all_files = [[os.path.join(path, f) for f in os.listdir(path)] for path in path2folder]

###################### READ ALL IMAGES ######################

from my_image_reader import MyImageReader

# Read all images
normal_images   = MyImageReader.import_from_path(path= all_files[0], img_width=256, img_height=256)
bacteria_images = MyImageReader.import_from_path(path= all_files[1], img_width=256, img_height=256)
virus_images    = MyImageReader.import_from_path(path= all_files[2], img_width=256, img_height=256)

###################### PLOT SOME IMAGES ######################

plt.imshow(normal_images[0])
plt.imshow(bacteria_images[0])
plt.imshow(virus_images[0])

###################### IMAGE PROCESSING ######################

# convert image to grayscale

# normalize values between 0 and 1

# reshape array into 4-d array

###################### DEFINE DATA GENERATOR ######################

# Create an image data generator
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale= 1./255,
    validation_split= 0.3
)

# training data
train_generator = datagen.flow_from_directory(
    directory= data_dir,
    target_size= (256, 256),
    color_mode= 'grayscale',
    class_mode= 'categorical',
    batch_size= 64,
    shuffle= True,
    seed = 1234,
    subset= 'training'
)

# validation generator
validation_generator = datagen.flow_from_directory(
    directory= data_dir,
    target_size= (256, 256),
    color_mode= 'grayscale',
    class_mode= 'categorical',
    batch_size= 64,
    shuffle= True,
    seed = 1234,
    subset= 'validation'
)

###################### DEFINE A FUNCTION FOR CREATING KERAS MODEL ######################

def make_model(metrics):
    # define the model
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(filters=64, kernel_size=(7, 7), strides=3, padding="same", activation='relu', input_shape= (256, 256, 1)),
        tf.keras.layers.Conv2D(filters=64, kernel_size=(7, 7), strides=2, padding="same", activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=1, padding="same"),
        tf.keras.layers.Conv2D(filters=64, kernel_size=(5, 5), strides=2, padding="same", activation='relu'),
        tf.keras.layers.Conv2D(filters=64, kernel_size=(5, 5), strides=2, padding="same", activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=1, padding="same"),
        tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=1, padding="same", activation='relu'),
        tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=1, padding="same", activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding="same"),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units = 512, activation='relu'),
        tf.keras.layers.Dense(units= 256, activation='relu'),
        tf.keras.layers.Dense(units=64, activation='relu'),
        tf.keras.layers.Dense(units=3, activation='softmax')
    ])
    # view summary
    model.summary()

    # compile
    model.compile(
        optimizer= tf.keras.optimizers.Adam(learning_rate=3e-4),
        loss     = tf.keras.losses.CategoricalCrossentropy(),
        metrics  = metrics
    )

    return model

###################### DEFINE MODEL METRICS ######################

metrics = [
    tf.keras.metrics.Precision(),
    tf.keras.metrics.Recall(),
    tf.keras.metrics.AUC(name='auc'),
    tf.keras.metrics.AUC(name='prc', curve='PR')
]

###################### DEFINE MODEL PARAMETERS ######################

batch_size = 64
epochs     = 100

###################### DEFINE CALLBACKS ######################

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_prc',
    verbose=1,
    patience=10,
    mode='max',
    restore_best_weights=True
)

save_best = tf.keras.callbacks.ModelCheckpoint(

)

###################### TRAIN THE MODEL ######################

# define the model
model = make_model(metrics)

# train
history = model.fit_generator(
    train_generator,
    epochs = epochs,
    validation_data= validation_generator,
    callbacks= [early_stopping],
    verbose=1
)




