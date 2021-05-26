
###################### IMPORTING LIBRARIES ######################
import tensorflow
import tensorflow as tf                 # deep learning
import numpy as np                      # multi-dimensional array
import matplotlib.pyplot as plt         # plot
import matplotlib.image as pltimg       # alternative image processing
from PIL import Image                   # image processing
import os                               # operating system io

# set the seed for reproducibility
tf.random.set_seed(1234)

###################### LIST ALL FILES IN THE DIRECTORY ######################

data_dir = "/Users/davidecolombo/Desktop/dataset/chest_xray_keras/"
class_names  = os.listdir(data_dir)
class_names = [c for c in class_names if not c.startswith('.')]     # remove the hidden folders

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

# Create a data generator for VGG16 architecture
vgg16_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function= tf.keras.applications.vgg16.preprocess_input,
    validation_split= 0.3
)

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

###################### DEFINE TRAINING SET FOR VGG16 TRAIN ######################

# define vgg16 network image size
vgg16_imgsize = 224

vgg16_train = vgg16_datagen.flow_from_directory(
    directory= data_dir,
    target_size= (vgg16_imgsize, vgg16_imgsize),
    class_mode= 'categorical',
    batch_size= 64,
    shuffle= True,
    seed= 1234,
    subset= 'training'
)

vgg16_validation = vgg16_datagen.flow_from_directory(
    directory= data_dir,
    target_size= (vgg16_imgsize, vgg16_imgsize),
    class_mode= 'categorical',
    batch_size= 64,
    shuffle= True,
    seed= 1234,
    subset= 'validation'
)

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

###################### DEFINE THE VGG MODEL ######################

def make_vgg16(metrics, img_size, channels):
    conv_base = tf.keras.applications.VGG16(
        include_top=False,
        weights='imagenet',
        pooling='None',
        input_shape= (img_size, img_size, channels)
    )

    # freeze the layers
    for layer in conv_base.layers:
        layer.trainable = False

    # define the model
    top_model = conv_base.output
    top_model = tf.keras.layers.Flatten()(top_model)
    top_model = tf.keras.layers.Dense(units = 128, activation='relu')(top_model)
    # top_model = tf.keras.layers.Dropout(rate=0.5)(top_model)
    # top_model = tf.keras.layers.Dense(units = 128, activation='relu')(top_model)
    # top_model = tf.keras.layers.Dropout(rate=0.5)(top_model)
    # top_model = tf.keras.layers.Dense(units = 64, activation='relu')(top_model)
    top_model = tf.keras.layers.Dropout(rate=0.5)(top_model)
    output_layer = tf.keras.layers.Dense(units=3, activation='softmax')(top_model)

    # final model
    model = tf.keras.Model(inputs = conv_base.input, outputs = output_layer)

    # summary
    model.summary()

    # compile the model
    model.compile(
        optimizer = tf.keras.optimizers.Adam(),
        loss      = tf.keras.losses.CategoricalCrossentropy(),
        metrics   = metrics
    )

    return model

###################### USE VGG16 FOR FEATURE EXTRACTION ######################

# must have 3 channels
vgg16_model = make_vgg16(metrics = metrics, img_size=vgg16_imgsize, channels=3)

vgg16_history = vgg16_model.fit_generator(
    vgg16_train,
    validation_data= vgg16_validation,
    epochs= 50,
    steps_per_epoch= 4101 // 64,
    validation_steps= 1755 // 64,
    verbose= 1,
    callbacks= [early_stopping, save_best]
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

###################### DEFINE MODEL PARAMETERS ######################

batch_size = 64
epochs     = 50

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



