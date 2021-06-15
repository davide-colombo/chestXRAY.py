
###################### IMPORTING LIBRARIES ######################

import tensorflow as tf
import matplotlib.pyplot as plt

from DatasetUtils import DatasetUtils
from ImageUtils import ImageUtils
from StatUtils import StatUtils
from PlotUtils import PlotUtils
from MyCustomMetrics import MyCustomMetrics
from ModelFactory import ModelFactory

# set the seed for reproducibility
tf.random.set_seed(1234)

###################### LIST ALL FILES IN THE DATASET DIRECTORY ######################

my_data_utils = DatasetUtils()

path2root = '/Users/davidecolombo/Desktop/dataset/chest_xray_final'
all_files = my_data_utils.list_files_from_directory(path2root)
class_name = my_data_utils.filepath2class(all_files)

###################### GET CLASS FILE PATH ######################

bacteria_files = my_data_utils.class2filepath(all_files, 'bacteria')
normal_files   = my_data_utils.class2filepath(all_files, 'normal')
virus_files    = my_data_utils.class2filepath(all_files, 'virus')

###################### IMPORT CLASS IMAGES ######################

my_img_utils = ImageUtils()
bacteria_images = my_img_utils.import_images(bacteria_files)
normal_images   = my_img_utils.import_images(normal_files)
virus_images    = my_img_utils.import_images(virus_files)

###################### COMPUTE MEAN AND VARIANCE FOR EACH IMAGE ######################

my_stat_utils = StatUtils()
bacteria_mean, bacteria_var, bacteria_inv_var = my_stat_utils.get_all_stats(bacteria_images)
normal_mean, normal_var, normal_inv_var       = my_stat_utils.get_all_stats(normal_images)
virus_mean, virus_var, virus_inv_var          = my_stat_utils.get_all_stats(virus_images)

###################### PLOT HISTOGRAM AND DENSITY DISTRIBUTION ######################

my_plot_utils = PlotUtils()

kwargs = dict(hist_kws = {'alpha': .6}, kde_kws = {'linewidth': 2})
my_plot_utils.plot_class_histogram(class_stats = [bacteria_mean, normal_mean, virus_mean],
                                   xlim  = (0, 1),
                                   title = 'Histogram and Density distribution of mean values',
                                   ylabel = 'Density',
                                   xlabel = 'Mean',
                                   **kwargs)

my_plot_utils.plot_class_histogram(class_stats = [bacteria_var, normal_var, virus_var],
                                   xlim  = (0, 0.1),
                                   title = 'Histogram and Density distribution of variance values',
                                   ylabel = 'Density',
                                   xlabel = 'Variance',
                                   **kwargs)

my_plot_utils.plot_class_histogram(class_stats = [bacteria_inv_var, normal_inv_var, virus_inv_var],
                                   xlim  = (0, 250),
                                   title = 'Histogram and Density distribution of inverse of the variance values',
                                   ylabel = 'Density',
                                   xlabel = 'Inverse of the Variance',
                                   **kwargs)

###################### TRAIN AND VALIDATION SPLIT ######################

train_files, train_classes, val_files, val_classes = my_data_utils.training_validation_split(all_files, class_name,
                                                                                             val_size=0.2)
train_files, train_classes, test_files, test_classes = my_data_utils.training_test_split(train_files, train_classes,
                                                                                         test_size=0.2)

###################### CREATE A BALANCED VERSION OF THE TRAINING SET ######################

train_files, train_classes = my_data_utils.random_oversampling(train_files)

###################### READ ALL IMAGES ######################

my_img_utils = ImageUtils()
train_images = my_img_utils.import_images(filepath=train_files, scaling=True, resizing=True, reshaping=True)

###################### SHUFFLE TRAINING IMAGES ######################

train_images_shuffle, train_classes_shuffle = my_data_utils.random_shuffle_elements(x = train_images, y = train_classes)

###################### READ VALIDATION AND TEST IMAGES ######################

validation_images = my_img_utils.__import_images(X_val, color_flag = ImageUtils.GRAYSCALE)
validation_images = my_img_utils.__resize_images(validation_images, d = (256, 256))
# validation_images = my_img_utils.scale_array_of_images(validation_images, scale_factor=255)
validation_images = my_img_utils.__reshape_images(validation_images, (256, 256, 1))

test_images = my_img_utils.__import_images(X_test, color_flag= ImageUtils.GRAYSCALE)
test_images = my_img_utils.__resize_images(test_images, d = (256, 256))
# test_images = my_img_utils.scale_array_of_images(test_images, scale_factor=255)
test_images = my_img_utils.__reshape_images(test_images, (256, 256, 1))

###################### RESHAPE IMAGE ARRAY ######################

train_images       = my_img_utils.__list2nparray(train_images_shuffle)
validation_images  = my_img_utils.__list2nparray(validation_images)
test_images        = my_img_utils.__list2nparray(test_images)

###################### EXPORT OVERSAMPLED DATASET ######################

oversampling_dir = '/Users/davidecolombo/Desktop/dataset/chest_xray_oversampling/'
train_dir = 'train/'
test_dir  = 'test/'
val_dir   = 'val/'

bacteria_dir = 'bacteria/'
normal_dir   = 'normal/'
virus_dir    = 'virus/'

bacteria_val_idx, normal_val_idx, virus_val_idx = my_data_utils.get_multiclass_indices(y= y_val, class_list=['bacteria', 'normal', 'virus'])

len(bacteria_val_idx)
len(virus_val_idx)
len(normal_val_idx)
len(validation_images[virus_val_idx, :, :, :])

# EXPORT VALIDATION - VIRUS IMAGES
my_img_utils.export_images(save_prefix='virus_',
                           save_dir=oversampling_dir + val_dir + virus_dir,
                           images=validation_images[virus_val_idx, :, :, :])

# EXPORT VALIDATION - BACTERIA IMAGES
my_img_utils.export_images(save_prefix='bacteria_',
                           save_dir=oversampling_dir + val_dir + bacteria_dir,
                           images=validation_images[bacteria_val_idx, :, :, :])

bacteria_test_idx, normal_test_idx, virus_test_idx = my_data_utils.get_multiclass_indices(y_test, class_list= ['bacteria', 'normal', 'virus'])
len(bacteria_test_idx)
len(normal_test_idx)
len(virus_test_idx)

# EXPORT TEST - VIRUS IMAGES
my_img_utils.export_images(save_prefix='virus_',
                           save_dir=oversampling_dir + test_dir + virus_dir,
                           images=test_images[virus_test_idx, :, :, :])

# EXPORT TEST - NORMAL IMAGES
my_img_utils.export_images(save_prefix='normal_',
                           save_dir=oversampling_dir + test_dir + normal_dir,
                           images=test_images[normal_test_idx, :, :, :])

# EXPORT TEST - BACTERIA IMAGES
my_img_utils.export_images(save_prefix='bacteria_',
                           save_dir=oversampling_dir + test_dir + bacteria_dir,
                           images=test_images[bacteria_test_idx, :, :, :])

bacteria_train_idx, normal_train_idx, virus_train_idx = my_data_utils.get_multiclass_indices(train_classes_shuffle, ['bacteria', 'normal', 'virus'])
len(virus_train_idx)
len(bacteria_train_idx)
len(normal_train_idx)

# EXPORT TRAIN - VIRUS IMAGES
my_img_utils.export_images(save_prefix='virus_',
                           save_dir=oversampling_dir + train_dir + virus_dir,
                           images=train_images[virus_train_idx, :, :, :])

# EXPORT TRAIN - NORMAL IMAGES
my_img_utils.export_images(save_prefix='normal_',
                           save_dir=oversampling_dir + train_dir + normal_dir,
                           images=train_images[normal_train_idx, :, :, :])

# EXPORT TRAIN - VIRUS IMAGES
my_img_utils.export_images(save_prefix='bacteria_',
                           save_dir=oversampling_dir + train_dir + bacteria_dir,
                           images=train_images[bacteria_train_idx, :, :, :])

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
    my_custom_metric.macro_specificity,
    my_custom_metric.bacteria_precision,             # bacteria precision
    my_custom_metric.bacteria_recall,                # bacteria recall
    my_custom_metric.bacteria_spec,
    my_custom_metric.normal_precision,               # normal precision
    my_custom_metric.normal_recall,                  # normal recall
    my_custom_metric.normal_spec,
    my_custom_metric.virus_precision,                # virus precision
    my_custom_metric.virus_recall,                   # virus recall
    my_custom_metric.virus_spec
]

###################### DEFINE CALLBACKS ######################

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor  = 'val_loss',
    verbose  = 1,
    patience = 10,
    mode     ='min',
    restore_best_weights = True
)

model_ckpt = tf.keras.callbacks.ModelCheckpoint(
    filepath = os.getcwd() + '/checkpoint/',
    monitor  = 'val_loss',
    mode     = 'min',
    verbose  = 1,
    save_best_only = True
)

###################### DEFINE THE MODEL ######################

my_model = ModelFactory.make_model(custom_metrics)

###################### TRAIN THE MODEL ######################

history = my_model.fit(
    train_images,
    train_one_hot,
    validation_data  = (validation_images, validation_one_hot),
    epochs           = epochs,
    batch_size       = train_batch,
    validation_batch_size = test_batch,
    verbose          = 1,
    callbacks        = [early_stopping, model_ckpt]
)

###################### IMAGE AUGMENTATION ######################

# normal_img   = 1268
# bacteria_img = 2225
# virus_img    = 1196
#
# normal2augment = bacteria_img - normal_img
# virus2augment  = bacteria_img - virus_img
#
# virus_augment_dir  = '/Users/davidecolombo/Desktop/dataset/chest_xray_augmented/train/virus/'
# normal_augment_dir = '/Users/davidecolombo/Desktop/dataset/chest_xray_augmented/train/normal/'
#
# virus_prefix = 'virus_'
# normal_prefix = 'normal_'
#
# datagen = tf.keras.preprocessing.image.ImageDataGenerator(
#     rescale=1. / 255,
#     horizontal_flip=True,
#     rotation_range = 10,
#     brightness_range=[0.8, 1.2],
#     width_shift_range=[-30, 30]
# )
#
# import random
#
# train_virus_path = [name for name in X_train if 'virus' in name]
# # print(train_virus_path)
#
# # random.sample(train_virus_path, 10)
#
# for i in range(0, 4):
#     path = ''.join(random.sample(train_virus_path, 1))
#     img = tf.keras.preprocessing.image.load_img(path= path, color_mode='grayscale', interpolation='nearest')
#     x = tf.keras.preprocessing.image.img_to_array(img)
#     x = x.reshape((1, ) + x.shape)
#     k = 0
#     for batch in datagen.flow(x, batch_size=1,
#                               save_prefix= virus_prefix,
#                               save_to_dir=virus_augment_dir,
#                               save_format='jpeg'):
#         k += 1
#         if k >= 0:
#             break
#
# train_normal_path = [name for name in X_train if 'normal' in name]
# random.sample(train_normal_path, 10)
#
# for i in range(0, 3):
#     path = ''.join(random.sample(train_normal_path, 1))
#     img = tf.keras.preprocessing.image.load_img(path= path, color_mode='grayscale', interpolation='nearest')
#     x = tf.keras.preprocessing.image.img_to_array(img)
#     x = x.reshape((1, ) + x.shape)
#     k = 0
#     for batch in datagen.flow(x, batch_size=1,
#                               save_prefix= normal_prefix,
#                               save_to_dir=normal_augment_dir,
#                               save_format='jpeg'):
#         k += 1
#         if k >= 0:
#             break
