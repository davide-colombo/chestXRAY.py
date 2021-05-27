
import tensorflow as tf
from ModelFactory import ModelFactory
from MyCustomMetrics import MyCustomMetrics

tf.random.set_seed(1234)

###################### DEFINE PARAMETERS ######################

data_dir    = "/Users/davidecolombo/Desktop/dataset/chest_xray_keras/"
input_shape = (256, 256, 3)
batch_size  = 128
epochs      = 50

###################### DEFINE GENERATOR ######################

datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function= tf.keras.applications.xception.preprocess_input,
    validation_split= 0.3
)

###################### TRAINING SET ######################

training = datagen.flow_from_directory(
    data_dir,
    target_size = (256, 256),
    class_mode  = 'categorical',
    batch_size  = batch_size,
    subset      = 'training',
    shuffle     = True,
    seed        = 333
)

###################### VALIDATION SET ######################

validation = datagen.flow_from_directory(
    data_dir,
    target_size = (256, 256),
    class_mode  = 'categorical',
    batch_size  = batch_size,
    subset      = 'validation',
    shuffle     = True,
    seed        = 333
)

###################### DEFINE METRICS ######################

my_custom_metric = MyCustomMetrics()

custom_metrics = [
    my_custom_metric.categorical_weighted_accuracy,
    my_custom_metric.categorical_balanced_accuracy,
    my_custom_metric.categorical_f1_score,
    my_custom_metric.categorical_precision,
    my_custom_metric.categorical_recall,
    my_custom_metric.categorical_true_positives
]

###################### DEFINE THE MODEL ######################

xception_model = ModelFactory.make_xception(metrics = custom_metrics, learning_rate = 3e-4, shape = input_shape)

xception_history = xception_model.fit(
    training,
    batch_size= batch_size,
    epochs = epochs,
    steps_per_epoch = 4101 // batch_size,
    validation_steps= 1755 // batch_size,
    validation_data= validation,
    verbose = 1
)
