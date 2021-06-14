
import tensorflow as tf
from ModelFactory import ModelFactory
from MyCustomMetrics import MyCustomMetrics

tf.random.set_seed(1234)

###################### DEFINE VARIABLES ######################

train_dir   = "/Users/davidecolombo/Desktop/dataset/chest_xray_final/train/"
test_dir    = "/Users/davidecolombo/Desktop/dataset/chest_xray_final/test/"
val_dir     = "/Users/davidecolombo/Desktop/dataset/chest_xray_final/val/"

# the same used for VGG16
input_shape = (224, 224, 3)
train_batch = 128
test_batch  = 64
epochs      = 50

###################### DEFINE GENERATOR ######################

datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function = tf.keras.applications.xception.preprocess_input
)

###################### TRAINING SET ######################

training = datagen.flow_from_directory(
    train_dir,
    target_size = (224, 224),
    color_mode  = 'rgb',
    class_mode  = 'categorical',
    batch_size  = train_batch,
    shuffle     = True,
    seed        = 1234
)

###################### VALIDATION SET ######################

validation = datagen.flow_from_directory(
    val_dir,
    target_size = (224, 224),
    color_mode  = 'rgb',
    class_mode  = 'categorical',
    batch_size  = test_batch,
    shuffle     = True,
    seed        = 1234
)

###################### TEST SET ######################

test = datagen.flow_from_directory(
    test_dir,
    target_size = (224, 224),
    color_mode = 'rgb',
    class_mode = 'categorical',
    batch_size = test_batch,
    shuffle     = True,
    seed        = 1234
)

###################### DEFINE CLASS WEIGHTS ######################

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

###################### DEFINE METRICS ######################

my_custom_metric = MyCustomMetrics()

custom_metrics = [
    my_custom_metric.balanced_accuracy,
    my_custom_metric.macro_f1score,
    my_custom_metric.macro_precision,
    my_custom_metric.macro_recall,
    my_custom_metric.bacteria_precision,             # bacteria precision
    my_custom_metric.macro_bacteria_recall,                # bacteria recall
    my_custom_metric.normal_precision,               # normal precision
    my_custom_metric.macro_normal_recall,                  # normal recall
    my_custom_metric.virus_precision,                # virus precision
    my_custom_metric.macro_virus_recall                    # virus recall
]

###################### MAKE MODEL ######################

xception_model = ModelFactory.make_xception(metrics = custom_metrics, learning_rate = 3e-4, shape = input_shape)

###################### TRAIN THE MODEL ######################

xception_history = xception_model.fit(
    training,
    epochs = epochs,
    steps_per_epoch = 4686 // train_batch,
    validation_steps= 585 // test_batch,
    validation_data= validation,
    class_weight = class_weight,
    # callbacks = [],
    verbose = 1
)

###################### VISUALIZE TRAINING INSIGHTS ######################


###################### TEST THE MODEL ######################



