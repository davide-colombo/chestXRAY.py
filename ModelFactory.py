
import tensorflow as tf

class ModelFactory:

    @staticmethod
    def make_vgg16(metrics, learning_rate, img_size, channels):

        conv_base = tf.keras.applications.VGG16(
            include_top=False,
            weights='imagenet',
            pooling='None',
            input_shape=(img_size, img_size, channels)
        )

        # freeze the layers
        for layer in conv_base.layers:
            layer.trainable = False

        # define the model
        model = tf.keras.Sequential([
            conv_base,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(units=3, activation='softmax')
        ])

        # summary
        model.summary()

        # compile the model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=metrics
        )

        # define the model
        # top_model = conv_base.output
        # top_model = tf.keras.layers.GlobalAveragePooling2D()(top_model)
        # top_model = tf.keras.layers.Dense(units=128, activation='relu')(top_model)
        # top_model = tf.keras.layers.Dropout(rate=0.5)(top_model)
        # top_model = tf.keras.layers.Dense(units = 128, activation='relu')(top_model)
        # top_model = tf.keras.layers.Dropout(rate=0.5)(top_model)
        # top_model = tf.keras.layers.Dense(units = 64, activation='relu')(top_model)
        # top_model = tf.keras.layers.Dropout(rate=0.5)(top_model)
        # output_layer = tf.keras.layers.Dense(units=3, activation='softmax')(top_model)

        # final model
        # model = tf.keras.Model(inputs=conv_base.input, outputs=output_layer)

        return model

    @staticmethod
    def make_xception(metrics, learning_rate, shape):

        conv_base = tf.keras.applications.Xception(
            include_top=False, weights='imagenet', pooling='None', input_shape = shape
        )

        # freeze conv layers
        for layer in conv_base.layers:
            layer.trainable = False

        # define the model
        model = tf.keras.Sequential([
            conv_base,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(units = 3, activation = 'softmax')
        ])

        # summary
        model.summary()

        # compile
        model.compile(
            optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate),
            loss      = tf.keras.losses.CategoricalCrossentropy(),
            metrics   = metrics
        )

        return model


# MAKE MODEL FROM SCRATCH
    @staticmethod
    def make_model(metrics, lr = 3e-4):
        # define the model
        model = tf.keras.Sequential([
            # CONV BLOCK 1 (MISS L2 REG)
            tf.keras.layers.Conv2D(filters = 64, kernel_size = (7, 7), strides = 2, padding = "same", activation = 'relu', input_shape = (256, 256, 1)),
            tf.keras.layers.Conv2D(filters = 64, kernel_size = (1, 1), strides = 1, padding = 'same', activation = 'relu'),
            tf.keras.layers.MaxPool2D(pool_size = 2, strides = 1, padding = "same"),
            # CONV BLOCK 2 (MISS L2 REG)
            tf.keras.layers.Conv2D(filters=64, kernel_size=(5, 5), strides=2, padding="same", activation='relu'),
            tf.keras.layers.Conv2D(filters=64, kernel_size=(1, 1), strides=1, padding='same', activation='relu'),
            tf.keras.layers.MaxPool2D(pool_size=2, strides=1, padding="same"),
            # CONV BLOCK 3 (MISS L2 REG)
            tf.keras.layers.Conv2D(filters=64, kernel_size=(5, 5), strides=2, padding="same", activation='relu'),
            tf.keras.layers.Conv2D(filters=64, kernel_size=(1, 1), strides=1, padding='same', activation='relu'),
            tf.keras.layers.MaxPool2D(pool_size=2, strides=1, padding="same"),
            # CONV BLOCK 4 (MISS L2 REG)
            tf.keras.layers.Conv2D(filters=32, kernel_size=(7, 7), strides=2, padding="same", activation='relu'),
            tf.keras.layers.Conv2D(filters=64, kernel_size=(1, 1), strides=1, padding='same', activation='relu'),
            tf.keras.layers.MaxPool2D(pool_size=2, strides=1, padding="same"),
            # FLATTEN
            tf.keras.layers.Flatten(),
            # DENSE LAYER 1
            tf.keras.layers.Dense(units=256, activation='relu'),
            tf.keras.layers.Dense(units = 3, activation = 'softmax')
        ])

        # view summary
        model.summary()

        # compile
        model.compile(
            optimizer = tf.keras.optimizers.Adam(learning_rate = lr),
            loss      = tf.keras.losses.CategoricalCrossentropy(),
            metrics   = metrics
        )

        return model



