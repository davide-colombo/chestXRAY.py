
import tensorflow as tf

class ModelFactory:

    @staticmethod
    def make_vgg16(metrics, img_size, channels):

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
        top_model = conv_base.output
        top_model = tf.keras.layers.GlobalAveragePooling2D()(top_model)
        # top_model = tf.keras.layers.Dense(units=128, activation='relu')(top_model)
        # top_model = tf.keras.layers.Dropout(rate=0.5)(top_model)
        # top_model = tf.keras.layers.Dense(units = 128, activation='relu')(top_model)
        # top_model = tf.keras.layers.Dropout(rate=0.5)(top_model)
        # top_model = tf.keras.layers.Dense(units = 64, activation='relu')(top_model)
        # top_model = tf.keras.layers.Dropout(rate=0.5)(top_model)
        output_layer = tf.keras.layers.Dense(units=3, activation='softmax')(top_model)

        # final model
        model = tf.keras.Model(inputs=conv_base.input, outputs=output_layer)

        # summary
        model.summary()

        # compile the model
        model.compile(
            optimizer= tf.keras.optimizers.Adam(),
            loss     = tf.keras.losses.CategoricalCrossentropy(),
            metrics  = metrics
        )

        return model


