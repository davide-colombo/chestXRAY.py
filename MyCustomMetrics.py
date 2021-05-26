
import tensorflow as tf

class MyCustomMetrics:

    # private method
    def __categorical_confusion_mat(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)                    # cast to obtain a tensor
        y_pred = tf.cast(y_pred, tf.float32)
        y_pred = tf.cast(tf.equal(y_pred, tf.reduce_max(y_pred, axis=-1, keepdims=True)), tf.float32)
        y_true = tf.argmax(y_true, axis=1)                      # convert from one-hot to numeric
        y_pred = tf.argmax(y_pred, axis=1)
        return tf.math.confusion_matrix(y_true, y_pred)         # confusion matrix

    def categorical_true_positives(self, y_true, y_pred):
        cm = self.__categorical_confusion_mat(y_true, y_pred)
        return tf.linalg.trace(cm)

    def categorical_false_positives(self, y_true, y_pred):
        cm       = self.__categorical_confusion_mat(y_true, y_pred)
        diagonal = tf.linalg.diag_part(cm)
        return tf.reduce_sum(tf.subtract(tf.reduce_sum(cm, axis=0, keepdims=True), diagonal))

    def categorical_false_negatives(self, y_true, y_pred):
        cm       = self.__categorical_confusion_mat(y_true, y_pred)
        diagonal = tf.linalg.diag_part(cm)
        return tf.reduce_sum(tf.subtract(tf.reduce_sum(cm, axis=1), diagonal))

    def categorical_precision(self, y_true, y_pred):
        true_pos  = self.categorical_true_positives(y_true, y_pred)
        false_pos = self.categorical_false_positives(y_true, y_pred)
        return true_pos / (false_pos + true_pos)

    def categorical_recall(self, y_true, y_pred):
        true_pos  = self.categorical_true_positives(y_true, y_pred)
        false_neg = self.categorical_false_negatives(y_true, y_pred)
        return true_pos / (true_pos + false_neg)

    def categorical_f1_score(self, y_true, y_pred):
        precision = self.categorical_precision(y_true, y_pred)
        recall    = self.categorical_recall(y_true, y_pred)
        return 2 * (precision * recall) / (precision + recall)


# CHECK
# categorical_f1_score(y_true = [[0, 1, 0], [0, 1, 0], [0, 0, 1]],
#                      y_pred = [[0.1, 0.8, 0.1], [0.2, 0.7, 0.1], [0.4, 0.3, 0.3]])
