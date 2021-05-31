
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

    def __categorical_macro_recall(self, y_true, y_pred):
        cm = self.__categorical_confusion_mat(y_true, y_pred)
        true_pos  = tf.linalg.diag_part(cm)
        false_neg = tf.subtract(tf.reduce_sum(cm, axis=1), true_pos)
        num = tf.cast(true_pos, tf.float32)
        den = tf.cast(tf.add(true_pos, false_neg), tf.float32)
        den = tf.add(den, 1e-7)
        return tf.divide(num, den)

    def __categorical_macro_precision(self, y_true, y_pred):
        cm = self.__categorical_confusion_mat(y_true, y_pred)
        true_pos = tf.linalg.diag_part(cm)
        false_pos = tf.subtract(tf.reduce_sum(cm, axis=0), true_pos)
        num = tf.cast(true_pos, tf.float32)
        den = tf.cast(tf.add(true_pos, false_pos), tf.float32)
        den = tf.add(den, 1e-7)
        return tf.divide(num, den)

    def __get_class_weight(self, y_true, y_pred):
        cm = self.__categorical_confusion_mat(y_true, y_pred)
        actual = tf.reduce_sum(cm, axis=1)
        total = tf.reduce_sum(actual)
        return tf.divide(actual, total)

########################## PUBLIC METHODS ##########################

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

    def macro_weighted_precision(self, y_true, y_pred):
        class_weight    = tf.cast(self.__get_class_weight(y_true, y_pred), tf.float32)
        class_precision = tf.cast(self.__categorical_macro_precision(y_true, y_pred), tf.float32)
        return tf.reduce_sum(tf.multiply(class_precision, class_weight))

    def macro_weighted_recall(self, y_true, y_pred):
        class_recall = tf.cast(self.__categorical_macro_recall(y_true, y_pred), tf.float32)
        class_weight = tf.cast(self.__get_class_weight(y_true, y_pred), tf.float32)
        return tf.reduce_sum(tf.multiply(class_recall, class_weight))

    def macro_weighted_f1score(self, y_true, y_pred):
        class_weight = tf.cast(self.__get_class_weight(y_true, y_pred), tf.float32)
        p            = tf.cast(self.__categorical_macro_precision(y_true, y_pred), tf.float32)
        r            = tf.cast(self.__categorical_macro_recall(y_true, y_pred), tf.float32)
        num = tf.multiply(p, r)
        num = tf.multiply(num, 2.0)
        den = tf.add(p, r)
        den = tf.add(den, 1e-7)
        f1score = tf.divide(num, den)
        return tf.reduce_sum(tf.multiply(f1score, class_weight))

    def categorical_recall(self, y_true, y_pred):
        true_pos  = self.categorical_true_positives(y_true, y_pred)
        false_neg = self.categorical_false_negatives(y_true, y_pred)
        return true_pos / (true_pos + false_neg)

    # @Source: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
    def categorical_f1_score(self, y_true, y_pred):
        precision = self.categorical_precision(y_true, y_pred)
        recall    = self.categorical_recall(y_true, y_pred)
        return 2 * (precision * recall) / (precision + recall)

    # @Source: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html#sklearn.metrics.accuracy_score
    def categorical_weighted_accuracy(self, y_true, y_pred):
        cm       = self.__categorical_confusion_mat(y_true, y_pred)
        weight   = tf.cast(self.__get_class_weight(y_true, y_pred), tf.float32)
        actual   = tf.cast(tf.reduce_sum(cm, axis = 1), tf.float32)
        correct  = tf.cast(tf.linalg.diag_part(cm), tf.float32)
        accuracy = tf.divide(correct, tf.add(actual, 1e-7))
        return tf.reduce_sum(tf.multiply(accuracy, weight))

    # @Source: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html
    def categorical_balanced_accuracy(self, y_true, y_pred):
        cm       = self.__categorical_confusion_mat(y_true, y_pred)
        actual   = tf.cast(tf.reduce_sum(cm, axis = 1), tf.float32)
        correct  = tf.cast(tf.linalg.diag_part(cm), tf.float32)
        accuracy = tf.divide(correct, tf.add(actual, 1e-7))
        return tf.reduce_mean(accuracy)

# THIS IS A TEST

from sklearn.metrics import f1_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import accuracy_score

my_custom_metrics = MyCustomMetrics()

# CHECK
my_custom_metrics.macro_weighted_f1score(y_true = [[0, 1, 0],       [0, 1, 0],       [0, 0, 1],       [1, 0, 0], [0, 0, 1]],
                                         y_pred = [[0.1, 0.8, 0.1], [0.2, 0.7, 0.1], [0.4, 0.3, 0.3], [0, 1, 0], [0, 0, 1]])

f1_score(y_true = [1, 1, 2, 0, 2],
         y_pred = [1, 1, 0, 1, 2],
         average = 'weighted')

balanced_accuracy_score(y_true = [1, 1, 2, 0, 2],
                        y_pred = [1, 1, 0, 1, 2])

accuracy_score(y_true = [1, 1, 2, 0, 2],
               y_pred = [1, 1, 0, 1, 2],
               normalize=True)

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

precision_score(y_true = [1, 1, 2, 0, 2],
                y_pred = [1, 1, 0, 1, 2],
                average = 'weighted')

recall_score(y_true = [1, 1, 2, 0, 2],
             y_pred = [1, 1, 0, 1, 2],
             average = 'weighted')
