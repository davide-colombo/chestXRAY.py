
import tensorflow as tf

class MyCustomMetrics:

    def __categorical_confusion_mat(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        y_pred = tf.cast(tf.equal(y_pred, tf.reduce_max(y_pred, axis=-1, keepdims=True)), tf.float32)
        y_true = tf.argmax(y_true, axis=1)
        y_pred = tf.argmax(y_pred, axis=1)
        return tf.math.confusion_matrix(y_true, y_pred)

    def __categorical_macro_recall(self, y_true, y_pred):
        true_pos  = self.__macro_true_pos(y_true, y_pred)
        false_neg = self.__macro_false_neg(y_true, y_pred)
        num = tf.cast(true_pos, tf.float32)
        den = tf.cast(tf.add(true_pos, false_neg), tf.float32)
        den = tf.add(den, 1e-7)
        return tf.divide(num, den)

    def __categorical_macro_precision(self, y_true, y_pred):
        true_pos  = self.__macro_true_pos(y_true, y_pred)
        false_pos = self.__macro_false_pos(y_true, y_pred)
        num = tf.cast(true_pos, tf.float32)
        den = tf.cast(tf.add(true_pos, false_pos), tf.float32)
        den = tf.add(den, 1e-7)
        return tf.divide(num, den)

    def __macro_false_pos(self, y_true, y_pred):
        cm        = self.__categorical_confusion_mat(y_true, y_pred)
        true_pos  = tf.linalg.diag_part(cm)
        return tf.subtract(tf.reduce_sum(cm, axis=0), true_pos)

    def __macro_true_pos(self, y_true,y_pred):
        cm = self.__categorical_confusion_mat(y_true, y_pred)
        return tf.linalg.diag_part(cm)

    def __macro_false_neg(self, y_true, y_pred):
        cm       = self.__categorical_confusion_mat(y_true, y_pred)
        true_pos = tf.linalg.diag_part(cm)
        return tf.subtract(tf.reduce_sum(cm, axis=1), true_pos)

###################### CLASS FALSE POSITIVES ######################

    def false_neg_bacteria(self, y_true, y_pred):
        false_neg = self.__macro_false_neg(y_true, y_pred)
        return tf.reduce_sum(tf.multiply(false_neg, tf.constant([1, 0, 0])))

    def false_neg_normal(self, y_true, y_pred):
        false_neg = self.__macro_false_neg(y_true, y_pred)
        return tf.reduce_sum(tf.multiply(false_neg, tf.constant([0, 1, 0])))

    def false_neg_virus(self, y_true, y_pred):
        false_neg = self.__macro_false_neg(y_true, y_pred)
        return tf.reduce_sum(tf.multiply(false_neg, tf.constant([0, 0, 1])))

###################### CLASS FALSE POSITIVES ######################

    def false_pos_bacteria(self, y_true, y_pred):
        false_pos = self.__macro_false_pos(y_true, y_pred)
        return tf.reduce_sum(tf.multiply(false_pos, tf.constant([1, 0, 0])))

    def false_pos_normal(self, y_true, y_pred):
        false_pos = self.__macro_false_pos(y_true, y_pred)
        return tf.reduce_sum(tf.multiply(false_pos, tf.constant([0, 1, 0])))

    def false_pos_virus(self, y_true, y_pred):
        false_pos = self.__macro_false_pos(y_true, y_pred)
        return tf.reduce_sum(tf.multiply(false_pos, tf.constant([0, 0, 1])))

###################### CLASS TRUE POSITIVES ######################

    def true_pos_bacteria(self, y_true, y_pred):
        true_pos = self.__macro_true_pos(y_true, y_pred)
        return tf.reduce_sum(tf.multiply(true_pos, tf.constant([1, 0, 0])))

    def true_pos_normal(self, y_true, y_pred):
        true_pos = self.__macro_true_pos(y_true, y_pred)
        return tf.reduce_sum(tf.multiply(true_pos, tf.constant([0, 1, 0])))

    def true_pos_virus(self, y_true, y_pred):
        true_pos = self.__macro_true_pos(y_true, y_pred)
        return tf.reduce_sum(tf.multiply(true_pos, tf.constant([0, 0, 1])))

###################### CLASS RECALL ######################

    def macro_virus_recall(self, y_true, y_pred):
        class_recall = tf.cast(self.__categorical_macro_recall(y_true, y_pred), tf.float32)
        return tf.reduce_sum(tf.multiply(class_recall, tf.constant([0, 0, 1], dtype = tf.float32)))

    def macro_normal_recall(self, y_true, y_pred):
        class_recall = tf.cast(self.__categorical_macro_recall(y_true, y_pred), tf.float32)
        return tf.reduce_sum(tf.multiply(class_recall, tf.constant([0, 1, 0], dtype=tf.float32)))

    def macro_bacteria_recall(self, y_true, y_pred):
        class_recall = tf.cast(self.__categorical_macro_recall(y_true, y_pred), tf.float32)
        return tf.reduce_sum(tf.multiply(class_recall, tf.constant([1, 0, 0], dtype=tf.float32)))

###################### CLASS PRECISION ######################

    def macro_virus_precision(self, y_true, y_pred):
        class_precision = tf.cast(self.__categorical_macro_precision(y_true, y_pred), tf.float32)
        return tf.reduce_sum(tf.multiply(class_precision, tf.constant([0, 0, 1], dtype=tf.float32)))

    def macro_normal_precision(self, y_true, y_pred):
        class_precision = tf.cast(self.__categorical_macro_precision(y_true, y_pred), tf.float32)
        return tf.reduce_sum(tf.multiply(class_precision, tf.constant([0, 1, 0], dtype=tf.float32)))

    def macro_bacteria_precision(self, y_true, y_pred):
        class_precision = tf.cast(self.__categorical_macro_precision(y_true, y_pred), tf.float32)
        return tf.reduce_sum(tf.multiply(class_precision, tf.constant([1, 0, 0], dtype = tf.float32)))

###################### MACRO-AVERAGED METRICS ######################

    def macro_precision(self, y_true, y_pred):
        class_precision = tf.cast(self.__categorical_macro_precision(y_true, y_pred), tf.float32)
        return tf.reduce_mean(class_precision)

    def macro_recall(self, y_true, y_pred):
        class_recall = tf.cast(self.__categorical_macro_recall(y_true, y_pred), tf.float32)
        return tf.reduce_mean(class_recall)

    def macro_f1score(self, y_true, y_pred):
        p            = tf.cast(self.__categorical_macro_precision(y_true, y_pred), tf.float32)
        r            = tf.cast(self.__categorical_macro_recall(y_true, y_pred), tf.float32)
        num = tf.multiply(p, r)
        num = tf.multiply(num, 2.0)
        den = tf.add(p, r)
        den = tf.add(den, 1e-7)
        f1score = tf.divide(num, den)
        return tf.reduce_mean(f1score)

    # @Source: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html
    def balanced_accuracy(self, y_true, y_pred):
        cm       = self.__categorical_confusion_mat(y_true, y_pred)
        actual   = tf.cast(tf.reduce_sum(cm, axis = 1), tf.float32)
        correct  = tf.cast(tf.linalg.diag_part(cm), tf.float32)
        accuracy = tf.divide(correct, tf.add(actual, 1e-7))
        return tf.reduce_mean(accuracy)

###################### THIS IS A TEST ######################

# from sklearn.metrics import f1_score
# from sklearn.metrics import balanced_accuracy_score
#
# my_custom_metrics = MyCustomMetrics()
#
# # CHECK
# my_custom_metrics.macro_weighted_f1score(y_true = [[0, 1, 0],       [0, 1, 0],       [0, 0, 1],       [1, 0, 0], [0, 0, 1]],
#                                          y_pred = [[0.1, 0.8, 0.1], [0.2, 0.7, 0.1], [0.4, 0.3, 0.3], [0, 1, 0], [0, 0, 1]])
#
# f1_score(y_true = [1, 1, 2, 0, 2],
#          y_pred = [1, 1, 0, 1, 2],
#          average = 'macro')
#
# balanced_accuracy_score(y_true = [1, 1, 2, 0, 2],
#                         y_pred = [1, 1, 0, 1, 2])
#
# from sklearn.metrics import precision_score
# from sklearn.metrics import recall_score
#
# precision_score(y_true = [1, 1, 2, 0, 2],
#                 y_pred = [1, 1, 0, 1, 2],
#                 average = 'macro')
#
# recall_score(y_true = [1, 1, 2, 0, 2],
#              y_pred = [1, 1, 0, 1, 2],
#              average = 'macro')


# no good for class imbalance!!!
# accuracy_score(y_true = [1, 1, 2, 0, 2],
#                y_pred = [1, 1, 0, 1, 2],
#                normalize=True)

# def __get_class_weight(self, y_true, y_pred):
    #     cm     = self.__categorical_confusion_mat(y_true, y_pred)
    #     actual = tf.reduce_sum(cm, axis=1)
    #     total  = tf.reduce_sum(actual)
    #     return tf.divide(actual, total)

# @Source: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html#sklearn.metrics.accuracy_score
    # def macro_weighted_accuracy(self, y_true, y_pred):
    #     cm       = self.__categorical_confusion_mat(y_true, y_pred)
    #     weight   = tf.cast(self.__get_class_weight(y_true, y_pred), tf.float32)
    #     actual   = tf.cast(tf.reduce_sum(cm, axis = 1), tf.float32)
    #     correct  = tf.cast(tf.linalg.diag_part(cm), tf.float32)
    #     accuracy = tf.divide(correct, tf.add(actual, 1e-7))
    #     return tf.reduce_sum(tf.multiply(accuracy, weight))

# class_weight    = tf.cast(self.__get_class_weight(y_true, y_pred), tf.float32)

