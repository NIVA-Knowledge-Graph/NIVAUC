## custom_metrics

from sklearn.metrics import roc_auc_score,precision_score,recall_score,f1_score,fbeta_score
import tensorflow as tf


def my_round(x,th):
    return tf.round(x-th+0.5)

def my_auc(y_true,y_pred):
    try:
        return roc_auc_score(y_true, y_pred)
    except ValueError:
        return 1

def keras_auc(y_true, y_pred):
    return tf.py_function(my_auc, (y_true, y_pred), tf.double)

def keras_recall(y_true, y_pred):
    return tf.py_function(recall_score, (y_true, my_round(y_pred,0.5)), tf.double)

def keras_precision(y_true, y_pred):
    return tf.py_function(precision_score, (y_true, my_round(y_pred,0.5)), tf.double)

def keras_f1(y_true, y_pred):
    return 2 * (keras_precision(y_true,y_pred)*keras_recall(y_true,y_pred))/(keras_precision(y_true,y_pred)+keras_recall(y_true,y_pred))

def keras_fb(y_true, y_pred, beta = 2):
    return (1+beta**2) * (keras_precision(y_true,y_pred)*keras_recall(y_true,y_pred))/(beta**2*keras_precision(y_true,y_pred)+keras_recall(y_true,y_pred))

