import numpy as np


def _get_conf_matrix(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    conf_matrix = np.bincount(
        n_class * label_true[mask].astype(int) + label_pred[mask],
        minlength=n_class ** 2,
    ).reshape(n_class, n_class)
    return conf_matrix


def scores(label_trues, label_preds, n_class):
    conf_matrix = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        conf_matrix += _get_conf_matrix(lt.flatten(), lp.flatten(), n_class)

    condition_positive = np.sum(conf_matrix, axis=1)
    acc = np.diag(conf_matrix).sum() / conf_matrix.sum()
    cls_recall = np.diag(conf_matrix) / conf_matrix.sum(axis=1)
    macro_recall = np.nanmean(cls_recall)
    cls_precision = np.diag(conf_matrix) / np.sum(conf_matrix, axis=0)
    macro_precision = np.nanmean(cls_precision)
    cls_F_score = (2 * cls_precision * cls_recall) / (cls_precision + cls_recall)
    F_score = np.nanmean(cls_F_score)

    cls_IoU = np.diag(conf_matrix) / (conf_matrix.sum(axis=1) + conf_matrix.sum(axis=0) - np.diag(conf_matrix))
    valid = conf_matrix.sum(axis=1) > 0  # added, if some class don't exit, then don't take it into count
    mean_IoU = np.nanmean(cls_IoU[valid])
    freq = conf_matrix.sum(axis=1) / conf_matrix.sum()
    freq_weighted_IoU = (freq[freq > 0] * cls_IoU[freq > 0]).sum()

    return {
        "Frequency Weighted IoU": freq_weighted_IoU,
        "Mean IoU": mean_IoU,
        "Class IoU": dict(zip(range(n_class), cls_IoU)),
        "Accuracy": acc,
        "Macro Recall": macro_recall,
        "Class Recall": dict(zip(range(n_class), cls_recall)),
        "Class Precision": dict(zip(range(n_class), cls_precision)),
        "Macro Precision": macro_precision,
        "F Score": F_score,
        "Class Samples Num": dict(zip(range(n_class), condition_positive))
    }
