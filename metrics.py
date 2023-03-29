from sklearn.metrics import average_precision_score, roc_curve, auc, precision_recall_curve, confusion_matrix
import numpy as np
import warnings
warnings.filterwarnings("ignore")


class EcgMetrics(object):
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.metric_names = ['roc_auc_micro', 'roc_auc_macro', 'avg_prec_micro', 'avg_prec_macro']

    def compute_metrics(self, y_pred, y_true):
        # allocation
        metrics = dict()

        # compute roc auc (micro and macro)
        metrics['roc_auc_micro'], metrics['roc_auc_macro'] = compute_roc_auc(y_pred, y_true, self.n_classes)

        # compute average precision (micro and macro)
        metrics['avg_prec_micro'] = average_precision_score(y_true, y_pred, average='micro')
        metrics['avg_prec_macro'] = average_precision_score(y_true, y_pred, average='macro')

        return metrics


def get_optimal_precision_recall(y_true, y_score):
    """Find precision and recall values that maximize f1 score."""
    n = np.shape(y_true)[1]
    opt_precision = []
    opt_recall = []
    opt_threshold = []
    for k in range(n):
        # Get precision-recall curve
        precision, recall, threshold = precision_recall_curve(y_true[:, k], y_score[:, k])
        # Compute f1 score for each point (use nan_to_num to avoid nans messing up the results)
        f1_score = np.nan_to_num(2 * precision * recall / (precision + recall))
        # Select threshold that maximize f1 score
        index = np.argmax(f1_score)
        opt_precision.append(precision[index])
        opt_recall.append(recall[index])
        t = threshold[index-1] if index != 0 else threshold[0]-1e-10
        opt_threshold.append(t)
    return np.array(opt_precision), np.array(opt_recall), np.array(opt_threshold)


def compute_scores_after_thresh(y_true, y_pred, n = None):
    """Find precision and recall values that maximize f1 score."""
    if n is None:
        n = np.shape(y_true)[1]
    scores = {name: np.zeros(n) for name in ['precision (PPV)', 'recall (SEN)',
                                             'NPV', 'f1_score']}
    for i in range(n):
        tn, fp, fn, tp = confusion_matrix(y_true[:, i], y_pred[:, i]).ravel()
        scores['precision (PPV)'][i] = tp / (tp + fp)
        scores['recall (SEN)'][i] = tp / (tp + fn)
        scores['NPV'][i] = tn / (tn + fn)
        scores['f1_score'][i] = 2 * scores['precision (PPV)'][i] * scores['recall (SEN)'][i] / (scores['precision (PPV)'][i] + scores['recall (SEN)'][i])
    return scores


def compute_roc_auc(y_pred, y_true, n_classes):
    """
    from https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
    """

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), y_pred.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    # Finally average it
    mean_tpr /= n_classes

    # compute macro auc
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    return roc_auc['micro'], roc_auc['macro']
