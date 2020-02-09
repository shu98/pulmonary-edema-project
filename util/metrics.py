import sklearn.metrics

def sample_avg_auc(y_true, y_pred):
    return sklearn.metrics.roc_auc_score(y_true=y_true,
            y_score=y_pred, average='samples')

def expected(y_true, y_pred):
    return sum(y_pred)

def observed(y_true, y_pred):
    return sum(y_true)

def observed_over_expected(y_true, y_pred):
    return float(observed(y_true=y_true, y_pred=y_pred))/ expected(y_true=y_true, y_pred=y_pred)

def precision(y_true, y_pred):
    return sklearn.metrics.precision_score(y_true, y_pred, average="binary")

def recall(y_true, y_pred):
    return sklearn.metrics.recall_score(y_true, y_pred, average="binary")

def f1(y_true, y_pred):
    return sklearn.metrics.f1_score(y_true, y_pred, average="binary")

def sens_spec(y_true, y_pred):
    tp, tn, fp, fn = 0, 0, 0, 0
    for index, value in enumerate(y_true):
        if value == 1.0 and y_pred[index] == 1.0: tp += 1
        if value == 0.0 and y_pred[index] == 0.0: tn += 1
        if value == 1.0 and y_pred[index] == 0.0: fn += 1
        if value == 0.0 and y_pred[index] == 1.0: fp += 1

    return {'sens': tp / (tp + fn),
            'spec': tn / (tn + fp),
            'fpr': fp / (tn + fp),
            'fnr': fn / (tp + fn),
            'tpv': tp / (tp + fp),
            'npv': tn / (tn + fn)}

METRIC_ORDER = ['accuracy', 'precision', 'recall', 'f1','sens_spec']

METRIC_TO_EVAL_FUNC = {
    'accuracy': sklearn.metrics.accuracy_score,
    'precision': precision,
    'recall': recall,
    'f1': f1,
    'auc': sample_avg_auc,
    'expected': expected,
    'observed': observed,
    'O/E': observed_over_expected,
    'sens_spec': sens_spec
}