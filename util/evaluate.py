import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import numpy as np 
from util.metrics import METRIC_ORDER, METRIC_TO_EVAL_FUNC

def evaluate(y_true, y_pred, average="binary"):    
    results = {}
    sens_spec = {}
    for m in METRIC_ORDER:
        if m == 'sens_spec': 
            if average == "binary":
                sens_spec = METRIC_TO_EVAL_FUNC[m](y_true, y_pred)
        elif m in ['precision', 'recall', 'f1']:
            results[m] = METRIC_TO_EVAL_FUNC[m](y_true, y_pred, average=average)
        else: 
            results[m] = METRIC_TO_EVAL_FUNC[m](y_true, y_pred)

    results.update(sens_spec)
    return results  

def evaluate_nclass(y_true, y_pred, classes):
    """
    Computes metrics for each class as class vs. all 
    """
    results = {}
    for c in classes:
        y_true_copy = change_to_binary(c, np.copy(y_true))
        y_pred_copy = change_to_binary(c, np.copy(y_pred))
        

        results[c] = evaluate(y_true_copy, y_pred_copy)
    return results

def change_to_binary(target_class, array):
    for i in range(array.shape[0]):
        if array[i] == target_class:
            array[i] = 1.0
        else:
            array[i] = 0.0

    return array

