import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from util.metrics import METRIC_ORDER, METRIC_TO_EVAL_FUNC

def evaluate(y_true,  y_pred):
    results = {}
    sens_spec = {}
    for m in METRIC_ORDER:
        if m == 'sens_spec': 
            sens_spec = METRIC_TO_EVAL_FUNC[m](y_true, y_pred)
        else: 
            results[m] = METRIC_TO_EVAL_FUNC[m](y_true, y_pred)

    results.update(sens_spec)
    return results 	