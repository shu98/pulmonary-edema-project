import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import math
import numpy as np  

class Annotation:

    def __init__(self, subject, study, date, edema=float('nan'), severity=float('nan'), previous=None, comparison=float('nan')):
        """
        subject     anonymized patient ID
        study       study ID 
        date        date of study 

        edema       yes/no presence of edema, nan if no mention 
        severity    severity score for pulmonary edema, nan if no mention
        previous    previous radiology report annotation that this report is compared to 
        comparison  better/worse/same condition, nan if no comparison 

        """
        self.subject = subject
        self.study = study 
        self.date = date 

        self.previous = previous
        self.edema = abs(self._infer_edema(edema, comparison))
        self.severity = severity 
        self.comparison = self._infer_comparison(comparison)

    def _infer_edema(self, edema, comparison):
        if not math.isnan(edema):
            return edema 

        if comparison == 0.0 or math.isnan(comparison):
            return self.previous.edema 

        return float('nan')

    def _infer_comparison(self, comparison):
        if not math.isnan(comparison):
            return comparison 

        # No previous report to compare to 
        if self.previous is None:
            return float('nan')

        # Both reports have a severity score 
        if not math.isnan(self.previous.severity) and not math.isnan(self.severity):
            return np.sign(self.previous.severity - self.severity)

        if self.severity == 0.0 or self.edema == 0.0:
            if self.previous.edema == 0.0:
                return 0.0

            if self.previous.edema == 1.0 or self.previous.severity > 0.0:
                return 1.0

        if self.severity > 0.0 or self.edema == 1.0:
            if self.previous.edema == 1.0 or self.previous.severity > 0.0:
                return 0.0

            if self.previous.edema == 0.0 or self.previous.severity == 0.0:
                return -1.0 

        return float('nan')







