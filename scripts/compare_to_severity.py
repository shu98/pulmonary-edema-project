import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from datetime import datetime, timedelta
import math
import numpy as np 
import os 
import pandas as pd
from pprint import pprint 
from scipy.special import softmax

from dataset.organize_reports import sort_by_date, get_metadata
from scripts.compare_to_ray import *

def collect_no_severity(severity_pairwise, comparison_labels):
    comparison_no_severity, comparison_severity = [], []
    no_comparison_no_severity, severity_no_comparison = 0, 0

    for subject, study1, study2, label1, label2 in severity_pairwise:
        if math.isnan(comparison_labels['comparison'][study2]):
            # No comparison, no severity 
            if (math.isnan(label1) or math.isnan(label2)): no_comparison_no_severity += 1
            # No comparison, yes severity
            else: severity_no_comparison += 1 
            continue 

        if (math.isnan(label1) or math.isnan(label2)):
            # Yes comparison, no severity
            if not math.isnan(comparison_labels['comparison'][study2]):
                comparison_no_severity.append([subject, study1, study2, label1, label2, comparison_labels['comparison'][study2]])
            # No comparison, no severity 
            else: no_comparison_no_severity += 1

        # Yes comparison, yes severity
        if not math.isnan(label1) and not math.isnan(label2) and not math.isnan(comparison_labels['comparison'][study2]):
            comparison_severity.append([subject, study1, study2, label1, label2, comparison_labels['comparison'][study2]])

    print("Total pairs with severity but no comparison:", severity_no_comparison)
    print("Total pairs with no comparison or severity:", no_comparison_no_severity)
    print("Total pairs:", len(severity_pairwise))
    return comparison_no_severity, comparison_severity

def main():
    metadata_file = "data/hf-metadata.csv"
    metadata_file = os.path.join(os.environ['PE_PATH'], metadata_file)

    comparisons_file = "results/03302020/automatic-document-labels-severity.csv"
    comparisons_file = os.path.join(os.environ['PE_PATH'], comparisons_file)
    comparisons = pd.read_csv(comparisons_file, dtype={'study': 'str'})
    comparisons.set_index('study', inplace=True)

    severities = pd.concat([comparisons['severity']], axis=1)
    metadata = get_metadata(metadata_file, subset=severities)
    severity_pairwise = get_pairwise_severity(sort_by_date(metadata), severities)
    
    columns = ['subject', 'previous_study', 'next_study', 'previous_label', 'next_label', 'comparison_label']
    discrepancies = pd.DataFrame(compare(severity_pairwise, comparisons), columns=columns) 
    discrepancies.to_csv(os.path.join(os.environ['PE_PATH'], "results/03302020/comparison-severity-label-diff.csv"))

    comparison_no_severity, comparison_severity = collect_no_severity(severity_pairwise, comparisons)

    comparison_no_severity = pd.DataFrame(comparison_no_severity, columns=columns)
    comparison_severity = pd.DataFrame(comparison_severity, columns=columns)
    comparison_no_severity.to_csv(os.path.join(os.environ['PE_PATH'], "results/03302020/comparison-with-no-severity-labels.csv"))
    comparison_severity.to_csv(os.path.join(os.environ['PE_PATH'], "results/03302020/comparison-and-severity-labels.csv"))

if __name__ == "__main__":
    main()
