import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from dataset.organize_reports import sort_by_date, get_metadata
import math 
import pandas as pd 
from dataset.organize_reports import sort_by_date, get_metadata
from scripts.ranking_initial import get_full_series, get_pairwise_severity 

def count_same_label(single_series):
    total_same = 0
    total_nan = 0
    total_same_with_comparison = 0
    total_nan_with_comparison = 0
    for i, s in enumerate(single_series):
        subject, study1, study2, label1, label2, comparison = s
        if math.isnan(label1) or math.isnan(label2):
            total_nan += 1
            if not math.isnan(comparison):
                total_nan_with_comparison += 1
        elif label1 == label2:
            total_same += 1
            if not math.isnan(comparison):
                total_same_with_comparison += 1 

    return total_same, total_same_with_comparison, total_nan, total_nan_with_comparison

def count_same_label_all(series):
    total_same = 0
    total_nan = 0
    total_same_with_comparison = 0
    total_nan_with_comparison = 0

    for subject in series:
        results = count_same_label(subject)
        total_same += results[0]
        total_same_with_comparison += results[1]
        total_nan += results[2]
        total_nan_with_comparison += results[3]

    return total_same, total_same_with_comparison, total_nan, total_nan_with_comparison

def main():
    metadata_file = "data/hf-metadata.csv"
    metadata_file = os.path.join(os.environ['PE_PATH'], metadata_file)

    severity_file = "results/automatic-document-labels.csv"
    severity_file = os.path.join(os.environ['PE_PATH'], severity_file)
    severities = pd.read_csv(severity_file, dtype={'study': 'str'})
    severities.set_index('study', inplace=True)

    metadata = get_metadata(metadata_file, severities)
    pairwise_severity = get_pairwise_severity(sort_by_date(metadata), severities)

    all_series = get_full_series(pairwise_severity, severities)
    total_same, total_same_with_comparison, total_nan, total_nan_with_comparison = count_same_label_all(all_series)
    
    print("Total same with comparison:", total_same_with_comparison)
    print("Total same:", total_same)
    print("Total nan with comparison:", total_nan_with_comparison)
    print("Total nan:", total_nan)
    print("Total with severity label:", severities.count()['severity'])

    comparison_file = "results/03182020/comparisons-labeled-from-automatic-all.csv"
    comparison_file = os.path.join(os.environ['PE_PATH'], comparison_file)
    comparisons = pd.read_csv(comparison_file, dtype={'study': 'str'})
    comparisons.set_index('study', inplace=True)
    print("Total reports with pairwise comparison:", comparisons.count()['predicted'])

if __name__ == "__main__":
    main()

