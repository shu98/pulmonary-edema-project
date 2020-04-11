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

def read_file(filename): 
    file = os.path.join(os.environ['PE_PATH'], filename)
    df = pd.read_csv(file, dtype={"subject": "str", "study":"str"}, index_col=0)
    return df 

def collect_no_severity(severity_pairwise, comparison_labels):
    comparison_no_severity, comparison_severity, severity_no_comparison = [], [], []
    no_comparison_no_severity = 0
    reports_with_severity = set()

    for subject, study1, study2, label1, label2 in severity_pairwise:
        if not math.isnan(label1):
            reports_with_severity.add(study1)
        if not math.isnan(label2):
            reports_with_severity.add(study2)

        if math.isnan(comparison_labels['comparison'][study2]):
            # No comparison, no severity 
            if (math.isnan(label1) or math.isnan(label2)): 
                no_comparison_no_severity += 1
            # No comparison, yes severity
            else: 
                severity_no_comparison.append([subject, study1, study2, label1, label2, comparison_labels['comparison'][study2]])
            continue 

        if (math.isnan(label1) or math.isnan(label2)):
            # Yes comparison, no severity
            if not math.isnan(comparison_labels['comparison'][study2]):
                comparison_no_severity.append([subject, study1, study2, label1, label2, comparison_labels['comparison'][study2]])
            # No comparison, no severity 
            else: 
                no_comparison_no_severity += 1

        # Yes comparison, yes severity
        if not math.isnan(label1) and not math.isnan(label2) and not math.isnan(comparison_labels['comparison'][study2]):
            comparison_severity.append([subject, study1, study2, label1, label2, comparison_labels['comparison'][study2]])

    print("Total pairs with no comparison or severity:", no_comparison_no_severity)
    print("Total studies with severity label:", len(reports_with_severity))
    print("Total pairs:", len(severity_pairwise))
    return comparison_no_severity, comparison_severity, severity_no_comparison

def evaluate_manual_comparisons():
    manual_labels = read_file("results/analysis/comparison-severity-label-diff.csv")
    comparison_need_to_check = pd.DataFrame(columns=manual_labels.columns)
    incorrect = 0
    for index, row in manual_labels.iterrows():
        if row["comparison_label"] != row["manual_label"]:
            comparison_need_to_check = comparison_need_to_check.append(row, ignore_index=True)
            incorrect += 1 

        elif row["need_to_check"] == 1.0:
            comparison_need_to_check = comparison_need_to_check.append(row, ignore_index=True)

    result_file = "results/analysis/comparison-severity-label-to-check.csv"
    comparison_need_to_check.to_csv(os.path.join(os.environ['PE_PATH'], result_file))
    print("Total incorrect comparison:", incorrect)

def get_same_severity():
    differences = read_file("results/analysis/comparison-and-severity-labels.csv")
    same_severity = pd.DataFrame(columns=differences.columns)
    total_diff = 0
    for index, row in differences.iterrows():
        if row['previous_label'] == row['next_label']:
            same_severity = same_severity.append(row, ignore_index=True)
            if row['comparison_label'] != 0:
                total_diff += 1 

    print("Total discrepancies:", total_diff)
    print("Total with same severity labels:", same_severity.shape[0])
    same_severity.to_csv(os.path.join(os.environ['PE_PATH'], "results/analysis/same-severity-labels.csv"))

def get_class_distribution_errors():
    manual_labels = read_file("results/analysis/comparison-severity-label-diff.csv")
    differences = read_file("results/analysis/comparison-and-severity-labels.csv")
    
    results = {'total_better': 0, 'total_worse': 0, 'total_same': 0,
                'better_correct': 0,'better_as_worse': 0, 'better_as_same': 0, 
                'worse_correct': 0, 'worse_as_better': 0, 'worse_as_same': 0,
                'same_correct': 0, 'same_as_better': 0, 'same_as_worse': 0}

    for index, row in differences.iterrows():
        severity_label_comparison = -np.sign(row['next_label'] - row['previous_label'])

        if row['comparison_label'] == 1.0:
            results['total_better'] += 1 

            if severity_label_comparison == 1:
                results['better_correct'] += 1 
            elif severity_label_comparison == 0:
                results['better_as_same'] += 1 
            elif severity_label_comparison == -1:
                results['better_as_worse'] += 1

        elif row['comparison_label'] == -1.0:
            results['total_worse'] += 1 

            if severity_label_comparison == -1:
                results['worse_correct'] += 1 
            elif severity_label_comparison == 0:
                results['worse_as_same'] += 1 
            elif severity_label_comparison == 1:
                results['worse_as_better'] += 1

        elif row['comparison_label'] == 0.0:
            results['total_same'] += 1 

            if severity_label_comparison == 0:
                results['same_correct'] += 1 
            elif severity_label_comparison == -1:
                results['same_as_worse'] += 1 
            elif severity_label_comparison == 1:
                results['same_as_better'] += 1

    pprint(results)

def main_collect_data():
    metadata_file = "data/hf-metadata.csv"
    metadata_file = os.path.join(os.environ['PE_PATH'], metadata_file)

    comparisons = read_file("results/03302020/automatic-document-labels-severity.csv")
    comparisons.set_index('study', inplace=True)

    severities = pd.concat([comparisons['severity']], axis=1)
    metadata = get_metadata(metadata_file, subset=severities)
    severity_pairwise = get_pairwise_severity(sort_by_date(metadata), severities)
    
    columns = ['subject', 'previous_study', 'next_study', 'previous_label', 'next_label', 'comparison_label']
    discrepancies = pd.DataFrame(compare(severity_pairwise, comparisons), columns=columns) 
    discrepancies.to_csv(os.path.join(os.environ['PE_PATH'], "results/03302020/comparison-severity-label-diff.csv"))

    comparison_no_severity, comparison_severity, severity_no_comparison = collect_no_severity(severity_pairwise, comparisons)

    comparison_no_severity = pd.DataFrame(comparison_no_severity, columns=columns)
    comparison_severity = pd.DataFrame(comparison_severity, columns=columns)
    severity_no_comparison = pd.DataFrame(severity_no_comparison, columns=columns)
    # comparison_no_severity.to_csv(os.path.join(os.environ['PE_PATH'], "results/analysis/comparison-with-no-severity-labels.csv"))
    # comparison_severity.to_csv(os.path.join(os.environ['PE_PATH'], "results/analysis/comparison-and-severity-labels.csv"))
    # severity_no_comparison.to_csv(os.path.join(os.environ['PE_PATH'], "results/analysis/severity-with-no-comparison.csv"))

if __name__ == "__main__":
    # main_collect_data()
    # evaluate_manual_comparisons()
    # get_class_distribution_errors()
    get_same_severity()


