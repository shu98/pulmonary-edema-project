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

def parse_txt(filename):
    labels = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.split(",")
            study = line[0].strip()
            logits = line[1].strip()[1:-1].strip().split()
            for index, value in enumerate(logits):
                logits[index] = float(value)
            label = np.argmax(softmax(logits))
            labels.append([study, label])

    labels_df = pd.DataFrame(labels, columns=['study', 'severity'])
    labels_df = labels_df.drop_duplicates(subset='study', keep='first')
    labels_df.set_index('study', inplace=True)

    return labels_df

def compare(severity_pairwise, comparison_labels):
    discrepancies = []
    does_not_exist = 0
    for subject, study1, study2, label1, label2 in severity_pairwise:
        if math.isnan(comparison_labels['comparison'][study2]):
            continue 

        if math.isnan(label1) or math.isnan(label2):
            does_not_exist += 1
            continue

        if np.sign(label2 - label1) != comparison_labels['comparison'][study2]:
            discrepancies.append([subject, study1, study2, label1, label2, comparison_labels['comparison'][study2]])

    print("Total pairs with no severity labels:", does_not_exist)
    return discrepancies

def get_pairwise_severity(series, severity_labels):
    """
    Sort radiology reports by date for a single patient
    
    Inputs 
    report_list     list of studies for a single patient 

    Returns 
    list of tuples of study reports (subject, study1, study2, label1, label2), where the date for study1 is prior to study2
    """
    severity_pairiwse = []
    for subject, reports in series.items():
        to_write = sorted(reports, key=lambda x: x[1])
        # Only keep subjects with more than one study 
        if len(to_write) > 0:
            study1, date1 = to_write[0]
            index = 1
            while index < len(to_write):
                study2, date2 = to_write[index][0], to_write[index][1]
                # Check that two consecutive reports in the queue are chronologically consecutive before adding pair to final list
                # This check exists because all studies for patient are put into the same queue, even if the patient has multiple
                # radiograph series (e.g. taken during separate hospital visits)
                if abs(date1 - date2) <= timedelta(days=2):
                    severity_pairiwse.append((subject, study1, study2, severity_labels['severity'][study1], severity_labels['severity'][study2]))
                
                study1, date1 = to_write[index]
                index += 1

    return severity_pairiwse

def main():
    ray_file = "data/comparison-data/ray_results_images_preds.txt"
    ray_file = os.path.join(os.environ['PE_PATH'], ray_file)
    severities = parse_txt(ray_file)

    metadata_file = "data/hf-metadata.csv"
    metadata_file = os.path.join(os.environ['PE_PATH'], metadata_file)

    comparisons_file = "results/04142020/comparisons-document-all.csv"
    comparisons_file = os.path.join(os.environ['PE_PATH'], comparisons_file)
    comparisons = pd.read_csv(comparisons_file, dtype={'study': 'str'})
    comparisons.set_index('study', inplace=True)

    severities = severities.loc[comparisons.index, :]
    metadata = get_metadata(metadata_file, subset=severities)
    severity_pairwise = get_pairwise_severity(sort_by_date(metadata), severities)
    
    discrepancies = pd.DataFrame(compare(severity_pairwise, comparisons), columns=['subject', 'previous_study', 'next_study', 'previous_label', 'next_label', 'comparison_label'])
    # discrepancies.to_csv(os.path.join(os.environ['PE_PATH'], "results/comparison-ray-diff.csv"))

if __name__ == "__main__":
    main()
