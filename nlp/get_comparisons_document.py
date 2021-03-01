import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import argparse
import math 
import numpy as np 
import pandas as pd
import re

from util.document import collect_sentences

def get_magnitude(new_magnitude, current_magnitude):
    label_to_score = {'irrelevant': 0, 'mild': 1, 'moderate': 2, 'marked': 3}
    if label_to_score[new_magnitude] > label_to_score[current_magnitude]:
        return new_magnitude
    return current_magnitude

def aggregate_sentences(labels, start, end):
    """
    Returns document-level label aggregated from individual sentences as (comparison_label, magnitude_of_change)
    
    If all sentences agree on the comparison, then the magnitude of change is determined by the last sentence with a comparison label. 
    If there is disagreement between sentences, then 'inf' is returned. 
    If there is no comparison, then None is returned. 
    """
    flags = {'better': False, 'worse': False, 'same': False}
    magnitude = {'better': 'irrelevant', 'worse': 'irrelevant'}

    index = start
    while index < end:
        if labels['predicted'][index] == -1.0:
            flags['better'] = True
            magnitude['better'] = get_magnitude(labels['magnitude'][index], magnitude['better'])
        if labels['predicted'][index] == 1.0:
            flags['worse'] = True
            magnitude['worse'] = get_magnitude(labels['magnitude'][index], magnitude['worse'])
        if labels['predicted'][index] == 0.0:
            flags['same'] = True 

        index += 1

    if sum(flags.values()) > 1:
        return (float('inf'), 'irrelevant')

    if flags['worse']: return (1.0, magnitude['worse'])
    if flags['better']: return (-1.0, magnitude['better'])
    if flags['same']: return (0.0, 'irrelevant')

    return (None, 'irrelevant')

def resolve_disagreements(labels, start, end):
    """
    Resolves disagreements if sentences in a single document have different comparison labels. Considers sentences
    at the end of the document first. 
    """
    nochange_file = os.path.join(os.environ['PE_PATH'], 'keywords', 'no_change_regex.csv')
    nochange_keywords = pd.read_csv(nochange_file)['regex'].tolist()

    index = end - 1
    related_rad_flag = None
    general_no_change = None
    while index >= start:
        if not math.isnan(labels['chexpert_label'][index]) or not math.isnan(labels['keyword_label'][index]):
            if not math.isnan(labels['predicted'][index]):
                return (labels['predicted'][index], labels['magnitude'][index])
        elif not math.isnan(labels['related_rad_label'][index]) and related_rad_flag is None:
            related_rad_flag = (labels['predicted'][index], labels['magnitude'][index])
        else:
            for key in nochange_keywords:
                if re.search(key, labels['sentence'][index].lower()) is not None and general_no_change is None:
                    general_no_change = (labels['predicted'][index], labels['magnitude'][index])
        index -= 1

    if related_rad_flag is not None:
        return related_rad_flag

    return general_no_change

def get_comparison_labels(labels_path, output_path):
    document_labels = []

    labels = pd.read_csv(labels_path, index_col=0, dtype={'subject': 'str', 'study': 'str'})
    index = 0
    while index < labels.shape[0]:
        subject = labels['subject'][index]
        study = labels['study'][index]
        report, start, end = collect_sentences(labels, subject, study, start_index=index)
        document_label, magnitude = aggregate_sentences(labels, start, end)
        if document_label == float('inf'):
            document_label, magnitude = resolve_disagreements(labels, start, end)

        document_labels.append([subject, study, document_label, magnitude])
        index = end

    output = pd.DataFrame(document_labels, columns=['subject', 'study', 'comparison', 'magnitude'])
    output.to_csv(output_path)

def main():
    parser = argparse.ArgumentParser(description='Assign document-level comparison labels from sentence-level labels')
    parser.add_argument('comparison_labels_path', type=str, help='Path to file with comparison labels')
    parser.add_argument('output_path', type=str, help='Path to file where comparison labels are written')
    args = parser.parse_args()

    comparison_labels_path = os.path.join(os.environ['PE_PATH'], args.comparison_labels_path)
    output_path = os.path.join(os.environ['PE_PATH'], args.output_path)
    get_comparison_labels(comparison_labels_path, output_path)

if __name__ == "__main__":
    main()

