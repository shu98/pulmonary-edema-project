# 
# Identifies sentences relevant to the diagnosis of pulmonary edema and evaluates performance of algorithm 
# Requires PE_PATH environment variable to be set to root directory of codebase
#

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import argparse
import math
import numpy as np 
import pandas as pd
import pprint
import re 

from util.evaluate import evaluate
from util.negation import is_positive

class CONSTANTS:

    noedema_file = os.path.join(os.environ['PE_PATH'], 'keywords', 'no_edema_regex.csv')
    keywords_file = os.path.join(os.environ['PE_PATH'], 'keywords', 'keywords_regex.csv')
    relatedrad_file = os.path.join(os.environ['PE_PATH'], 'keywords', 'related_rad_regex.csv')
    nochange_file = os.path.join(os.environ['PE_PATH'], 'keywords', 'no_change_regex.csv')

def assign_keyword_label(sentence):
    """
    sentence          sentence to be labeled 

    Returns a value indicating the presence/absence of pulmonary edema based on keyword-matching
        - 1.0 pulmonary edema present
        - 0.0 pulmonary edema absent
        - nan no mention of pulmonary edema
    """ 
    # List of keywords indicating no pulmonary edema
    noedema_keywords = pd.read_csv(CONSTANTS.noedema_file)['regex'].tolist()
    # List of keywords related to pulmonary edema 
    keywords = pd.read_csv(CONSTANTS.keywords_file)['regex'].tolist()
    
    flag = False 
    keyword_label = float('nan')
    # First check if no pulmonary edema is explicitly mentioned 
    for key in noedema_keywords:
        if re.search(key, sentence.lower()) is not None:
            flag = True 
            keyword_label = 0.0

    # Use more general keyword approach to assign mention label 
    if not flag:
        keyword_check = is_positive(sentence, keywords)
        if keyword_check is True: keyword_label = 1.0
        elif keyword_check is False: keyword_label = 0.0

    return keyword_label 

def assign_related_rad(sentence):
    """
    sentence           sentence to be labeled 

    Returns a value indicating the presence/absence of radiologic features related to, but not definitive for, pulmonary edema
        - 1.0 related radiologic feature present
        - 0.0 related radiologic feature absent
        - nan no mention of related radiologic feature
    """
    # List of keywords for radiologic features related to pulmonary edema 
    keywords = pd.read_csv(CONSTANTS.relatedrad_file)['regex'].tolist()

    related_rad_label = float('nan')
    # Check if there is any mention of a radiologic feature 
    rad_check = is_positive(sentence, keywords, mode='sum')
    if rad_check is True: related_rad_label = 1.0
    elif rad_check is False: related_rad_label = 0.0

    return related_rad_label

def assign_other_finding(chexpert_row):
    """
    chexpert_row  output of CheXpert labeler (1, 0, nan) for 14 observations in chest radiographs, represented as a Series 

    Returns a value indicating the presence/absence of other finding(s) that are not pulmonary edema 
        - 1.0 other finding(s) present
        - 0.0 other finding(s) absent
        - nan no mention of other findings
    """
    other_finding = float('nan')
    # Columns to ignore 
    ignore_labels = set(['Reports', 'Edema', 'Support Devices', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Pneumothorax'])
    for col in chexpert_row.index:
        # Skip if column should be ignored or if cell value is empty, e.g. no mention 
        if col in ignore_labels or math.isnan(chexpert_row[col]): 
            continue 

        # If no finding is positive, then assume no other findings are present 
        elif col == 'No Finding' and chexpert_row[col] == 1.0:
            for other_col in chexpert_row.index:
                if other_col not in ignore_labels and chexpert_row[other_col] == 0.0:
                    other_finding = 0.0 

        # Handle 'Lung Opacity' label, which could be indicative of pulmonary edema depending on context
        elif col == 'Lung Opacity':
            # Bilateral opacities are a related radiographic finding for pulmonary edema
            if 'bilateral' in chexpert_row['Reports'] or 'both' in chexpert_row['Reports']:
                continue
            # Unilateral opacities are likely not relevant to pulmonary edema (indicative of other findings)
            elif 'right' in chexpert_row['Reports'] and 'left' not in chexpert_row['Reports']:
                other_finding = abs(chexpert_row[col])
            elif 'left' in chexpert_row['Reports'] and 'right' not in chexpert_row['Reports']:
                other_finding = abs(chexpert_row[col])

        # If not a special case, set other finding label equal to cell value                     
        elif math.isnan(other_finding):
            other_finding = abs(chexpert_row[col])

        # If there is positive mention of at least one finding, then other finding label should be 1.0
        else:
            other_finding = max(other_finding, abs(chexpert_row[col]))

    return other_finding 

def get_other_finding_mention(chexpert_row):
    ignore_labels = set(['Reports', 'Edema', 'No Finding'])
    for col in chexpert_row.index:
        # Skip if column should be ignored or if cell value is empty, e.g. no mention 
        if col not in ignore_labels and not math.isnan(chexpert_row[col]): 
            return True 

    return False  

def get_final_label(sentence, chexpert_label, keyword_label, related_rad_label, other_finding, chexpert_row):
    """
    sentence            sentence to label
    chexpert_label      CheXpert label for pulmonary edema
    keyword_label       Keyword label from output of assign_keyword_label()
    related_rad_label   Related radiologic feature label from output of assign_related_rad()
    other_finding       Other finding label from output of assign_other_finding()
    chexpert_row        output of CheXpert labeler (1, 0, nan) for 14 observations in chest radiographs, represented as a Series 

    Returns a value indicating whether sentence is relevant or not relevant to pulmonary edema (1, 0)
    """
    final_label = 0.0
    nochange_keywords = pd.read_csv(CONSTANTS.nochange_file)['regex'].tolist()

    # If pulmonary edema is mentioned as present, then consider sentence relevant
    if chexpert_label == 1.0 or keyword_label == 1.0:
        final_label = 1.0

    # If pulmonary edema is mentioned as absent, then consider sentence relevant 
    elif chexpert_label == 0.0 or keyword_label == 0.0:
        final_label = 1.0

    # If there is a related radiographic feature and no mention of another finding, then consider sentence relevant
    elif not math.isnan(related_rad_label) and math.isnan(other_finding):
        final_label = 1.0

    # If sentence indicates no general change in condition, then consider it to be relevant
    if final_label == 0.0:
        for key in nochange_keywords:
            if re.search(key, sentence.lower()) is not None:
                # Exclude phrases like "no change in cardiomegaly" or "stable atelectasis"
                if not get_other_finding_mention(chexpert_row): 
                    final_label = 1.0
                    break 
        
    return final_label

def get_all_relevance_data(chexpert_row, metadata, true_labels):
    sentence = chexpert_row['Reports']
    chexpert_label = abs(chexpert_row['Edema'])
    chexpert_label_unprocessed = chexpert_row['Edema']
    other_finding = assign_other_finding(chexpert_row)
    keyword_label = assign_keyword_label(sentence)
    related_rad_label = assign_related_rad(sentence)

    final_label = get_final_label(sentence, chexpert_label, keyword_label, related_rad_label, other_finding, chexpert_row)

    if true_labels:
        return [sentence, metadata['subject'], metadata['study'], final_label, metadata['relevant'], chexpert_label, chexpert_label_unprocessed, \
                keyword_label, related_rad_label, other_finding, metadata['comparison'], metadata['comparison label']]

    else:
        return [sentence, metadata['subject'], metadata['study'], final_label, chexpert_label, chexpert_label_unprocessed, keyword_label, \
                related_rad_label, other_finding]

def run_labeler(chexpert_label_path, metadata_labels_path, true_labels=False):
    chexpert_sentences = pd.read_csv(chexpert_label_path)
    metadata = pd.read_csv(metadata_labels_path, dtype={'subject': 'str', 'study': 'str'})
    
    all_data = []
    for index, row in chexpert_sentences.iterrows():
        processed_row = get_all_relevance_data(row, metadata.iloc[index, :], true_labels=true_labels)
        all_data.append(processed_row)

    columns = []
    if true_labels:
        columns = ['sentence', 'subject', 'study', 'relevance', 'ground_truth_relevance', 'chexpert_label', 'chexpert_unprocessed', \
                    'keyword_label', 'related_rad_label', 'other_finding', 'comparison_finding', 'comparison_label']
    else:
        columns = ['sentence', 'subject', 'study', 'relevance', 'chexpert_label', 'chexpert_unprocessed', 'keyword_label', 'related_rad_label', 'other_finding']

    df = pd.DataFrame(all_data, columns=columns)
    return df 

def main_label():
    """
    Run and evaluate labeler for pulmonary edema relevance. Also saves output labels of automatic labeler

    Requires as inputs
        1. CSV file with results of CheXpert labeler
        2. Filename to write the results of this automatic labeler
        3. CSV file with true labels (1, 0) 
    """
    parser = argparse.ArgumentParser(description='Get sentences relevant to pulmonary edema')

    # Relative paths to PE_PATH
    parser.add_argument('chexpert_labels_path', type=str, help='Path to file with chexpert-labeled sentences')
    parser.add_argument('output_labels_path', type=str, help='Path to file to write output labels')
    parser.add_argument('true_labels_path', type=str, help='Path to file with ground-truth relevance labels')
    args = parser.parse_args()

    chexpert_labels_path = os.path.join(os.environ['PE_PATH'], args.chexpert_labels_path)
    output_labels_path = os.path.join(os.environ['PE_PATH'], args.output_labels_path)
    true_labels_path = os.path.join(os.environ['PE_PATH'], args.true_labels_path)

    final_labels = run_labeler(chexpert_labels_path, true_labels_path, true_labels=True)
    final_labels.to_csv(output_labels_path)

def main_evaluate():
    """
    Evaluate results of automatic labeler that identifies whether a sentence is related to pulmonary edema diagnosis. 
    
    Requires as inputs
        1. CSV file with true labels (1, 0) 
        2. CSV file with predicted labels (1, 0)
        3. Filename to write evaluation results 
    """
    parser = argparse.ArgumentParser(description='Get sentences relevant to pulmonary edema')

    # Relative paths to PE_PATH
    parser.add_argument('true_labels_path', type=str, help='Path to file with ground-truth relevance labels')
    parser.add_argument('predicted_labels_path', type=str, help='Path to file with predicted relevance labels')
    parser.add_argument('output_path', type=str, help='Path to file to write evaluation results')
    args = parser.parse_args()

    true_labels_path = os.path.join(os.environ['PE_PATH'], args.true_labels_path)  
    predicted_labels_path = os.path.join(os.environ['PE_PATH'], args.predicted_labels_path)
    output_path = os.path.join(os.environ['PE_PATH'], args.output_path)

    evaluate_labeler(true_labels_path, predicted_labels_path, output_path=output_path)

def main_predict():
    """
    Run labeler for pulmonary edema relevance. Also saves output labels of automatic labeler

    Requires as inputs
        1. CSV file with results of CheXpert labeler
        2. Filename to write the results of this automatic labeler
        3. CSV file with metadata
    """
    parser = argparse.ArgumentParser(description='Get sentences relevant to pulmonary edema')

    # Relative paths to PE_PATH
    parser.add_argument('chexpert_labels_path', type=str, help='Path to file with chexpert-labeled sentences')
    parser.add_argument('output_labels_path', type=str, help='Path to file to write output labels')
    parser.add_argument('metadata_labels_path', type=str, help='Path to file with subject and study labels')
    args = parser.parse_args()

    chexpert_labels_path = os.path.join(os.environ['PE_PATH'], args.chexpert_labels_path)
    output_labels_path = os.path.join(os.environ['PE_PATH'], args.output_labels_path)
    metadata_labels_path = os.path.join(os.environ['PE_PATH'], args.metadata_labels_path)

    final_labels = run_labeler(chexpert_labels_path, metadata_labels_path)
    final_labels.to_csv(output_labels_path)

def test_script():
    chexpert = pd.read_csv("data/dataset-small/chexpert-labels-small.csv", dtype={'subject': 'str', 'study': 'str'})
    metadata = pd.read_csv("data/dataset-small/sentences-split-small.csv", dtype={'subject': 'str', 'study': 'str'})
    print(get_all_relevance_data(chexpert.iloc[119, :], metadata.iloc[119, :], False))

if __name__ == "__main__":
    main_label()
    # test_script()

