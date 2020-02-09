# 
# Identifies sentences relevant to the diagnosis of pulmonary edema and evaluates performance of algorithm 
# Requires PE_PATH environment variable to be set to root directory of codebase
#

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import argparse
import util.evaluate as evaluate
import math
import numpy as np 
import pandas as pd
import pprint
import re 

# Set up negex path
negex_path = os.path.join(os.environ['PE_PATH'], 'negex/negex.python/')
sys.path.insert(0, negex_path)
import negex

class CONSTANTS:

    noedema_file = os.path.join(os.environ['PE_PATH'], 'keywords', 'no_edema_regex.csv')
    keywords_file = os.path.join(os.environ['PE_PATH'], 'keywords', 'keywords_regex.csv')
    relatedrad_file = os.path.join(os.environ['PE_PATH'], 'keywords', 'related_rad_regex.csv')
    nochange_file = os.path.join(os.environ['PE_PATH'], 'keywords', 'no_change_regex.csv')

def is_positive(sentence, keywords, mode='last'):
    """
    sentence    sentence to check
    keywords    list of keywords to look for in sentence 
    mode        aggregation method when more than one keyword is present
                'first'     return flag for first keyword found
                'last'      return flag for last keyword found 
                'majority'  return T/F depending on relative number of keywords that are positive/negative
                'sum'       return T if at least one keyword is positive

    Returns a flag indicating the presence/absence of keywords in a sentence
        - True  1+ keywords present in sentence and assertion is positive
        - False 1+ keywords present in sentence and there is negation 
        - None  no keywords present 
    """
    flag = None
    num_pos, num_neg = 0, 0
    for key in keywords:
        # Open negex rules 
        rfile = open(negex_path+'negex_triggers.txt', 'r')
        irules = negex.sortRules(rfile.readlines())
        st = sentence.lower()

        # Search for keyword in sentence 
        match_obj = re.search(key, st)

        # If keyword found, check whether it is negated 
        if match_obj is not None:
            tagger = negex.negTagger(sentence = st, phrases = [st[match_obj.start():match_obj.end()]], rules = irules, negP = False)
            if tagger.getNegationFlag() == 'affirmed': 
                flag = True
                num_pos += 1 
            if tagger.getNegationFlag() == 'negated': 
                flag = False
                num_neg += 1

            # If mode is 'first', return immediately after first keyword is detected 
            if mode == 'first': return flag

            # If mode is 'sum', return immediately after first positive keyword is detected 
            if mode == 'sum' and flag is True: return flag 

    # If mode is 'majority', return T if more positive keywords than negative, else return F
    if mode == 'majority':
        if num_pos >= num_neg: return True 
        else: return False 

    return flag

def assign_keyword_label(sentences):
    """
    sentences           column of sentences to be labeled 

    Returns a column indicating the presence/absence of pulmonary edema based on keyword-matching
        - 1.0 pulmonary edema present
        - 0.0 pulmonary edema absent
        - nan no mention of pulmonary edema
    """ 
    keyword_label = pd.DataFrame(float('nan'), index=np.arange(sentences.shape[0]), columns=['keyword_label'])

    # List of keywords indicating no pulmonary edema
    noedema_keywords = pd.read_csv(CONSTANTS.noedema_file)['regex'].tolist()
    # List of keywords related to pulmonary edema 
    keywords = pd.read_csv(CONSTANTS.keywords_file)['regex'].tolist()
    
    for index, row in sentences.iterrows():
        sentence = row['sentence']
        flag = False 

        # First check if no pulmonary edema is explicitly mentioned
        for key in noedema_keywords:
            if re.search(key, sentence.lower()) is not None:
                flag = True 
                keyword_label['keyword_label'][index] = 0.0

        # Use more general keyword approach to assign mention label 
        if not flag:
            keyword_check = is_positive(row['sentence'], keywords)
            if keyword_check is True: keyword_label['keyword_label'][index] = 1.0
            elif keyword_check is False: keyword_label['keyword_label'][index] = 0.0

    return keyword_label 

def assign_related_rad(sentences):
    """
    sentences           column of sentences to be labeled 

    Returns a column indicating the presence/absence of radiologic features related to, but not definitive for, pulmonary edema
        - 1.0 related radiologic feature present
        - 0.0 related radiologic feature absent
        - nan no mention of related radiologic feature
    """
    # List of keywords for radiologic features related to pulmonary edema 
    keywords = pd.read_csv(CONSTANTS.relatedrad_file)['regex'].tolist()

    related_rad_label = pd.DataFrame(float('nan'), index=np.arange(sentences.shape[0]), columns=['related_rad_label'])
    for index, row in sentences.iterrows():
        # Check if there is any mention of a radiologic feature 
        rad_check = is_positive(row['sentence'], keywords, mode='sum')
        if rad_check is True: related_rad_label['related_rad_label'][index] = 1.0
        elif rad_check is False: related_rad_label['related_rad_label'][index] = 0.0

    return related_rad_label

def assign_other_finding(chexpert_sentences):
    """
    chexpert_sentences  output of CheXpert labeler (1, 0, nan) for 14 observations in chest radiographs 

    Returns a column indicating the presence/absence of other finding(s) that are not pulmonary edema 
        - 1.0 other finding(s) present
        - 0.0 other finding(s) absent
        - nan no mention of other findings
    """
    other_finding = pd.DataFrame(float('nan'), index=np.arange(chexpert_sentences.shape[0]), columns=['other_finding'])

    # Columns to ignore 
    ignore_labels = set(['Reports', 'Edema', 'Support Devices'])

    for index, row in chexpert_sentences.iterrows():
        for col in chexpert_sentences.columns:
            # Skip if column should be ignored or if cell value is empty, e.g. no mention 
            if col in ignore_labels or math.isnan(row[col]): 
                continue 

            # If no finding is positive, then assume no other findings are present 
            elif col == 'No Finding' and row[col] == 1.0:
                other_finding['other_finding'][index] = 0.0

            # Handle 'Lung Opacity' label, which could be indicative of pulmonary edema
            elif col == 'Lung Opacity':

                # Bilateral opacities are a related radiographic finding for pulmonary edema
                if 'bilateral' in row['Reports'] or 'both' in row['Reports']:
                    continue

                # Unilateral opacities are likely not relevant to pulmonary edema (indicative of other findings)
                elif 'right' in row['Reports'] and 'left' not in row['Reports']:
                    other_finding['other_finding'][index] = abs(row[col])
                elif 'left' in row['Reports'] and 'right' not in row['Reports']:
                    other_finding['other_finding'][index] = abs(row[col])

            # If not a special case, set other finding label equal to cell value                     
            elif math.isnan(other_finding['other_finding'][index]):
                other_finding['other_finding'][index] = abs(row[col])

            # If there is positive mention of at least one finding, then other finding label should be 1.0
            else:
                other_finding['other_finding'][index] = max(other_finding['other_finding'][index], abs(row[col]))

    return other_finding 

def get_final_label(labels):
    """
    labels  Pandas dataframe with the following columns
            - Sentences
            - CheXpert label for pulmonary edema
            - Keyword label from output of assign_keyword_label()
            - Related radiologic feature label from output of assign_related_rad()
            - Other finding label from output of assign_other_finding()

    Returns a column indicating whether sentence is relevant or not relevant to pulmonary edema (1, 0)
    """
    final_label = pd.DataFrame(0.0, index=np.arange(labels.shape[0]), columns=['relevant'])
    nochange_keywords = pd.read_csv(CONSTANTS.nochange_file)['regex'].tolist()

    for index, row in labels.iterrows():
        # If pulmonary edema is mentioned as present, then consider sentence relevant
        if row['chexpert_label'] == 1.0 or row['keyword_label'] == 1.0:
            final_label['relevant'][index] = 1.0

        # If pulmonary edema is mentioned as absent, then consider sentence relevant 
        elif row['chexpert_label'] == 0.0 or row['keyword_label'] == 0.0:
            final_label['relevant'][index] = 1.0

        # If there is a related radiographic feature and no mention of another finding, then consider sentence relevant
        elif not math.isnan(row['related_rad_label']) and math.isnan(row['other_finding']):
            final_label['relevant'][index] = 1.0

        # If sentence indicates no general change in condition, then consider it to be relevant
        for key in nochange_keywords:
            if re.search(key, row['sentence'].lower()) is not None:
                final_label['relevant'][index] = 1.0

    return final_label

def evaluate_labeler(true_labels_path, predicted_labels_path, output_path=None):
    true_labels = pd.read_csv(true_labels_path)
    predicted_labels = pd.read_csv(predicted_labels_path)

    result = evaluate.evaluate(true_labels['relevant'].values, predicted_labels['relevant'].values)

    if output_path is not None:
        result_df = pd.Series(result).to_frame()
        result_df.to_csv(output_path)
    
    pprint.pprint(result)        

def run_labeler(chexpert_label_path, true_labels_path):
    chexpert_sentences = pd.read_csv(chexpert_label_path)
    true_labels = pd.read_csv(true_labels_path)

    sentences = chexpert_sentences['Reports'].to_frame().rename(columns={'Reports': 'sentence'})
    chexpert_label = abs(chexpert_sentences.loc[:, 'Edema']).to_frame().rename(columns={'Edema': 'chexpert_label'})
    other_finding = assign_other_finding(chexpert_sentences)
    keyword_label = assign_keyword_label(sentences)
    related_rad_label = assign_related_rad(sentences)

    labels = pd.concat([sentences, chexpert_label, keyword_label, related_rad_label, other_finding], axis=1)

    final_labels = pd.concat([sentences, get_final_label(labels), chexpert_label, keyword_label, related_rad_label, other_finding], axis=1)
    return final_labels

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

    final_labels = run_labeler(chexpert_labels_path, true_labels_path)
    final_labels.to_csv(output_labels_path)

    evaluate_labeler(true_labels_path, output_labels_path)

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

if __name__ == "__main__":
    main_label()
    # main_evaluate()

    