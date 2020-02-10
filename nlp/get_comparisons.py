import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import argparse
import math 
import numpy as np 
import pandas as pd
import re

from nlp.get_relevant_sentences import CONSTANTS
from util.negation import is_positive

class COMPARISON_KEYWORDS:

    better = os.path.join(os.environ['PE_PATH'], 'keywords/comparison', 'better.csv')
    worse = os.path.join(os.environ['PE_PATH'], 'keywords/comparison', 'worse.csv')
    same = os.path.join(os.environ['PE_PATH'], 'keywords/comparison', 'same.csv')

def get_search_radius(sentence, match_obj, radius):
    start = sentence[:match_obj.start()-1]
    end = sentence[match_obj.end()+1:]

    start_split = start.split(" ")
    start_split = start_split[-radius:]

    end_split = end.split(" ")
    end_split = end_split[:radius]

    to_search = "{} {} {}".format(" ".join(start_split), sentence[match_obj.start(), match_obj.end()], " ".join(end_split))
    return to_search

def get_comparison_simple(sentence, better, worse, same, radius=4):
    sentence = sentence.lower()
    flags = {'better': False, 'worse': False, 'same': False}
    for word in better:
        if re.search(word, sentence) is not None:
            flags['better'] = True
    for word in worse: 
        if re.search(word, sentence) is not None:
            flags['worse'] = True
    for word in same:
        if re.search(word, sentence) is not None:
            flags['same'] = True

    # Worse label takes precedent over others 
    if flags['worse']: return -1.0
    elif flags['better']: return 1.0
    elif flags['same']: return 0.0

    return float('nan')

def compare(relevant_sentences_path, output_path):
    sentences = pd.read_csv(relevant_sentences_path)

    better = pd.read_csv(COMPARISON_KEYWORDS.better)['regex'].tolist()
    worse = pd.read_csv(COMPARISON_KEYWORDS.worse)['regex'].tolist()
    same = pd.read_csv(COMPARISON_KEYWORDS.same)['regex'].tolist()

    all_comparisons = better + worse + same
    pe_keywords = pd.read_csv(CONSTANTS.keywords_file)['regex'].tolist() + pd.read_csv(CONSTANTS.relatedrad_file)['regex'].tolist()

    comparisons = pd.DataFrame(float('nan'), index=np.arange(sentences.shape[0]), columns=['direction'])
    for index, sent in sentences.iterrows():
        if sent['relevant'] == 1.0:
            label = get_comparison_simple(sent['sentence'], better, worse, same)
            neg_check = (is_positive(sent['sentence'], all_comparisons) == False)
            neg_check = neg_check or (is_positive(sent['sentence'], pe_keywords) == False and not math.isnan(label))

            if neg_check and abs(label) == 1.0:
                comparisons['direction'][index] = 0.0
            else:
                comparisons['direction'][index] = label

    comparisons = pd.concat([sentences['sentence'], sentences['relevant'], comparisons['direction'], sentences['expected comparison label']], axis=1)
    comparisons = comparisons.rename(columns={"direction": "actual", "expected comparison label": "ground_truth"})
    evaluate(comparisons)
    comparisons.to_csv(output_path)

def evaluate(comparisons):
    incorrect = 0
    correct = 0
    positive = 0
    negative = 0

    for index, sent in comparisons.iterrows():
        if sent['relevant'] == 1.0:
            if math.isnan(sent['ground_truth']) and math.isnan(sent['actual']):
                correct += 1 
            elif sent['actual'] != sent['ground_truth']:
                print(sent['actual'], sent['ground_truth'], sent['sentence'])
                incorrect += 1
            else:
                correct += 1

            if math.isnan(sent['ground_truth']): negative += 1
            else: positive += 1

    print("Incorrect:", incorrect, "Correct:", correct)
    print("Positive:", positive, "Negative:", negative)

def compare_main():
    parser = argparse.ArgumentParser(description='Get comparisons for pulmonary edema')
    parser.add_argument('relevant_sentences_path', type=str, help='Path to file with sentences to label pulmonary edema')
    parser.add_argument('output_path', type=str, help='Path to file where comparison labels are written')
    args = parser.parse_args()

    relevant_sentences_path = os.path.join(os.environ['PE_PATH'], args.relevant_sentences_path)
    output_path = os.path.join(os.environ['PE_PATH'], args.output_path)
    compare(relevant_sentences_path, output_path)

if __name__ == "__main__":
    compare_main()



