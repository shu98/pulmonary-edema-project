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
    start = match_obj.start()
    end = match_obj.end()

    while start >= 0 and sentence[start] != " ": start -= 1
    start = max(start, 0)

    while end < len(sentence) and sentence[end] != " ": end += 1
    end = min(end, len(sentence)-1)

    start_sent = sentence[:start]
    end_sent = sentence[end:] 

    start_split = start_sent.strip().split(" ")[-max(0, radius):]
    end_split = end_sent.strip().split(" ")[:max(0, radius)]

    to_search = "{} {} {}".format(" ".join(start_split), sentence[match_obj.start():match_obj.end()], " ".join(end_split))
    return to_search

def contains_word(sentence, keywords):
    for word in keywords:
        match_obj = re.search(word, sentence) 
        if match_obj is not None:
            return (sentence[match_obj.start():match_obj.end()], True) 

    return (None, False)

def get_comparison_simple(sentence, better, worse, same, radius=8):
    sentence = sentence.lower()
    flags = {'better': False, 'worse': False, 'same': False}
    keywords = {'better': set(), 'worse': set(), 'same': set()}
    
    pe_keywords = pd.read_csv(CONSTANTS.keywords_file)['regex'].tolist() + pd.read_csv(CONSTANTS.relatedrad_file)['regex'].tolist() 
    nochange_keywords = pd.read_csv(CONSTANTS.nochange_file)['regex'].tolist()

    for word in better:
        match_obj = re.search(word, sentence)
        if match_obj is not None:
            to_search = get_search_radius(sentence, match_obj, radius)
            word, does_contain = contains_word(to_search, pe_keywords)
            if does_contain:
                flags['better'] = True
                keywords['better'].add((sentence[match_obj.start():match_obj.end()], to_search))
    
    for word in worse: 
        match_obj = re.search(word, sentence)
        if match_obj is not None:
            to_search = get_search_radius(sentence, match_obj, radius)
            word, does_contain = contains_word(to_search, pe_keywords)
            if does_contain:
                flags['worse'] = True
                keywords['worse'].add((sentence[match_obj.start():match_obj.end()], to_search))
    
    for word in same:
        match_obj = re.search(word, sentence)
        if match_obj is not None:
            to_search = get_search_radius(sentence, match_obj, radius)
            word, does_contain = contains_word(to_search, pe_keywords)
            if does_contain:
                flags['same'] = True
                keywords['same'].add((sentence[match_obj.start():match_obj.end()], to_search))

            if not does_contain:
                word, does_contain = contains_word(sentence, nochange_keywords)
                if does_contain:
                    flags['same'] = True
                    keywords['same'].add((word, sentence))

    # Worse label takes precedent over others 
    if flags['worse']: return (1.0, keywords['worse'])
    elif flags['better']: return (-1.0, keywords['better'])
    elif flags['same']: return (0.0, keywords['same'])

    return (float('nan'), [])

def compare_sentence(sentence, better, worse, same, pe_keywords):
    label, words = get_comparison_simple(sentence, better, worse, same)
    neg_check = (is_positive(sentence, [w[0] for w in words]) == False)
    neg_check = neg_check or (is_positive(sentence, pe_keywords) == False and \
            not (math.isnan(label) or re.search("resolv(e|ing|ed)", sentence.lower())))

    keywords = "//".join(["/".join(word) for word in words])
    negation = 0.0
    comparison = float('nan')
    if neg_check and abs(label) == 1.0:
        comparison = 0.0
        negation = 1.0
    else:
        comparison = label

    return (comparison, negation, keywords)

def compare(relevant_sentences_path, true_labels=False):
    sentences = pd.read_csv(relevant_sentences_path)
    better = pd.read_csv(COMPARISON_KEYWORDS.better)['regex'].tolist()
    worse = pd.read_csv(COMPARISON_KEYWORDS.worse)['regex'].tolist()
    same = pd.read_csv(COMPARISON_KEYWORDS.same)['regex'].tolist()

    pe_keywords = pd.read_csv(CONSTANTS.keywords_file)['regex'].tolist() + pd.read_csv(CONSTANTS.relatedrad_file)['regex'].tolist()

    comparisons = pd.DataFrame(float('nan'), index=np.arange(sentences.shape[0]), columns=['predicted'])
    negation = pd.DataFrame(0.0, index=np.arange(sentences.shape[0]), columns=['negation'])
    keywords = pd.DataFrame("", index=np.arange(sentences.shape[0]), columns=['keywords'])

    for index, sent in sentences.iterrows():
        if sent['relevant'] == 1.0:
            comp, neg, kw = compare_sentence(sent['sentence'], better, worse, same, pe_keywords)
            comparisons['predicted'][index] = comp 
            negation['negation'][index] = neg 
            keywords['keywords'][index] = kw 

    if true_labels:
        comparisons = pd.concat([sentences['sentence'], sentences['subject'], sentences['study'], sentences['relevant'], sentences['ground_truth_relevant'], sentences['keyword_label'], sentences['chexpert_label'], \
                                sentences['related_rad_label'], sentences['other_finding'], sentences['comparison_label'], comparisons['predicted'], negation['negation'], keywords['keywords']], axis=1)
        comparisons = comparisons.rename(columns={"comparison_label": "ground_truth"})
        return comparisons
    else:
        comparisons = pd.concat([sentences['sentence'], sentences['subject'], sentences['study'], sentences['relevant'], sentences['keyword_label'], sentences['chexpert_label'], sentences['related_rad_label'], \
                                sentences['other_finding'], comparisons['predicted'], negation['negation'], keywords['keywords']], axis=1)
        return comparisons

def evaluate(comparisons):
    incorrect, incorrect_relevance = 0, 0
    correct = 0
    positive = 0
    negative = 0

    for index, sent in comparisons.iterrows():
        if sent['ground_truth_relevant'] != sent['relevant']:
            print(sent['ground_truth_relevant'], sent['relevant'], sent['predicted'], sent['ground_truth'], sent['sentence'])
            incorrect_relevance += 1 
        elif math.isnan(sent['ground_truth']) and math.isnan(sent['predicted']):
            correct += 1 
        elif sent['predicted'] == -sent['ground_truth']:
            correct += 1
        else:
            print(sent['ground_truth_relevant'], sent['relevant'], sent['predicted'], sent['ground_truth'], sent['sentence'])
            incorrect += 1

    print("Incorrect:", incorrect, "Incorrect relevance:", incorrect_relevance, "Correct:", correct)
    # print("Positive:", positive, "Negative:", negative)

def compare_main():
    parser = argparse.ArgumentParser(description='Get comparisons for pulmonary edema')
    parser.add_argument('relevant_sentences_path', type=str, help='Path to file with sentences to label pulmonary edema')
    parser.add_argument('output_path', type=str, help='Path to file where comparison labels are written')
    args = parser.parse_args()

    relevant_sentences_path = os.path.join(os.environ['PE_PATH'], args.relevant_sentences_path)
    output_path = os.path.join(os.environ['PE_PATH'], args.output_path)
    comparisons = compare(relevant_sentences_path, true_labels=True)
    comparisons.to_csv(output_path)
    evaluate(comparisons)

def predict_main():
    parser = argparse.ArgumentParser(description='Get comparisons for pulmonary edema')
    parser.add_argument('relevant_sentences_path', type=str, help='Path to file with sentences to label pulmonary edema')
    parser.add_argument('output_path', type=str, help='Path to file where comparison labels are written')
    args = parser.parse_args()

    relevant_sentences_path = os.path.join(os.environ['PE_PATH'], args.relevant_sentences_path)
    output_path = os.path.join(os.environ['PE_PATH'], args.output_path)
    comparisons = compare(relevant_sentences_path)
    comparisons.to_csv(output_path)

def test_script():
    relevant_sentences = "results/04112020/automatic-relevance-labels-small.csv"
    better = pd.read_csv(COMPARISON_KEYWORDS.better)['regex'].tolist()
    worse = pd.read_csv(COMPARISON_KEYWORDS.worse)['regex'].tolist()
    same = pd.read_csv(COMPARISON_KEYWORDS.same)['regex'].tolist()

    pe_keywords = pd.read_csv(CONSTANTS.keywords_file)['regex'].tolist() + pd.read_csv(CONSTANTS.relatedrad_file)['regex'].tolist()
    sentence = "Pulmonary edema pattern is again seen"
    print(compare_sentence(sentence, better, worse, same, pe_keywords))

if __name__ == "__main__":
    compare_main()
    # test_script()



