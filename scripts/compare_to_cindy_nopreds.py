import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from datetime import datetime, timedelta 
import math 
import numpy as np 
import pandas as pd 
from pprint import pprint
from dataset.organize_reports import sort_by_date, get_metadata

def process_pairwise_cindy(data):
    pairs = []
    pairs_index = {}
    pairs_index_backward = {}
    for index, row in data.iterrows():
        pairs.append({'subject': row['sub_id'],
                    'study1': row['study_id1'],
                    'study2': row['study_id2'],
                    'label1': row['edema1'],
                    'label2': row['edema2'],
                    'predicted_comp': np.sign(row['edema_diff'])})
        pairs_index["{}{}".format(row['study_id1'], row['study_id2'])] = index
        pairs_index_backward["{}{}".format(row['study_id2'], row['study_id1'])] = index

    return pairs, pairs_index, pairs_index_backward

def process_pairwise(data, comparison_labels):
    pairs = []
    pairs_index = {}
    pairs_index_backward = {}
    for index, row in enumerate(data):
        pairs.append({'subject': row[0],
                    'study1': row[1],
                    'study2': row[2],
                    'predicted_comp': comparison_labels['comparison'][row[2]]})
        pairs_index["{}{}".format(row[1], row[2])] = index
        pairs_index_backward["{}{}".format(row[2], row[1])] = index

    return pairs, pairs_index, pairs_index_backward

def get_pairwise_reports(series):
    """
    Sort radiology reports by date for a single patient
    
    Inputs 
    report_list     list of studies for a single patient 

    Returns 
    list of tuples of study reports (subject, study1, study2, label1, label2), where the date for study1 is prior to study2
    """
    pairiwse = []
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
                    pairiwse.append((subject, study1, study2))
                
                study1, date1 = to_write[index]
                index += 1

    return pairiwse

def compare(pairs, pairs_cindy, pairs_index, pairs_index_cindy, pairs_index_backward, pairs_index_backward_cindy):
    overlap1 = set(pairs_index.keys()).intersection(set(pairs_index_cindy.keys()))
    overlap2 = set(pairs_index_backward.keys()).intersection(set(pairs_index_backward_cindy.keys()))
    overlap3 = set(pairs_index.keys()).intersection(set(pairs_index_backward_cindy.keys()))
    overlap4 = set(pairs_index.keys()).intersection(set(pairs_index_backward_cindy.keys()))
    overlap = overlap1.union(overlap2).union(overlap3).union(overlap4)

    total = 0 

    discrepancies = []
    no_comparison_label = []
    agreements = [] 
    all_pairs = []
    visited = set()

    for pair in overlap:
        is_backward = False
        is_backward_cindy = False
        pair_data = None
        pair_data_cindy = None 

        if pair in pairs_index:
            comparison_index = pairs_index[pair]
            pair_data = pairs[comparison_index]
        else:
            comparison_index = pairs_index_backward[pair]
            pair_data = pairs[comparison_index]
            pair_data['study1'], pair_data['study2'] = pair_data['study2'], pair_data['study1']
            pair_data['predicted_comp'] = -pair_data['predicted_comp']
            is_backward = True 

        if pair in pairs_index_cindy:
            comparison_index_cindy = pairs_index_cindy[pair]
            pair_data_cindy = pairs_cindy[comparison_index_cindy]
        else:
            comparison_index_cindy = pairs_index_backward_cindy[pair]
            pair_data_cindy = pairs_cindy[comparison_index_cindy]
            pair_data_cindy['study1'], pair_data_cindy['study2'] = pair_data_cindy['study2'], pair_data_cindy['study1']
            pair_data_cindy['label1'], pair_data_cindy['label2'] = pair_data_cindy['label2'], pair_data_cindy['label1']
            pair_data_cindy['predicted_comp'] = -pair_data_cindy['predicted_comp']
            is_backward_cindy = True 


        if str(pair_data['subject']) != str(pair_data_cindy['subject']) \
            or (str(pair_data['study1']) != str(pair_data_cindy['study1']))\
            or (str(pair_data['study2']) != str(pair_data_cindy['study2'])):

            print(pair_data['subject'], pair_data_cindy['subject'], pair_data['subject'] == pair_data_cindy['subject'])
            print(pair_data['study1'], pair_data_cindy['study1'], pair_data['study1'] == pair_data_cindy['study1'])
            print(pair_data['study2'], pair_data_cindy['study2'], pair_data['study2'] == pair_data_cindy['study2'])
            raise Exception("Mismatch in study ids. This should not happen")

        # if str(pair_data['subject']) != str(pair_data_cindy['subject']) \
        #     or (str(pair_data['study1']) != str(pair_data_cindy['study1']) and str(pair_data['study1']) != str(pair_data_cindy['study2'])) \
        #     or (str(pair_data['study2']) != str(pair_data_cindy['study2']) and str(pair_data['study2']) != str(pair_data_cindy['study1'])):

        #     print(pair_data['subject'], pair_data_cindy['subject'], pair_data['subject'] == pair_data_cindy['subject'])
        #     print(pair_data['study1'], pair_data_cindy['study1'], pair_data['study1'] == pair_data_cindy['study1'])
        #     print(pair_data['study2'], pair_data_cindy['study2'], pair_data['study2'] == pair_data_cindy['study2'])
        #     raise Exception("Mismatch in study ids. This should not happen")

        if (pair_data['study1'], pair_data['study2']) in visited or (pair_data['study2'], pair_data['study1']) in visited:
            continue 
        else:
            total += 1 
            visited.add((pair_data['study1'], pair_data['study2']))

        # comparison_label = -pair_data['predicted_comp'] if is_backward else pair_data['predicted_comp']
        # comparison_label_cindy = -pair_data_cindy['predicted_comp'] if is_backward_cindy else pair_data_cindy['predicted_comp']
        # true_label = -pair_data_cindy['true_comp'] if is_backward_cindy else pair_data_cindy['true_comp']

        # to_append = [
        #                 pair_data['subject'],
        #                 pair_data['study1'],
        #                 pair_data['study2'],
        #                 pair_data_cindy['label1'] if pair_data['study1'] == pair_data_cindy['study1'] else pair_data_cindy['label2'],
        #                 pair_data_cindy['label2'] if pair_data['study2'] == pair_data_cindy['study2'] else pair_data_cindy['label1'],
        #                 comparison_label,
        #                 comparison_label_cindy,
        #                 true_label
        #             ]

        comparison_label = pair_data['predicted_comp']
        comparison_label_cindy = pair_data_cindy['predicted_comp']

        to_append = [
                        pair_data['subject'],
                        pair_data['study1'],
                        pair_data['study2'],
                        pair_data_cindy['label1'] if pair_data['study1'] == pair_data_cindy['study1'] else pair_data_cindy['label2'],
                        pair_data_cindy['label2'] if pair_data['study2'] == pair_data_cindy['study2'] else pair_data_cindy['label1'],
                        comparison_label,
                        comparison_label_cindy if pair_data['study1'] == pair_data_cindy['study1'] else -comparison_label_cindy,
                    ]

        if math.isnan(comparison_label):
            no_comparison_label.append(to_append)
            continue 

        if comparison_label != comparison_label_cindy:
            discrepancies.append(to_append)

        elif comparison_label == comparison_label_cindy:
            agreements.append(to_append)

    # print("Total pairs with no comparison label:", no_comparison_label)
    print("Total pairs:", total)
    return discrepancies, no_comparison_label, agreements

def main():
    cindy_data = pd.read_csv("data/comparison-data/cindy_mimic_labV2_pairs_test.csv", \
        dtype={'study_id1': 'str', 'study_id2': 'str', 'sub_id': 'str', 'y_pred': 'float'})
    
    all_pairs_cindy, pairs_index_cindy, pairs_index_backward_cindy = process_pairwise_cindy(cindy_data)

    metadata_file = "data/hf-metadata.csv"
    metadata_file = os.path.join(os.environ['PE_PATH'], metadata_file)

    comparisons_file = "results/04122020/comparisons-document-all.csv"
    comparisons_file = os.path.join(os.environ['PE_PATH'], comparisons_file)
    comparisons = pd.read_csv(comparisons_file, dtype={'subject': 'str', 'study': 'str'})
    comparisons.set_index('study', inplace=True)

    metadata = get_metadata(metadata_file, subset=comparisons)
    pairwise_reports = get_pairwise_reports(sort_by_date(metadata))

    all_pairs, pairs_index, pairs_index_backward = process_pairwise(pairwise_reports, comparisons)
    discrepancies, no_comparison_label, agreements = compare(all_pairs, all_pairs_cindy, pairs_index, pairs_index_cindy, pairs_index_backward, pairs_index_backward_cindy)
    
    columns = ['subject', 'study1', 'study2', 'label1', 'label2', 'comparison_nlp', 'true_comp']
    discrepancies = pd.DataFrame(discrepancies, columns=columns)
    discrepancies.to_csv(os.path.join(os.environ['PE_PATH'], "results/04202020/test/comparison-cindy-diff.csv"))

    no_comparison_label = pd.DataFrame(no_comparison_label, columns=columns)
    no_comparison_label.to_csv(os.path.join(os.environ['PE_PATH'], "results/04202020/test/comparison-cindy-no-comparison.csv"))

    agreements = pd.DataFrame(agreements, columns=columns)
    agreements.to_csv(os.path.join(os.environ['PE_PATH'], "results/04202020/test/comparison-cindy-agree.csv"))

    all_pairs_df = []
    for pair in all_pairs: 
        all_pairs_df.append([pair['subject'], pair['study1'], pair['study2'], pair['predicted_comp']])

    all_pairs_df = pd.DataFrame(all_pairs_df, columns=['subject', 'study1', 'study2', 'predicted_comparison'])
    all_pairs_df.to_csv(os.path.join(os.environ['PE_PATH'], "results/04202020/test/all-pairs-comparison.csv"))

if __name__ == "__main__":
    main()


