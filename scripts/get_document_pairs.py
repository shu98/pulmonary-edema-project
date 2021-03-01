import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from datetime import datetime, timedelta 
import math 
import pandas as pd 
from pprint import pprint
from dataset.organize_reports import sort_by_date, get_metadata

def process_pairwise(data, comparison_labels):
    pairs = []
    pairs_index = {}
    for index, row in enumerate(data):
        pairs.append({'subject': row[0],
                    'study1': row[1],
                    'study2': row[2],
                    'predicted_comp': comparison_labels['comparison'][row[2]]})
        pairs_index["{}{}".format(row[1], row[2])] = index

    return pairs, pairs_index

def get_pairwise_reports(series):
    """
    Sort radiology reports by date for a single patient
    
    Inputs 
    report_list     list of studies for a single patient 

    Returns 
    list of tuples of study reports (subject, study1, study2), where the date for study1 is prior to study2
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

def sort_pairwise(pairs, pairs_index):
    all_pairs = []

    for pair in pairs:
        comparison_index = pairs_index["{}{}".format(pair['study1'], pair['study2'])]
        pair_data = pairs[comparison_index]

        comparison_label = pair_data['predicted_comp']
        to_append = [pair_data['subject'], pair_data['study1'], pair_data['study2'], comparison_label]
        all_pairs.append(to_append)

    return all_pairs

def main():
    metadata_file = "data/hf-metadata.csv"
    metadata_file = os.path.join(os.environ['PE_PATH'], metadata_file)

    comparisons_file = "results/11152020/comparisons-document-all.csv"
    comparisons_file = os.path.join(os.environ['PE_PATH'], comparisons_file)
    comparisons = pd.read_csv(comparisons_file, dtype={'subject': 'str', 'study': 'str'})
    comparisons.set_index('study', inplace=True)

    metadata = get_metadata(metadata_file, subset=comparisons)
    pairwise_reports = get_pairwise_reports(sort_by_date(metadata))

    all_pairs, pairs_index = process_pairwise(pairwise_reports, comparisons)
    all_pairs = sort_pairwise(all_pairs, pairs_index)
    
    # all_pairs_df = []
    # for pair in all_pairs: 
    #     all_pairs_df.append([pair['subject'], pair['study1'], pair['study2'], pair['predicted_comp']])

    all_pairs_df = pd.DataFrame(all_pairs, columns=['subject', 'study1', 'study2', 'predicted_comparison'])
    all_pairs_df.to_csv(os.path.join(os.environ['PE_PATH'], "results/11152020/comparisons-document-all-pairs.csv"), index=False)

if __name__ == "__main__":
    main()


