import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import numpy as np 
import pandas as pd 

from dataset.organize_reports import sort_by_date, get_metadata

if __name__ == "__main__":
    sentences_file = "data/dataset-small/ground-truth-labels-new.csv"
    sentences_file = os.path.join(os.environ['PE_PATH'], sentences_file)

    sentences = pd.read_csv(sentences_file, dtype={'subject': 'str', 'study': 'str'})
    document_list = set()
    for index, row in sentences.iterrows():
        document_list.add((row['subject'], row['study']))

    documents = pd.DataFrame(document_list, columns=['subject', 'study'])
    documents.set_index('study', inplace=True)

    metadata_file = os.path.join(os.environ['PE_PATH'], "data/hf-metadata.csv")
    metadata = get_metadata(metadata_file, documents)

    series_sorted = sort_by_date(metadata)
    document_list = []
    for subject, series in series_sorted.items():
        for study, date in series:
            document_list.append([subject, study])

    documents = pd.DataFrame(document_list, columns=['subject', 'study'])
    comparisons = pd.DataFrame(float('nan'), index=np.arange(documents.shape[0]), columns=['ground_truth'])
    documents = pd.concat([documents, comparisons], axis=1)
    documents.to_csv(os.path.join(os.environ['PE_PATH'], "data/dataset-small/ground-truth-comparisons-document-test.csv"))
