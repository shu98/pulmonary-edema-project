import argparse
import os
import pandas as pd

def print_incorrect_classified():
    """
    Get sentences whose relevance to pulmonary edema is incorrectly classified. 
    """
    parser = argparse.ArgumentParser(description='Print incorrectly classified sentences')

    # Relative paths to PE_PATH
    parser.add_argument('true_labels_path', type=str, help='Path to file with ground-truth relevance labels')
    parser.add_argument('predicted_labels_path', type=str, help='Path to file with predicted relevance labels')
    args = parser.parse_args()

    true_labels_path = os.path.join(os.environ['PE_PATH'], args.true_labels_path)  
    predicted_labels_path = os.path.join(os.environ['PE_PATH'], args.predicted_labels_path)

    true_labels = pd.read_csv(true_labels_path)
    predicted_labels = pd.read_csv(predicted_labels_path)

    for index, row in true_labels.iterrows():
        if row['relevant'] != predicted_labels['relevant'][index]:
            print("{},{},\"{}\"".format(row['relevant'], predicted_labels['relevant'][index], row['sentence']))

if __name__ == "__main__":
    print_incorrect_classified()