import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

def collect_sentences(labels, subject, study, start_index=0):
    report = []
    index = start_index
    while index < labels.shape[0] and labels['subject'][index] == subject and labels['study'][index] == study:
        report.append(labels['sentence'][index])
        index += 1

    return report, start_index, index