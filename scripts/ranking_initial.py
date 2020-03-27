import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import argparse
from dataset.organize_reports import sort_by_date, get_metadata, get_datetime
from datetime import datetime, timedelta
import math
from nlp.annotation import Annotation 
import numpy as np 
import pandas as pd
from pprint import pprint 
from util.document import collect_sentences

def small_test():
    series =  [('57539618', 2.0, float('nan'), 1.0),
                ('55735807', 2.0, -1.0, 1.0),
                ('58536937', 0.0, float('nan'), 0.0),
                ('55281127', float('nan'), -1.0, 1.0),
                ('56762822', 0.0, float('nan'), 0.0),
                ('59614225', 0.0, float('nan'), 0.0),
                ('50772344', float('nan'), float('nan'), 0.0),
                ('57486705', float('nan'), float('nan'), float('nan')),
                ('51961926', 0.0, float('nan'), 0.0),
                ('58479559', float('nan'), float('nan'), float('nan')),
                ('57481090', float('nan'), 0.0, float('nan')),
                ('57747740', 1.0, -1.0, 1.0),
                ('55605617', float('nan'), float('nan'), float('nan')),
                ('55879987', float('nan'), 0.0, float('nan')),
                ('52015079', float('nan'), 0.0, float('nan')),
                ('53437264', float('nan'), 0.0, float('nan'))]

    new_series = []
    for index, item in enumerate(series):
        study, score, comp, edema = item 
        previous = None if index == 0 else new_series[-1]
        new_series.append(Annotation(None, study, None, edema=edema, severity=score, previous=previous, comparison=comp))

    for s in new_series:
        print(s.study, s.severity, s.comparison)

def another_test():
    series = [('56122988', 0.0, float('nan'), 0.0),
             ('58038331', float('nan'), float('nan'), float('nan')),
             ('51318630', float('nan'), float('nan'), float('nan')),
             ('56490401', 0.0, 0.0, 0.0),
             ('57289788', float('nan'), 0.0, 1.0),
             ('51697762', float('nan'), 0.0, 1.0),
             ('55999797', float('nan'), 0.0, 1.0),
             ('51803497', float('nan'), 1.0, 1.0),
             ('52996004', float('nan'), 1.0, 1.0),
             ('56108243', float('nan'), -1.0, 1.0),
             ('56838761', float('nan'), -1.0, 1.0),
             ('52647330', float('nan'), 0.0, float('nan')),
             ('56392162', float('nan'), 1.0, 1.0),
             ('55059718', float('nan'), 1.0, 1.0),
             ('54812204', float('nan'), 1.0, 1.0),
             ('59048456', float('nan'), 0.0, 1.0),
             ('54080354', float('nan'), 0.0, 1.0),
             ('50089402', float('nan'), 0.0, 1.0),
             ('58536693', float('nan'), 0.0, 1.0),
             ('57409239', float('nan'), float('nan'), float('nan'))]

    new_series = []
    for index, item in enumerate(series):
        study, score, comp, edema = item 
        previous = None if index == 0 else new_series[-1]
        new_series.append(Annotation(None, study, None, edema=edema, severity=score, previous=previous, comparison=comp))

    for s in new_series:
        print(s.study, s.severity, s.comparison)

def get_full_series(pairwise_severity, comparison_labels):
    """
    Construct series of consecutive radiology reports 

    Inputs
    pairwise_severity   list of pairwise comparisons between reports (subject, study1, study2, label1, label2)
    comparison_labels   CSV file with comparison label for individual radiology reports
                        expected column 'comparison'

    Returns
    List of series, where one series is a list of tuples (subject, study1, study2, label1, label2, comparison)
    One subject may have multiple series 
    """
    series = []
    current = [(pairwise_severity[0]) + (comparison_labels['comparison'][pairwise_severity[0][2]],)]
    index = 1
    while index < len(pairwise_severity):
        subject, study1, study2, label1, label2 = pairwise_severity[index]
        if study1 != current[-1][2]:
            series.append(current)
            current = [(pairwise_severity[index]) + (comparison_labels['comparison'][study2],)]
        else:
            current.append((pairwise_severity[index]) + (comparison_labels['comparison'][study2],))

        index += 1

    return series

def get_full_series_as_dict(pairwise_severity, comparison_labels, edema_labels):
    series = []
    current = [{'subject': pairwise_severity[0][0], 
                'study1': pairwise_severity[0][1],
                'study2': pairwise_severity[0][2],
                'label1': pairwise_severity[0][3],
                'label2': pairwise_severity[0][4],
                'comparison': comparison_labels['comparison'][pairwise_severity[0][2]],
                'edema1': edema_labels['edema'][pairwise_severity[0][1]],
                'edema2': edema_labels['edema'][pairwise_severity[0][2]]
            }]

    index = 1
    while index < len(pairwise_severity):
        subject, study1, study2, label1, label2 = pairwise_severity[index]
        if study1 != current[-1]['study2']:
            series.append(current)
            current = [{'subject': pairwise_severity[index][0], 
                'study1': pairwise_severity[index][1],
                'study2': pairwise_severity[index][2],
                'label1': pairwise_severity[index][3],
                'label2': pairwise_severity[index][4],
                'comparison': comparison_labels['comparison'][study2],
                'edema1': edema_labels['edema'][study1],
                'edema2': edema_labels['edema'][study2]
            }]

        else:
            current.append({'subject': pairwise_severity[index][0], 
                'study1': pairwise_severity[index][1],
                'study2': pairwise_severity[index][2],
                'label1': pairwise_severity[index][3],
                'label2': pairwise_severity[index][4],
                'comparison': comparison_labels['comparison'][study2],
                'edema1': edema_labels['edema'][study1],
                'edema2': edema_labels['edema'][study2]
            })
        index += 1

    return series 

def split_by_inc_dec_segments(series):
    segments = []
    current = [(series[0]['study1'], 0)]
    current_direction = 0
    for comp in series:
        if math.isnan(comp['comparison']):
            segments.append(current)
            current = [(comp['study2'], 0)]
            current_direction = 0
            continue 

        if current_direction != 0:
            if comp['comparison'] != 0 and comp['comparison'] != current_direction:
                segments.append(current)
                current = [(comp['study1'], 0)]
                current_direction = 0

        last_study, last_score = current[-1]
        current.append((comp['study2'], last_score + comp['comparison'] ))

        if current_direction == 0:
            current_direction = comp['comparison'] 

    segments.append(current)
    return segments 

def construct_pairwise_comparisons(segment):
    pairs = []
    for i in range(len(segment)):
        for j in range(i+1, len(segment)):
            study1, score1 = segment[i]
            study2, score2 = segment[j]
            if score1 > score2: 
                pairs.append((study1, study2, 1))
            elif score2 > score1: 
                pairs.append((study2, study1, 1))
            else:
                pairs.append((study1, study2, 0))

    return pairs 

def sort_series(series):
    studies = {}
    for study1, study2, comp in series:
        if study1 not in studies:
            studies[study1] = 0

        studies[study1] += comp 

    studies_sortable = []
    for study, score in studies.items():
        studies_sortable.append((study, score))

    studies_sortable.sort(reverse=True, key=lambda x: x[1])
    return studies_sortable 

def get_pairwise_severity(series, severity_labels):
    """
    Sort radiology reports by date for a single patient
    
    Inputs 
    report_list     list of studies for a single patient 

    Returns 
    list of tuples of study reports (subject, study1, study2, label1, label2), where the date for study1 is prior to study2
    """
    severity_pairiwse = []
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
                    severity_pairiwse.append((subject, study1, study2, severity_labels['severity'][study1], severity_labels['severity'][study2]))
                
                study1, date1 = to_write[index]
                index += 1

    return severity_pairiwse

def rank_all(all_series, foldername):
    folderpath = os.path.join(os.environ['PE_PATH'], foldername)
    if not os.path.exists(folderpath):
        os.makedirs(folderpath)

    for i, series in enumerate(all_series):
        if len(series) == 0:
            continue 

        subject = series[0]['subject']
        if (i+1) % 500 == 0:
            print("Wrote {} rankings".format(i+1)) 

        comps = split_by_inc_dec_segments(series)
        all_pairs_example = []
        for segment in comps:
            all_pairs_example.extend(construct_pairwise_comparisons(segment))

        sorted_series = sort_series(all_pairs_example)
        for index, elem in enumerate(sorted_series):
            sorted_series[index] = list(elem)

        filename = "{}/p{}.csv".format(folderpath, subject)
        pd.DataFrame(sorted_series).to_csv(filename)

def get_edema_label_document(labels, start, end):
    index = start
    chexpert_label = float('nan')
    keyword_label = float('nan')
    document_label = float('nan')
    while index < end:
        if not math.isnan(labels['chexpert_label'][index]):
            if chexpert_label != 1.0:
                chexpert_label = labels['chexpert_label'][index]
        if not math.isnan(labels['keyword_label'][index]):
            if keyword_label != 1.0:
                keyword_label = hexpert_label = labels['keyword_label'][index]

        index += 1 
    
    if chexpert_label == 1.0 or keyword_label == 1.0:
        document_label = 1.0
    elif chexpert_label == 0.0 or keyword_label == 0.0:
        document_label = 0.0 

    return (document_label, chexpert_label, keyword_label)

def get_edema_labels(labels):
    document_labels = []
    index = 0
    while index < labels.shape[0]:
        subject = labels['subject'][index]
        study = labels['study'][index]
        report, start, end = collect_sentences(labels, subject, study, start_index=index)
        document_label, chexpert_label, keyword_label = get_edema_label_document(labels, start, end)
        document_labels.append([subject, study, document_label, chexpert_label, keyword_label])
        index = end

    output = pd.DataFrame(document_labels, columns=['subject', 'study', 'edema', 'chexpert_label', 'keyword_label'])
    return output 

def infer_edema_label(series):
    """
    Modifies input series by updating 'edema2' for each pair in the series based on a set of rules 
    """
    for s in series: 
        for index, elem in enumerate(s):
            if not math.isnan(elem['edema2']):
                continue 

            if elem['comparison'] == 0.0 or math.isnan(elem['comparison']):
                elem['edema2'] = elem['edema1']
                if index < len(s) - 1:
                    s[index + 1]['edema1'] = elem['edema1']

    return series

def infer_comparison_label(series):
    """
    Modifies input series by updating 'comparison' for each pair in the series based on a set of rules 
    """
    series = infer_edema_label(series)
    for s in series:
        for index, elem in enumerate(s):
            if not math.isnan(elem['comparison']):
                continue 
            # Both reports have a severity score 
            elif not math.isnan(elem['label1']) and not math.isnan(elem['label2']):
                elem['comparison'] = np.sign(elem['label1'] - elem['label2'])

            elif elem['label2'] == 0.0 or elem['edema2'] == 0.0:
                if elem['edema1'] == 0.0:
                    elem['comparison'] = 0.0

                elif elem['edema1'] == 1.0 or elem['label1'] > 0.0:
                    elem['comparison'] = 1.0

            elif elem['label2'] > 0.0 or elem['edema2'] == 1.0:
                if elem['edema1'] == 1.0 or elem['label1'] > 0.0:
                    elem['comparison'] = 0.0

                elif elem['edema1'] == 0.0 or elem['label1'] == 0.0:
                    elem['comparison'] = -1.0

    return series 
     
def main():
    parser = argparse.ArgumentParser(description='Rank radiology reports within a single patient')
    parser.add_argument('metadata_path', type=str, help='Path to file with metadata')
    parser.add_argument('document_labels_path', type=str, help='Path to file with document labels')
    # Necessary to get edema (presence/absence) labels
    parser.add_argument('relevance_labels_path', type=str, help='Path to file with automatic relevance labels')
    parser.add_argument('result_folder', type=str, help='Path to folder to write ranking results')
    args = parser.parse_args()

    # File is path likely data/hf-metadata.csv
    metadata_file = os.path.join(os.environ['PE_PATH'], args.metadata_path)

    # File path is likely results/03192020/automatic-document-labels.csv
    document_labels_file = os.path.join(os.environ['PE_PATH'], args.document_labels_path)
    document_labels = pd.read_csv(document_labels_file, dtype={'study': 'str', 'subject': 'str'})
    document_labels.set_index('study', inplace=True)

    # Get automatically labeled edema labels from relevance labels
    relevance_labels_file = os.path.join(os.environ['PE_PATH'], args.relevance_labels_path)
    relevance_labels = pd.read_csv(relevance_labels_file, dtype={'study': 'str', 'subject': 'str'})
    edema_labels = get_edema_labels(relevance_labels)
    edema_labels.set_index('study', inplace=True)

    metadata = get_metadata(metadata_file, document_labels)
    pairwise_severity = get_pairwise_severity(sort_by_date(metadata), document_labels)

    all_series = get_full_series_as_dict(pairwise_severity, document_labels, edema_labels)
    all_series = infer_comparison_label(all_series)
    rank_all(all_series, args.result_folder)
    
    current_max, second_max = all_series[0], all_series[1]
    if len(second_max) > len(current_max):
        current_max, second_max = second_max, current_max 

    for s in all_series:
        if len(s) > len(current_max):
            current_max, second_max = s, current_max 
        elif len(s) > len(second_max):
            second_max = s 

    # print(current_max[0] + (str(get_datetime(metadata['study_date'][current_max[0][1]], metadata['study_time'][current_max[0][1]])),))
    # for e in current_max:
    #     print(e + (str(get_datetime(metadata['study_date'][e[2]], metadata['study_time'][e[2]])),))

    # pprint(second_max)
    
    # comps = split_by_inc_dec_segments(current_max)
    # print(comps)
    # all_pairs_example = []
    # for segment in comps:
    #     all_pairs_example.extend(construct_pairwise_comparisons(segment))

    # pprint(sort_series(all_pairs_example))


if __name__ == "__main__":
    main()
    # another_test()


