import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from dataset.organize_reports import get_metadata, sort_by_date_subject
from datetime import datetime, timedelta
import math 
import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd
from pprint import pprint 

def read_rad_data(metadata_file, comparisons_file):
    comparisons = pd.read_csv(comparisons_file, dtype={'study': 'str', 'subject': 'str'})
    comparisons.set_index('study', inplace=True)
    metadata = get_metadata(metadata_file, subset=comparisons)
    metadata = metadata.rename(columns={'study_id': 'study', 'subject_id': 'subject'})
    # pairwise_reports = get_pairwise_reports(sort_by_date(metadata))
    comparisons = pd.concat([metadata, comparisons['comparison'], comparisons['severity']], axis=1)
    comparisons.reset_index(level=0, inplace=True)
    return comparisons

def read_lab_data(file):
    lab_data = pd.read_csv(file, dtype={'subject_id': 'str', 'stay_id': 'str', 'hadm_id': 'str'}) 
    lab_data = lab_data.rename(columns={'subject_id': 'subject', 'stay_id': 'stay', 'hadm_id': 'hadm', 'charttime': 'datetime'})
    return lab_data    

def get_pairwise_reports(series):
    """
    Sort radiology reports by date for a single patient
    
    Inputs 
    report_list     list of studies for a single patient 

    Returns 
    list of tuples of study reports (subject, study1, study2, date1, date2), where the date for study1 is prior to study2
    """
    pairwise = []
    for subject, reports in series.items():
        to_write = sorted(reports, key=lambda x: x['datetime'])
        # Only keep subjects with more than one study 
        if len(to_write) > 0:
            study1, date1 = to_write[0]['study'], to_write[0]['datetime']
            index = 1
            while index < len(to_write):
                study2, date2 = to_write[index]['study'], to_write[index]['datetime']
                # Check that two consecutive reports in the queue are chronologically consecutive before adding pair to final list
                # This check exists because all studies for patient are put into the same queue, even if the patient has multiple
                # radiograph series (e.g. taken during separate hospital visits)
                if abs(date1 - date2) <= timedelta(days=2):
                    pairwise.append({
                        'subject': subject,
                        'study1': study1,
                        'study2': study2,
                        'date1': date1,
                        'date2': date2
                        })
                
                study1, date1 = to_write[index]['study'], to_write[index]['datetime']
                index += 1

    return pairwise

def get_previous_report(series):
    """
    Returns
    dictionary mapping each study to the report immediately preceding it 
    """
    previous = {}
    for subject, reports in series.items():
        to_write = sorted(reports, key=lambda x: x['datetime'])
        # Only keep subjects with more than one study 
        if len(to_write) > 0:
            study1, date1 = to_write[0]['study'], to_write[0]['datetime']
            index = 1
            while index < len(to_write):
                study2, date2, comparison, severity = to_write[index]['study'], to_write[index]['datetime'], to_write[index]['comparison'], to_write[index]['severity']
                if abs(date1 - date2) <= timedelta(days=2):
                    previous[study2] = {'previous': study1, 'previous_datetime': date1, 'datetime': date2, 
                                        'subject': subject, 'comparison': comparison, 'severity': severity} 
                study1, date1 = to_write[index]['study'], to_write[index]['datetime']
                index += 1

    return previous 

def filter_data(lab_data, rad_data):
    subjects_rad = set(rad_data['subject'].tolist()) 
    subjects_lab = set(lab_data['subject'].tolist())
    subjects = subjects_rad.intersection(subjects_lab)

    rad_data_filtered = []
    for index, row in rad_data.iterrows():
        if row['subject'] in subjects: 
            rad_data_filtered.append(row.to_dict())

    lab_data_filtered = []
    for index, row in lab_data.iterrows():
        if row['subject'] in subjects:
            lab_data_filtered.append(row.to_dict())

    rad_data_filtered = pd.DataFrame(rad_data_filtered)
    rad_data_filtered.to_csv(os.path.join(os.environ['PE_PATH'], "data/correlation-filtered/rad-data-bg.csv"))

    lab_data_filtered = pd.DataFrame(lab_data_filtered)
    lab_data_filtered.to_csv(os.path.join(os.environ['PE_PATH'], "data/correlation-filtered/bg-data.csv"))

def filter_vital_data(lab_data, vital_data):
    stays_lab = set(lab_data['stay'].tolist()) 
    stays_vital = set(vital_data['stay'].tolist())
    stays = stays_lab.intersection(stays_vital)

    vital_data_filtered = []
    for index, row in vital_data.iterrows():
        if row['stay'] in stays: 
            vital_data_filtered.append(row.to_dict())

    vital_data_filtered = pd.DataFrame(vital_data_filtered)
    vital_data_filtered.to_csv(os.path.join(os.environ['PE_PATH'], "data/correlation-filtered/vital-data.csv"))

def get_datetime_lab(dt):
    return datetime.strptime(dt, '%Y-%m-%d %H:%M:%S UTC')

def get_datetime_rad(date, time):
    return datetime.strptime(date + time, '%Y%m%d%H%M%S.%f')

def sort_by_subject_date_lab(lab_data):
    lab_sorted = {}
    for index, row in lab_data.iterrows():
        if row['subject'] not in lab_sorted:
            lab_sorted[row['subject']] = []

        lab_sorted[row['subject']].append(row.to_dict())
        lab_sorted[row['subject']][-1]['datetime'] = get_datetime_lab(lab_sorted[row['subject']][-1]['datetime'])

    for subject in lab_sorted:
        lab_sorted[subject].sort(key=lambda x: x['datetime'])

    return lab_sorted

def sort_by_subject_date_rad(rad_data):
    """
    Inputs
    rad_data    dataframe of metadata for HF patients
                index column: study (e.g. study id)
                additional columns: subject_id, study_date, study_time

    Returns
    series      a dictionary mapping subject to list of (study, date) pairs
    """

    # Create dictionary mapping subject to list of tuples (study, date)
    data = {}
    for index, row in rad_data.iterrows():
        if row['subject'] not in data:
            data[row['subject']] = []
        data[row['subject']].append({'study': row['study'], 
                                    'datetime': get_datetime_rad(row['study_date'], row['study_time']), 
                                    'comparison': row['comparison'],
                                    'severity': row['severity']})

    # Create dictinoary mapping subject to list of (study, date) pairs
    for subject in data:
        data[subject].sort(key=lambda x: x['datetime'])

    return data

def find_closest_lab(rad_study, previous_study, lab_data_for_subject):
    previous_date = rad_study['datetime'] - timedelta(days=2) if previous_study is None else previous_study['previous_datetime'] 
    current_closest = None 
    for lab in lab_data_for_subject:
        if lab['datetime'] >= previous_date and lab['datetime'] < rad_study['datetime'] + timedelta(days=2):
            if current_closest is None or abs(lab['datetime'] - rad_study['datetime']) < abs(rad_study['datetime'] - current_closest['datetime']):
                current_closest = lab 

    return current_closest

def find_closest_lab_main(lab_sorted_by_subject, rad_sorted_by_subject, rad_previous_map):
    rad_sorted_by_subject_dict = {}
    for subject, series in rad_sorted_by_subject.items():
        rad_sorted_by_subject_dict[subject] = {}
        for index, study in enumerate(series):
            study_id = study['study']
            rad_sorted_by_subject_dict[subject][study_id] = {}
            rad_sorted_by_subject_dict[subject][study_id]['comparison'] = rad_sorted_by_subject[subject][index]['comparison']
            rad_sorted_by_subject_dict[subject][study_id]['datetime'] = rad_sorted_by_subject[subject][index]['datetime']
            rad_sorted_by_subject_dict[subject][study_id]['severity'] = rad_sorted_by_subject[subject][index]['severity']
            if subject not in lab_sorted_by_subject:
                # rad_sorted_by_subject[subject][index]['closest_lab'] = float('nan')
                rad_sorted_by_subject_dict[subject][study_id]['closest_lab'] = float('nan')
            else:
                previous_study = None if study_id not in rad_previous_map else rad_previous_map[study_id]
                # rad_sorted_by_subject[subject][index]['closest_lab'] = find_closest_lab(study, previous_study, lab_sorted_by_subject[subject])
                rad_sorted_by_subject_dict[subject][study_id]['closest_lab'] = find_closest_lab(study, previous_study, lab_sorted_by_subject[subject])
    
    return rad_sorted_by_subject_dict

def get_direction_of_change(rad_comp, rad_with_lab, rad_previous_map, clinical_var, threshold=0):
    lab_comparison = pd.DataFrame(float('nan'), index=np.arange(rad_comp.shape[0]), columns=['lab_comparison'])
    severity_comparison = pd.DataFrame(float('nan'), index=np.arange(rad_comp.shape[0]), columns=['severity_comparison'])
    for index, row in rad_comp.iterrows():
        current_study = row['study']
        previous_study = None if current_study not in rad_previous_map else rad_previous_map[current_study]['previous']
        if previous_study is None:
            continue 

        # Lab value change 
        current_lab = rad_with_lab[row['subject']][current_study]['closest_lab']
        previous_lab = rad_with_lab[row['subject']][previous_study]['closest_lab']
        if current_lab is None or previous_lab is None:
            continue 

        current_clinical = rad_with_lab[row['subject']][current_study]['closest_lab'][clinical_var]   
        previous_clinical = rad_with_lab[row['subject']][previous_study]['closest_lab'][clinical_var]  

        if math.isnan(current_clinical) or math.isnan(previous_clinical):
            continue 

        if abs(current_clinical - previous_clinical) <= threshold:
            lab_comparison['lab_comparison'][index] = 0
        else:
            lab_comparison['lab_comparison'][index] = np.sign(current_clinical - previous_clinical)

        # Severity value change 
        current_severity = rad_with_lab[row['subject']][current_study]['severity']
        previous_severity = rad_with_lab[row['subject']][previous_study]['severity']
        if math.isnan(current_severity) or math.isnan(previous_severity):
            continue 

        severity_comparison['severity_comparison'][index] = np.sign(current_severity - previous_severity)

    return pd.concat([rad_comp['study'], rad_comp['subject'], rad_comp['comparison'], severity_comparison, lab_comparison], axis=1)

def get_distributions(rad_comp, rad_with_lab, rad_previous_map, clinical_var):
    comparison_groups = {'better': {'report1': [], 'report2': [], 'diff': []}, 
                        'worse': {'report1': [], 'report2': [], 'diff': []},
                        'same': {'report1': [], 'report2': [], 'diff': []}}

    severity_groups = {'better': {'report1': [], 'report2': [], 'diff': []}, 
                        'worse': {'report1': [], 'report2': [], 'diff': []},
                        'same': {'report1': [], 'report2': [], 'diff': []}}

    for index, row in rad_comp.iterrows():
        current_study = row['study']
        previous_study = None if current_study not in rad_previous_map else rad_previous_map[current_study]['previous']
        if previous_study is None:
            continue 

        # Lab value change 
        current_lab = rad_with_lab[row['subject']][current_study]['closest_lab']
        previous_lab = rad_with_lab[row['subject']][previous_study]['closest_lab']
        if current_lab is None or previous_lab is None:
            continue 

        current_clinical = rad_with_lab[row['subject']][current_study]['closest_lab'][clinical_var]   
        previous_clinical = rad_with_lab[row['subject']][previous_study]['closest_lab'][clinical_var]  

        if math.isnan(current_clinical) or math.isnan(previous_clinical):
            continue 

        if row['comparison'] == 0.0:
            comparison_groups['same']['report1'].append(previous_clinical)
            comparison_groups['same']['report2'].append(current_clinical)
            comparison_groups['same']['diff'].append(current_clinical - previous_clinical)
        elif row['comparison'] == 1.0:
            comparison_groups['worse']['report1'].append(previous_clinical)
            comparison_groups['worse']['report2'].append(current_clinical)
            comparison_groups['worse']['diff'].append(current_clinical - previous_clinical)
        elif row['comparison'] == -1.0:
            comparison_groups['better']['report1'].append(previous_clinical)
            comparison_groups['better']['report2'].append(current_clinical)
            comparison_groups['better']['diff'].append(current_clinical - previous_clinical)

        # if math.isnan(row['comparison']):
        #     continue 

        # Severity value change 
        current_severity = rad_with_lab[row['subject']][current_study]['severity']
        previous_severity = rad_with_lab[row['subject']][previous_study]['severity']
        if math.isnan(current_severity) or math.isnan(previous_severity):
            continue 

        if np.sign(current_severity - previous_severity) == 0.0:
            severity_groups['same']['report1'].append(previous_clinical)
            severity_groups['same']['report2'].append(current_clinical)
            severity_groups['same']['diff'].append(current_clinical - previous_clinical)
        elif np.sign(current_severity - previous_severity) == 1.0:
            severity_groups['worse']['report1'].append(previous_clinical)
            severity_groups['worse']['report2'].append(current_clinical)
            severity_groups['worse']['diff'].append(current_clinical - previous_clinical)
        elif np.sign(current_severity - previous_severity) == -1.0:
            severity_groups['better']['report1'].append(previous_clinical)
            severity_groups['better']['report2'].append(current_clinical)
            severity_groups['better']['diff'].append(current_clinical - previous_clinical)

    return comparison_groups, severity_groups

def filter_main():
    # lab_path = "data/clinical-variables/pivoted_lab_m4.csv"
    lab_path = "data/clinical-variables/bg.csv" 
    lab_path = os.path.join(os.environ['PE_PATH'], lab_path)
    lab_data = read_lab_data(lab_path)

    metadata_file = "data/hf-metadata.csv"
    metadata_file = os.path.join(os.environ['PE_PATH'], metadata_file)
    comparisons_file = "results/05272020/comparisons-document-all.csv"
    comparisons_file = os.path.join(os.environ['PE_PATH'], comparisons_file)
    rad_data = read_rad_data(metadata_file, comparisons_file)

    filter_data(lab_data, rad_data)

def filter_main_vital():
    lab_path = "data/correlation-filtered/lab-data.csv" 
    lab_path = os.path.join(os.environ['PE_PATH'], lab_path)
    lab_data = read_lab_data(lab_path)

    vital_path = "data/clinical-variables/vital.csv" 
    vital_path = os.path.join(os.environ['PE_PATH'], vital_path)
    vital_data = read_lab_data(vital_path)

    filter_vital_data(lab_data, vital_data)

def test():
    lab_path = "data/correlation-filtered/bg-data.csv" 
    lab_path = os.path.join(os.environ['PE_PATH'], lab_path)
    lab_data = pd.read_csv(lab_path, dtype={'subject': 'str', 'stay': 'str', 'hadm': 'str', 'datetime': 'str'}) 

    rad_path = "data/correlation-filtered/rad-data-bg.csv" 
    rad_path = os.path.join(os.environ['PE_PATH'], rad_path)
    rad_data = pd.read_csv(rad_path, dtype={'subject': 'str', 'study': 'str', 'study_date': 'str', 'study_time': 'str'})

    lab_sorted_by_subject = sort_by_subject_date_lab(lab_data)
    rad_sorted_by_subject = sort_by_subject_date_rad(rad_data)
    rad_previous_map = get_previous_report(rad_sorted_by_subject)

    rad_with_lab = find_closest_lab_main(lab_sorted_by_subject, rad_sorted_by_subject, rad_previous_map)
    rad_lab_comparisons = get_direction_of_change(rad_data, rad_with_lab, rad_previous_map, 'po2')
    rad_lab_comparisons.to_csv(os.path.join(os.environ['PE_PATH'], "data/correlation-filtered/test-po2.csv"))
    # toprint = [r for r in rad_with_lab['10018081'] if r['closest_lab'] is not None]
    # pprint(rad_with_lab['10018081'])
    # pprint(len(toprint))

    comparison_groups, severity_groups = get_distributions(rad_data, rad_with_lab, rad_previous_map, 'po2')

    colors = ["cyan", "magenta", "green"]
    index = 0
    print("COMPARISON DISTRIBUTION")
    for key, value in comparison_groups.items():
        print(key, "({}, {})".format(np.median(value['report1']), np.median(value['report2'])))
        print(key, "({:0.4f}, {:0.4f})".format(np.mean(value['report1']), np.mean(value['report2'])))
        print(key, "({}, {}, {})".format(np.median(value['diff']), np.percentile(value['diff'], 25), np.percentile(value['diff'], 75)))
        plt.scatter(value['report1'], value['report2'], color=colors[index], s=5)
        index += 1

    print("SEVERITY DISTRIBUTION")
    for key, value in severity_groups.items():
        print(key, "({}, {})".format(np.median(value['report1']), np.median(value['report2'])))
        print(key, "({:0.4f}, {:0.4f})".format(np.mean(value['report1']), np.mean(value['report2'])))
        print(key, "({}, {}, {})".format(np.median(value['diff']), np.percentile(value['diff'], 25), np.percentile(value['diff'], 75)))

    plt.legend(['better', 'worse', 'same'])
    plt.show()

if __name__ == "__main__":
    test() 
    # filter_main()

    # severity = "results/04122020/automatic-document-labels-severity.csv"
    # severity = os.path.join(os.environ['PE_PATH'], severity)
    # severity = pd.read_csv(severity, dtype={'study': 'str', 'subject': 'str'})

    # comparisons = "results/04122020/comparisons-document-all.csv"
    # comparisons = os.path.join(os.environ['PE_PATH'], comparisons)
    # comparisons = pd.read_csv(comparisons, dtype={'study': 'str', 'subject': 'str'})

    # comparisons = pd.concat([comparisons, severity['severity']], axis=1)
    # comparisons.to_csv(os.path.join(os.environ['PE_PATH'], "results/05272020/comparisons-document-all.csv"))





