import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from datetime import datetime, timedelta
import math 
import pandas as pd 

def get_metadata(metadata_file, subset=None):
    metadata = pd.read_csv(metadata_file, dtype={'study_id': 'str', 'subject_id': 'str', 'study_date': 'str', 'study_time': 'str'}).dropna().drop_duplicates()
    metadata = metadata.rename(columns={"study_id": "study"})
    metadata = metadata.drop_duplicates(subset='study', keep='first')
    metadata.set_index('study', inplace=True)

    metadata = metadata.drop(columns=['dicom_id'])
    if subset is not None:
        metadata = metadata.reindex(subset.index)

    return metadata

def get_datetime(date, time):
    return datetime.strptime(date + time, '%Y%m%d%H%M%S.%f')

def sort_by_date_subject(report_list):
    """
    Sort radiology reports by date for a single patient
    
    Inputs 
    report_list     list of studies for a single patient 

    Returns 
    list of tuples of study reports (study1, date1)
    """

    visited = set()
    to_write = []

    # Compare the date/time of each radiology report to that of every other radiology report 
    for i in range(len(report_list)):
        for j in range(i+1, len(report_list)):
            prev_study, prev_date = report_list[i]
            current_study, current_date = report_list[j]

            # Consider two radiology reports to be consecutive if they are written within 2 days of each other
            if abs(current_date - prev_date) <= timedelta(days=2) and prev_study != current_study:
                # If within 2 days of each other, add (radiology report, date) to a queue of reports to be processed 
                if prev_study not in visited:
                    to_write.append((prev_study, prev_date))
                    visited.add(prev_study)
                if current_study not in visited:
                    to_write.append((current_study, current_date))
                    visited.add(current_study)

    to_write = sorted(to_write, key=lambda x: x[1])
    return to_write

def sort_by_date(metadata):
    """
    Inputs
    metadata    dataframe of metadata for HF patients
                index column: study (e.g. study id)
                additional columns: subject_id, study_date, study_time

    Returns
    series      a dictionary mapping subject to list of (study, date) pairs
    """

    # Create dictionary mapping subject to list of tuples (study, date)
    data = {}
    for index, row in metadata.iterrows():
        if row['subject_id'] not in data:
            data[row['subject_id']] = []

        data[row['subject_id']].append((index, get_datetime(row['study_date'], row['study_time'])))

    # Create dictinoary mapping subject to list of (study, date) pairs
    series = {}
    for subject in data:
        data[subject].reverse()
        series[subject] = sort_by_date_subject(data[subject])

    return series

