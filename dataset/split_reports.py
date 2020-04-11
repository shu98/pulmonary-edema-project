#
# Splits the FINDINGS and IMPRESSIONS sections of a set of radiology reports into individual sentences 
# Radiology reports for a given patient are ordered chronologically  
# Output file contains columns: sentence, study_id, subject_id
#

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import argparse
from datetime import datetime, timedelta
import math
import os
import pandas as pd 
from pprint import pprint
import random 
import re 

from dataset.organize_reports import sort_by_date, get_metadata

def should_skip(line):
    """
    Sections that should be skipped when splitting radiology report into sentences 
    """
    should_skip_set = set([
        "NOTIFICATION",
        "INDICATION",
        "RECOMMENDATION",
        "TECHNIQUE",
        "FINAL REPORT",
        "COMPARISON",
        "WET READ",
        "EXAMINATION",
        "HISTORY",
        "REASON FOR EXAMINATION"])

    for word in should_skip_set:
        if word in line: return True 

    return False 

def write_sentences(report_folder, metadata_file):
    """
    Orders radiology reports chronologically and splits the FINDINGS and IMPRESSIONS sections of each report into individual sentences. 

    report_folder   folder with radiology reports to be split
    metadata_file   metadata information for radiology reports, in particular study date/time
                    expects column names 'study_date' and 'study_time'
    """
    subjects = set(os.listdir(report_folder))
    metadata = get_metadata(metadata_file)

    all_sentences = []
    sorted_by_subject = sort_by_date(metadata)
    num_written = 0
    for subject, reports in sorted_by_subject.items():
        if "p{}".format(subject) not in subjects:
            continue 

        studies = set(os.listdir("{}/p{}".format(report_folder, subject)))
        for study, date in reports:
            if "s{}.txt".format(study) not in studies:
                continue 

            did_write_flag = False 
            with open("{}/p{}/s{}.txt".format(report_folder, subject, study), "r") as f:
                entire_report = f.readlines()
                # Replace new lines with spaces 
                lines = " ".join(entire_report).replace("\n", " ")

                # Add period so that string.split() will split before heading 
                lines = lines.replace("FINDINGS", ".FINDINGS")
                lines = lines.replace("IMPRESSION", ".IMPRESSION")
                lines = lines.replace("CONCLUSION", ".CONCLUSION")

                # Split sentences by punctuation marks
                lines = re.split('\.|\?|//', re.sub('\s+', ' ', lines))

                start_writing = False
                for line in lines:
                    # Check for the start of a new section
                    if should_skip(line):
                        start_writing = False
                        continue 
                    
                    start = 0
                    if "FINDINGS" in line or "IMPRESSION" in line or "CONCLUSION" in line:
                        start_writing = True
                        # Only include the part of the line that follows the FINDINGS or IMPRESSION heading
                        start = re.search('FINDINGS|IMPRESSION|CONCLUSION', line).start()

                    if start_writing:
                        line = line[start:]
                        line = re.sub('FINDINGS:|IMPRESSION:|CONCLUSION:', '', line)
                        # Ignore sentences that are too short 
                        if len(line) > 5:
                            all_sentences.append(["{}".format(line.strip()), subject, study])
                            did_write_flag = True 

                # Handle reports that don't have a FINDINGS or IMPRESSIONS section title
                if not did_write_flag:
                    for line in lines:
                        # Remove as many unnecessary lines as possible (e.g. if the line explicitly has a
                        # do-not-include section title like COMPARISON) 
                        if should_skip(line):
                            continue
                        if len(line) > 5:
                            all_sentences.append(["{}".format(line.strip()), subject, study])
                            did_write_flag = True 

            if did_write_flag: num_written += 1 
            else: print(subject, study)

    print("Wrote {} radiology reports".format(num_written))
    return pd.DataFrame([] + all_sentences)

def split_main():
    parser = argparse.ArgumentParser(description='Split the FINDINGS and IMPRESSIONS sections of a set of radiology reports into sentences')
    
    # Data folder should have structure
    # root
    #   subject1
    #       study1.txt
    #       study2.txt
    #       ...
    #   subject2
    #   ...
    parser.add_argument('data_folder_path', type=str, help='Path to folder with radiology reports')
    parser.add_argument('metadata_path', type=str, help='Path to file with metadata for radiology reports')
    parser.add_argument('output_path', type=str, help='Path to file where list of sentences is written')
    args = parser.parse_args()

    # data_folder_path = os.path.join(os.environ['PE_PATH'], args.data_folder_path)  
    data_folder_path = args.data_folder_path
    metadata_path = os.path.join(os.environ['PE_PATH'], args.metadata_path)  
    output_path = os.path.join(os.environ['PE_PATH'], args.output_path)  

    result = write_sentences(data_folder_path, metadata_path)
    result.to_csv(output_path, index=False, header=["sentence", "subject", "study"])

if __name__ == "__main__":
    split_main()    


