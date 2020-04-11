#
# Creates a metadata file needed to chronologically order individual patients' radiology reports 
# Required columns in output file: dicom_id, study_id, subject_id, study_date, study_time
#

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import argparse
import pandas as pd 

def get_metadata(hf_patients_file, metadata_file, output_file):
    """
    Create metadata file for list of CHF patients with study ID and subject ID

    hf_patients_file    CSV file with list of dicoms associated with heart failure patients
                        expects column names 'dicom_id', 'study_id', and 'subject_id'
    
    metadata_file       CSV file with dicom IDs as rows and a number of metadata fields as columns
                        expects columns with names 'dicom', 'StudyDate', and 'StudyTime'
    
    output_file         Name of the CSV file where output metadata is written
    """

    # Use 'dicom_id' as names for row indices
    hf_patients = pd.read_csv(hf_patients_file, sep=',', index_col="dicom_id")

    # Use 'dicom' as name
    metadata = pd.read_csv(metadata_file, index_col="dicom", dtype={"StudyDate": str, "StudyTime": str})

    # Disregard all columns except 'subject_id' and 'study_id'
    hf_patients = pd.concat([hf_patients['study_id'], hf_patients['subject_id']], axis=1)

    # Find study date/time for heart failure patients
    study_date = metadata["StudyDate"][hf_patients.index]
    study_time = metadata["StudyTime"][hf_patients.index]

    result = pd.concat([hf_patients, study_date, study_time], axis=1)
    result = result.rename(columns={"StudyDate": "study_date", "StudyTime": "study_time"})

    result.to_csv(output_file)

def main_metadata():
    parser = argparse.ArgumentParser(description='Create metadata file for radiology report dataset')
    parser.add_argument('hf_patients_file', type=str, help='File with a list of heart failure patients and associated reports')
    parser.add_argument('metadata_file', type=str, help='Original metadata file from which to extract necessary columns and rows')
    parser.add_argument('output_file', type=str, help='File path containing output metadata')
    args = parser.parse_args()

    get_metadata(args.hf_patients_file, args.metadata_file, args.output_file)

if __name__ == "__main__":
    main_metadata()

    

    

