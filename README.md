Requires
- Negex 
- Scispacy

1. Split the FINDING and IMPRESSIONS sections of radiology reports into individual sentences.
	a) Create metadata file: python3 dataset/process_files.py data/hf-patients.csv data/dicom-metadata.csv data/hf-metadata.csv
	b) Split radiology reports: python3 dataset/split_reports.py ../data/gold-standard data/hf-metadata.csv data/sentences-split.csv

2. Run CheXpert labeler on individual sentences. 
    a) Input: CSV file with one column of sentences 
    b) Command: python3 label.py --reports_path /path/to/chexpert/file

3. Identify sentences relevant to pulmonary edema using get_relevant_sentences.py
	a) Set PE_PATH environment variable to root of codebase 
    b) Input: output file of CheXpert labeler, filename to write results 
    c) Output: CSV file with automatic labels (boolean, relevant or not relevant)
    d) Command: python3 nlp/get_relevant_sentences.py data/chexpert-labels.csv data/automatic-relevance-labels.csv data/ground-truth-labels.csv
    e) Evaluate: python3 nlp/get_relevant_sentences.py data/ground-truth-labels.csv data/automatic-relevance-labels.csv data/evaluate-automatic-labels.csv
