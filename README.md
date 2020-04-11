Requires
- Negex 
- Scispacy

1. Split the FINDING and IMPRESSIONS sections of radiology reports into individual sentences.
	a) Create metadata file: python3 dataset/process_files.py data/hf-patients.csv data/dicom-metadata.csv data/hf-metadata.csv
	b) Split radiology reports: python dataset/split_reports.py /home/shu98/radiology/files data/hf-metadata.csv data/03182020/sentences-split-all.csv
	c) Environment should be pe

2. Run CheXpert labeler on individual sentences. 
    a) Input: CSV file with one column of sentences
    b) To create input file, run get_reports_for_chexpert.py on the split sentences: python scripts/get_reports_for_chexpert.py data/03182020/sentences-split-all.csv data/03182020/sentences-all-for-chexpert.csv
    c) Command: python3 label.py --reports_path /path/to/chexpert/file
    d) Environment should be chexpert-mimic-cxr
    e) Run script from mimic-cxr/txt/chexpert/chexpert-labeler
    f) Output: labeled_reports.csv (in the same folder as script)

3. Identify sentences relevant to pulmonary edema using get_relevant_sentences.py
	a) Set PE_PATH environment variable to root of codebase 
    b) Input: output file of CheXpert labeler, filename to write results 
    c) Output: CSV file with automatic labels (boolean, relevant or not relevant)
    d) For dataset with ground truth labels: python3 nlp/get_relevant_sentences.py data/chexpert-labels.csv results/automatic-relevance-labels.csv data/ground-truth-labels.csv
    e) Evaluate: python3 nlp/get_relevant_sentences.py data/ground-truth-labels.csv data/automatic-relevance-labels.csv data/evaluate-automatic-labels.csv
    f) For dataset without ground truth labels: python3 nlp/get_relevant_sentences.py data/03182020/chexpert-labels-all.csv results/03182020/automatic-relevance-labels-all.csv data/03182020/sentences-split-all.csv

4. Assign comparison labels to relevant sentences using get_comparisons.py
	a) PE_PATH environment variable should have been set in previous step
	b) Input: CSV file with relevance labels
	c) Output: CSV file with comparison labels 
	d) Command: python3 nlp/get_comparisons.py results/03182020/automatic-relevance-labels-all.csv results/03182020/comparisons-labeled-from-automatic-all.csv

5. Get document-level comparison labels using get_comparisons_document.py
	a) PE_PATH environment variable should have been set in previous step
	b) Input: CSV file with sentence-level comparison labels 
	c) Output: CSV file with document-level comparisons 
	d) Command: python nlp/get_comparisons_document.py results/03182020/comparisons-labeled-from-automatic-all.csv results/03182020/comparisons-document-all.csv

6. Annotate documents in HTML 
	a) Input: sentence-level relegance labels and comparison labels, folder with radiology reports
	b) Output: HTML file with sentences annotated based on model predictions 
	c) Command: python scripts/annotate_documents_sorted.py results/03182020/comparisons-labeled-from-automatic-all.csv data/hf-metadata.csv /home/shu98/radiology/files results/03182020/comparison-sentences-annotations-all.html