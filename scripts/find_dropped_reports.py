import pandas as pd 

split_sentences = pd.read_csv("data/03182020/sentences-split-all.csv")
document_labels = pd.read_csv("results/03182020/comparisons-document-all.csv")

split_sentence_studies = set()
for index, row in split_sentences.iterrows():
	split_sentence_studies.add(row['study'])

document_label_studies = set()
for index, row in document_labels.iterrows():
	document_label_studies.add(row['study'])

print(len(split_sentence_studies))
print(len(document_label_studies))