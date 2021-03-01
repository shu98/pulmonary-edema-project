import math 
import pandas as pd 

# document_df = pd.read_csv("results/04122020/comparisons-document-all.csv", dtype={'study': 'str', 'subject': 'str'})
# severity_df = pd.read_csv("results/03302020/automatic-document-labels-severity.csv", dtype={'study': 'str', 'subject': 'str'})
# severity_df = severity_df.loc[document_df.index, :]

# final_df = pd.concat([severity_df['subject'], severity_df['study'], -severity_df['comparison'], severity_df['severity']], axis=1)
# final_df.to_csv("results/04122020/automatic-document-labels-severity.csv", columns=['subject', 'study', 'comparison', 'severity'])


# severity = pd.read_csv("results/analysis/comparison-and-severity-labels.csv", dtype={'study': 'str', 'subject': 'str'})

# # Cindy train/test/val data 
# train = pd.concat([pd.read_csv("results/04202020/train/comparison-cindy-agree.csv", dtype={'study': 'str', 'subject': 'str'}),
# 				pd.read_csv("results/04202020/train/comparison-cindy-diff.csv", dtype={'study': 'str', 'subject': 'str'})])
# val = pd.concat([pd.read_csv("results/04202020/val/comparison-cindy-agree.csv", dtype={'study': 'str', 'subject': 'str'}),
# 				pd.read_csv("results/04202020/val/comparison-cindy-diff.csv", dtype={'study': 'str', 'subject': 'str'})])
# test = pd.concat([pd.read_csv("results/04202020/test/comparison-cindy-agree.csv", dtype={'study': 'str', 'subject': 'str'}),
# 				pd.read_csv("results/04202020/test/comparison-cindy-diff.csv", dtype={'study': 'str', 'subject': 'str'})])
# cindy = pd.concat([train, test, val])


# print("# of rows - Cindy:", cindy.shape[0])
# print("# of rows - severity:", severity.shape[0])

# cindy_pairs = set()
# for index, row in cindy.iterrows():
# 	cindy_pairs.add((row['study1'], row['study2']))

# severity_pairs = set()
# for index, row in severity.iterrows():
# 	severity_pairs.add((row['previous_study'], row['next_study']))

# overlap = set()
# for pair in cindy_pairs:
# 	if pair in severity_pairs or (pair[1], pair[0]) in severity_pairs:
# 		overlap.add(pair)
# 	else:
# 		print(pair)

# print(len(overlap))

comparisons = pd.read_csv("results/04112020/comparisons-labeled-from-automatic-small.csv", dtype={'study': 'str', 'subject': 'str'})
sentence_length = 0
sentence_median = []
for index, row in comparisons.iterrows():
	sentence_length += len(row['sentence'].split())
	sentence_median.append(len(row['sentence'].split()))

sentence_median.sort()

chexpert_labels = pd.read_csv("data/dataset-small/chexpert-labels-small.csv")
observations = 0
observations_pe = 0
total = 0
total_pe = 0
for index, row in chexpert_labels.iterrows():
	should_count = 0
	for col in chexpert_labels.columns:
		if col == "Reports" or col == 'No Finding':
			continue 
		elif not math.isnan(row[col]):
			observations += 1 
			should_count += 1

			if comparisons['relevant'][index] == 1.0:
				observations_pe += 1 

	if should_count == 0.0:
		continue 
	else:
		total += 1 
		if comparisons['relevant'][index] == 1.0:
			total_pe += 1 

print("Total number of sentences:", comparisons.shape[0])
print("Avg sentence length:", sentence_length / comparisons.shape[0])
print("Median sentence length:", sentence_median[len(sentence_median)//2])
print("Sentences relevant to pulmonary edema:", 272)
print("Observations mentioned per sentence:", observations / total)
print("Observations mentioned per relevant sentence:", observations_pe / total_pe)



