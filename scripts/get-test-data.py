import pandas as pd 
import pprint 
import random 

all_data = pd.read_csv("results/04202020/predict/all-pairs-comparison.csv", dtype={"subject": "str", "study1": "str", "study2": "str"}).dropna()
small_data = pd.read_csv("data/dataset-small/ground-truth-comparisons-document.csv", dtype={"subject": "str", "study": "str"})

labeled_data = set()
for index, row in small_data.iterrows():
	labeled_data.add((row["subject"], row["study"]))

print(all_data.shape)
test_data = set()
for index, row in all_data.iterrows():
	if (row["subject"], row["study1"]) not in labeled_data and (row["subject"], row["study2"]) not in labeled_data:
		test_data.add((row["subject"], row["study1"], row["study2"], row["predicted_comparison"]))

pprint.pprint(random.sample(test_data, 100))

