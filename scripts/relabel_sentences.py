import numpy as np
import os 
import pandas as pd 

labeled_partial = pd.read_csv(os.path.join(os.environ['PE_PATH'], "data/ground-truth-labels.csv"))
new_sentences = pd.read_csv(os.path.join(os.environ['PE_PATH'], "data/sentences-split.csv"), header=None, names=["sentence","subject","study"])

missing_cols = {"relevant": float,
                "other condition mention (yes/no/na)": float,
                "related radiologic feature (yes/no/na)": float,
                "pulmonary edema mention (yes/no/na)": float,
                "comparison": float,
                "interval change": str,
                "expected comparison label": float,
                "severity": str}


for col, val in missing_cols.items():
    new_col = None 
    if val == float:
        new_col = pd.DataFrame(float('nan'), index=np.arange(new_sentences.shape[0]), columns=[col])
    elif val == str:
        new_col = pd.DataFrame("", index=np.arange(new_sentences.shape[0]), columns=[col])

    new_sentences = pd.concat([new_sentences, new_col[col]], axis=1)

new_sentences = new_sentences.copy()

for labeled_index in range(labeled_partial.shape[0]):
    for new_index in range(new_sentences.shape[0]):
        if labeled_partial['sentence'][labeled_index] == new_sentences['sentence'][new_index] and \
            labeled_partial['subject'][labeled_index] == new_sentences['subject'][new_index] and \
            labeled_partial['study'][labeled_index] == new_sentences['study'][new_index]:
            
             for col in missing_cols:
                new_sentences[col][new_index] = labeled_partial[col][labeled_index]

new_sentences.to_csv("ground-truth-labels-new.csv")
