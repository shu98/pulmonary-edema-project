import numpy as np 
import pandas as pd 
import sklearn.metrics as m

# predicted = pd.read_csv("results/analysis/test/predicted_comparisons.csv")
# actual = pd.read_csv("results/analysis/test/seth_labels.csv")

# COMPARISON_TO_LABEL = {'better': -1.0, 'worse': 1.0, 'no change': 0.0, 'no comparison': 100}

# ########## COMBINE PREDICTED AND ACTUAL LABELS ##########

# actual_pairs = set()
# for idx, row in actual.iterrows(): 
#     actual_pairs.add((row['Subject_ID'], row['Study1_ID'], row['Study2_ID']))

# predicted_keep = []
# for idx, row in predicted.iterrows():
#     if (row['subject'], row['study1'], row['study2']) in actual_pairs:
#         predicted_keep.append({'subject': row['subject'],
#                                'study1': row['study1'],
#                                'study2': row['study2'],
#                                'comparison': row['predicted_comparison']})
    
#     elif (row['subject'], row['study2'], row['study1']) in actual_pairs:
#         predicted_keep.append({'subject': row['subject'],
#                                'study1': row['study2'],
#                                'study2': row['study1'],
#                                'comparison': -row['predicted_comparison']})

# final = actual.copy() 
# final = pd.concat([final, 
#                    pd.Series(100, index=np.arange(final.shape[0]), name='predicted', dtype=int),
#                    pd.Series(100, index=np.arange(final.shape[0]), name='actual', dtype=int)], axis=1)
# final.columns = map(str.lower, final.columns)

# for idx, row in final.iterrows():
#     for pair in predicted_keep:
#         if pair['subject'] == row['subject_id'] and \
#             pair['study1'] == row['study1_id'] and \
#             pair['study2'] == row['study2_id']:
            
#             final.at[idx, 'predicted'] = 100 if np.isnan(pair['comparison']) else pair['comparison']
#             final.at[idx, 'actual'] = COMPARISON_TO_LABEL[row['review_result']]

# final = final.drop(columns=['review_result'])
# final.to_csv("results/analysis/test/test_data_final.csv", index=False)

final = pd.read_csv("results/analysis/test/test_data_final.csv", index_col=None)

######### CONFUSION MATRIX ######### 
stats = pd.DataFrame([[0] * 4 for i in range(4)],
                     index=['pred_better', 'pred_same', 'pred_worse', 'pred_none'],
                     columns=['true_better', 'true_same', 'true_worse', 'true_none'])

for idx, row in final.iterrows():
    if row['predicted'] == 1.0 and row['actual'] == 1.0:
        stats['true_worse']['pred_worse'] += 1
    
    elif row['predicted'] == 1.0 and row['actual'] == 0.0:
        stats['true_same']['pred_worse'] += 1
    
    elif row['predicted'] == 1.0 and row['actual'] == -1.0:
        stats['true_better']['pred_worse'] += 1
    
    elif row['predicted'] == 1.0 and row['actual'] == 100:
        stats['true_none']['pred_worse'] += 1

    elif row['predicted'] == 0.0 and row['actual'] == 1.0:
        stats['true_worse']['pred_same'] += 1

    elif row['predicted'] == 0.0 and row['actual'] == 0.0:
        stats['true_same']['pred_same'] += 1
    
    elif row['predicted'] == 0.0 and row['actual'] == -1.0:
        stats['true_better']['pred_same'] += 1
    
    elif row['predicted'] == 0.0 and row['actual'] == 100:
        stats['true_none']['pred_same'] += 1
    
    elif row['predicted'] == -1.0 and row['actual'] == 1.0:
        stats['true_worse']['pred_better'] += 1

    elif row['predicted'] == -1.0 and row['actual'] == 0.0:
        stats['true_same']['pred_better'] += 1

    elif row['predicted'] == -1.0 and row['actual'] == -1.0:
        stats['true_better']['pred_better'] += 1
    
    elif row['predicted'] == -1.0 and row['actual'] == 100:
        stats['true_none']['pred_better'] += 1

print(stats)

########## ACCURACY ##########
def accuracy(data):
    correct = 0
    total = data.shape[0]
    for idx, row in data.iterrows():
        if row['predicted'] == row['actual']:
            correct += 1 
    return correct / total 

print("ACCURACY:", accuracy(final))

######### PRECISION & RECALL ######### 
def precision_recall_class(stats, label):
    correct = stats['true_' + label]['pred_' + label]

    # precision: tp / (tp + fp)
    predicted = stats.loc['pred_' + label].sum()
    precision = correct / predicted

    # recall: tp / (tp + fn)
    actual = stats['true_' + label].sum()
    recall = correct / actual 

    return precision, recall

def precision_recall(stats, label='all'):
    if label != 'all':
        return precision_recall_class(stats, label)
    
    elif label == 'all':
        worse = precision_recall_class(stats, 'worse')
        same = precision_recall_class(stats, 'same')
        better = precision_recall_class(stats, 'better')

        total = stats.values.sum()
        total_worse = stats['true_worse'].sum()
        total_same = stats['true_same'].sum()
        total_better = stats['true_better'].sum()

        precision = worse[0] * (total_worse / total) \
                    + same[0] * (total_same / total) \
                    + better[0] * (total_better / total)

        recall = worse[1] * (total_worse / total) \
                 + same[1] * (total_same / total) \
                 + better[1] * (total_better / total)  
        
        return precision, recall 

    return None, None

print(m.classification_report(final['actual'], final['predicted'], digits=5))






