import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import math
import pandas as pd 

file = os.path.join(os.environ['PE_PATH'], "data/correlation-filtered/test-po2.csv")
comparisons = pd.read_csv(file, dtype={'study': 'str', 'subject': 'str'})
count_with_all = 0
severity_agree = 0
severity_lab_total = 0
comparison_agree = 0
comparison_lab_total = 0

all_severity_agree = 0
all_comparison_agree = 0

for index, row in comparisons.iterrows():
	if not math.isnan(row['comparison']) and not math.isnan(row['severity_comparison']) and not math.isnan(row['lab_comparison']):
		count_with_all += 1 
		all_severity_agree += (row['severity_comparison'] == row['lab_comparison'])
		all_comparison_agree += (row['comparison'] == row['lab_comparison'])

	if not math.isnan(row['severity_comparison']) and not math.isnan(row['lab_comparison']):
		severity_lab_total += 1
		severity_agree += (row['severity_comparison'] == row['lab_comparison'])

	if not math.isnan(row['comparison']) and not math.isnan(row['lab_comparison']):
		comparison_lab_total += 1 
		comparison_agree += (row['comparison'] == row['lab_comparison'])


print("######PCO2######")
print("Total with all values:", count_with_all)
print("Severity agree when all three agree:", all_severity_agree)
print("Comparison agree when all three agree:", all_comparison_agree)
print("Severity agree: {}/{}".format(severity_agree, severity_lab_total))
print("Comparison agree: {}/{}".format(comparison_agree, comparison_lab_total))

