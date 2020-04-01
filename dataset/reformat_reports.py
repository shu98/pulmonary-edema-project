import argparse
import re 

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

def reformat(subject, study, report_folder):

    headings = set([
        "NOTIFICATION",
        "INDICATION",
        "RECOMMENDATION",
        "TECHNIQUE",
        "FINAL REPORT",
        "COMPARISON",
        "WET READ",
        "FINDINGS",
        "IMPRESSION",
        "CONCLUSION",
        "EXAMINATION",
        "HISTORY"])

    with open("{}/p{}/s{}.txt".format(report_folder, subject, study), "r") as f:
        entire_report = f.readlines()
        lines = " ".join(entire_report).replace("\n", " ")
        lines = re.sub('\s+', ' ', lines)
        for h in headings:
            if h.upper().strip() == "FINAL REPORT": lines = lines.replace(h, "\n\n{}\n\n".format(h))
            else: lines = lines.replace(h, "\n\n{}".format(h))

        return lines  

def main():
    parser = argparse.ArgumentParser(description='Reformat reports to remove unnecessary newlines')
    parser.add_argument('data_folder_path', type=str, help='Path to folder with radiology reports')
    args = parser.parse_args()

    data_folder = os.path.join(os.environ['PE_PATH'], args.data_folder_path)
    print(reformat("10569306", "55919708", data_folder))  

if __name__ == "__main__":
    main()


