import argparse
import os
import pandas as pd

class FONT_COLORS:
    green = "40C20C"
    red = "red"
    blue = "blue"

def collect_sentences(labels, subject, study, start_index=0):
    report = []
    index = start_index
    while index < labels.shape[0] and labels['subject'][index] == subject and labels['study'][index] == study:
        report.append(labels['sentence'][index])
        index += 1

    return report, start_index, index

def write_html(labels, start, end, reports_dir, annotations):
    index = start
    annotations.write("<div>")
    subject, study = labels['subject'][index], labels['study'][index]
    annotations.write("<p>SubjectID: {}, StudyID: {}</p>".format(subject, study))
    while index < end:
        if labels['relevant'][index] == labels['ground_truth'][index] and labels['relevant'][index] == 1:
            annotations.write("<p style=\"color:{}\">{}</p>".format(FONT_COLORS.green, labels['sentence'][index]))
        
        elif labels['relevant'][index] != labels['ground_truth'][index] and labels['relevant'][index] == 1:
            annotations.write("<p style=\"color:{}\">{}</p>".format(FONT_COLORS.blue, labels['sentence'][index]))
        
        elif labels['relevant'][index] != labels['ground_truth'][index] and labels['relevant'][index] == 0:
            annotations.write("<p style=\"color:{}\">{}</p>".format(FONT_COLORS.red, labels['sentence'][index]))
        
        else:
            annotations.write("<p>{}</p>".format(labels['sentence'][index]))

        index += 1

    annotations.write("</div>")
    annotations.write("<hr>")

def annotate(labels, reports_dir, output_file):

    # Assumes that labels is nonempty (>0 rows)
    index = 0
    to_write = open(output_file, "w+")

    tp = "<span style=\"color:{}\">{}</span>".format(FONT_COLORS.green, "true positive")
    fp = "<span style=\"color:{}\">{}</span>".format(FONT_COLORS.blue, "false positive")
    fn = "<span style=\"color:{}\">{}</span>".format(FONT_COLORS.red, "false negative")
    tn = "{}".format("true negative")

    to_write.write("<p>{}&nbsp;&nbsp;{}&nbsp;&nbsp;{}&nbsp;&nbsp;{}</p>".format(tp, fp, fn, tn))
    to_write.write("<hr>")

    while index < labels.shape[0]:
        subject = labels['subject'][index]
        study = labels['study'][index]
        report, start, end = collect_sentences(labels, subject, study, start_index=index)
        write_html(labels, start, end, reports_dir, to_write)
        index = end

    to_write.close()

def main():
    parser = argparse.ArgumentParser(description='Annotate relevant sentences using HTML markup')

    # Relative paths to PE_PATH
    parser.add_argument('labels_path', type=str, help='Path to file with relevance labels')
    parser.add_argument('reports_dir', type=str, help='Path to folder with radiology reports')
    parser.add_argument('output_path', type=str, help='Path to HTML file where annotations should be written')
    args = parser.parse_args()

    labels_path = os.path.join(os.environ['PE_PATH'], args.labels_path)
    reports_dir = os.path.join(os.environ['PE_PATH'], args.reports_dir)
    output_file = os.path.join(os.environ['PE_PATH'], args.output_path)

    labels = pd.read_csv(labels_path)

    annotate(labels, reports_dir, output_file)

if __name__ == "__main__":
    main()


