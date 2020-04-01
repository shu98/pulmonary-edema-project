import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import argparse
import math
import pandas as pd
import re

from dataset.organize_reports import sort_by_date, get_metadata, get_datetime
from dataset.reformat_reports import reformat
from nlp.get_comparisons_document import aggregate_sentences, resolve_disagreements
from util.document import collect_sentences 

class FONT_COLORS:
    # green = "#40C20C"
    green = "#CCEDD2"
    red = "red"
    lightred = "#f5c3bc"
    blue = "blue"
    # yellow = "#F8DB22"
    yellow = "#f6da63"
    purple = "purple"
    orange = "orange"
    gray = "#DDDDDD"
    black = "black"
    none = "none"

class HTML_CLASSES:
    better = "better"
    worse = "worse"
    same = "same"
    no_comparison = "no_comparison"
    keywords = "keywords"
    note = "note"
    not_relevant = "not_relevant"
    disagreement = "disagreement"
    hide = "hide"

def parse_keywords(string):
    keywords = string.split("//")
    formatted_words = []
    for word in keywords:
        pair = word.split("/")
        formatted_words.append(pair[0])

    return formatted_words

def create_html_skeleton(head, body):
    return "<html>\
                <head>{}</head> \
                <body>{}</body>\
            </html>".format(head, body)

def create_css_segment(content):
    return "<style type=\"text/css\">{}</style>".format(content)

def css_string():
    css_elements = [
        ".{} {{ background-color: {}; }}".format(HTML_CLASSES.better, FONT_COLORS.green),
        ".{} {{ background-color: {}; }}".format(HTML_CLASSES.worse, FONT_COLORS.lightred),
        ".{} {{ background-color: {}; }}".format(HTML_CLASSES.same, FONT_COLORS.yellow),
        ".{} {{ background-color: {}; }}".format(HTML_CLASSES.no_comparison, FONT_COLORS.gray),
        ".{} {{ background-color: {}; }}".format(HTML_CLASSES.not_relevant, FONT_COLORS.none),
        ".{} {{ color: {}; }}".format(HTML_CLASSES.keywords, FONT_COLORS.blue),
        ".{} {{ color: {}; }}".format(HTML_CLASSES.note, FONT_COLORS.red),
        ".{} {{ display: none; }}".format(HTML_CLASSES.hide)
    ]

    return "\n".join(css_elements)

def read_css_file(filename):
    return "<link rel=\"stylesheet\" type=\"text/css\" href=\"{}\"/>".format(filename)

def html_element(tag, class_, text):
    return "<{} class=\"{}\">{}</{}>".format(tag, class_, text, tag)

def newline_to_ptag(report):
    report_split = report.split("\n")
    for index, section in enumerate(report_split):
        report_split[index] = "<p>{}</p>".format(report_split[index])

    return "".join(report_split)

def markup_report(labels, start, end, reports_dir, true_labels, metadata):
    index = start 
    subject, study = labels['subject'][index], labels['study'][index]

    document_label = aggregate_sentences(labels, start, end)
    annotations = []
    if document_label == float('inf'):
        document_label = resolve_disagreements(labels, start, end)

    if document_label == 1.0:
        document_label = "better"
    elif document_label == -1.0:
        document_label = "worse"
    elif document_label == 0.0:
        document_label = "same"

    should_show = document_label == float('inf')
    if should_show:
        annotations.append("<div>")
    else:
        annotations.append("<div class=\"hide\">")
   
    study_date = get_datetime(metadata['study_date'][study], metadata['study_time'][study])
    annotations.append("<p>SubjectID: {}, StudyID: {}, Date: {}, {}</p>".format(subject, study, study_date, html_element("span", HTML_CLASSES.note, "Comparison: {}".format(document_label))))

    report = reformat(subject, study, reports_dir)
    while index < end:
        labeled_sentence = []
        if labels['relevant'][index] == 1:
            need_keywords = True 
            if labels['predicted'][index] == 1:
                labeled_sentence.append(html_element("span", HTML_CLASSES.better, labels['sentence'][index]))
            elif labels['predicted'][index] == -1:
                labeled_sentence.append(html_element("span", HTML_CLASSES.worse, labels['sentence'][index]))
            elif labels['predicted'][index] == 0:
                labeled_sentence.append(html_element("span", HTML_CLASSES.same, labels['sentence'][index]))
            else:
                labeled_sentence.append(html_element("span", HTML_CLASSES.no_comparison, labels['sentence'][index]))
                need_keywords = False

            if need_keywords and labels['relevant'][index]:
                keywords = parse_keywords(labels['keywords'][index]) 
                labeled_sentence.append("&nbsp;&nbsp; {}".format( 
                                        html_element("span", HTML_CLASSES.keywords, "Keywords: {}".format(", ".join(keywords)) 
                                    )))

        else: 
                labeled_sentence.append(html_element("span", HTML_CLASSES.not_relevant, labels['sentence'][index]))

        if true_labels:
            if labels['ground_truth'][index] != labels['predicted'][index] and not (math.isnan(labels['ground_truth'][index]) and math.isnan(labels['predicted'][index])):
                labeled_sentence.append("&nbsp;&nbsp; {}".format(html_element("span", HTML_CLASSES.note, "True label: {}".format(labels['ground_truth'][index]))))

        report = report.replace(labels['sentence'][index], "".join(labeled_sentence), 1)
        index += 1

    report = newline_to_ptag(report)
    annotations.append(report)
    annotations.append("</div>")
    
    if should_show: annotations.append("<hr>")
    else: annotations.append("<hr class=\"hide\">")

    return "\n".join(annotations)

def html_string(labels, start, end, reports_dir):
    index = start 
    subject, study = labels['subject'][index], labels['study'][index]

    annotations = []
    annotations.append("<div>")
    annotations.append("<p>SubjectID: {}, StudyID: {}, Date: {}</p>".format(subject, study, get_datetime(metadata['study_date']['study'], metadata['study_time']['study'])))
    while index < end:
        annotations.append("<p>")
        if labels['relevant'][index] == 1:
            need_keywords = True 
            if labels['predicted'][index] == 1:
                annotations.append(html_element("span", HTML_CLASSES.better, labels['sentence'][index]))
            elif labels['predicted'][index] == -1:
                annotations.append(html_element("span", HTML_CLASSES.worse, labels['sentence'][index]))
            elif labels['predicted'][index] == 0:
                annotations.append(html_element("span", HTML_CLASSES.same, labels['sentence'][index]))
            else:
                annotations.append(html_element("span", HTML_CLASSES.no_comparison, labels['sentence'][index]))
                need_keywords = False

            if need_keywords and labels['ground_truth_relevant'][index] == labels['relevant'][index]:
                keywords = parse_keywords(labels['keywords'][index]) 
                annotations.append("&nbsp;&nbsp; {}".format( 
                                        html_element("span", HTML_CLASSES.keywords, "Keywords: {}".format(", ".join(keywords)) 
                                    )))
        else: 
            annotations.append(html_element("span", HTML_CLASSES.not_relevant, labels['sentence'][index]))

        if labels['ground_truth'][index] != labels['predicted'][index] and not (math.isnan(labels['ground_truth'][index]) and math.isnan(labels['predicted'][index])):
            annotations.append("&nbsp;&nbsp; {}".format(html_element("span", HTML_CLASSES.note, "True label: {}".format(labels['ground_truth'][index]))))

        annotations.append("</p>")
        index += 1

    annotations.append("</div>")
    annotations.append("<hr>")
    return "\n".join(annotations)

def annotate(labels, metadata, reports_dir, output_file, true_labels=False):

    # Assumes that labels is nonempty (>0 rows)
    index = 0
    to_write = open(output_file, "w+")

    body = []
    body.append("<p>")
    body.append(html_element("span", HTML_CLASSES.better, "better"))
    body.append("&nbsp;&nbsp;")
    body.append(html_element("span", HTML_CLASSES.worse, "worse"))
    body.append("&nbsp;&nbsp;")
    body.append(html_element("span", HTML_CLASSES.same, "same"))
    body.append("&nbsp;&nbsp;")
    body.append(html_element("span", HTML_CLASSES.no_comparison, "no comparison"))
    body.append("</p>")
    body.append("<hr>")

    sorted_by_subject = sort_by_date(metadata)

    reports = {}
    while index < labels.shape[0]:
        subject = labels['subject'][index]
        study = labels['study'][index]
        if subject not in reports:
            reports[subject] = dict()

        report, start, end = collect_sentences(labels, subject, study, start_index=index)
        html_element_list = markup_report(labels, start, end, reports_dir, true_labels, metadata)
        if study in reports[subject]:
            raise Exception("Duplicate study for subject. This shouldn't happen.")

        reports[subject][study] = html_element_list
        # body.append(markup_report(labels, start, end, reports_dir, true_labels))
        index = end

    for subject, series in sorted_by_subject.items():
        for study_sorted, date in series:
            body.append(reports[subject][study_sorted])

    css = create_css_segment(css_string())
    # css = read_css_file("style.css")
    html = "\n".join(body)

    to_write.write(create_html_skeleton(css, html))
    to_write.close()

def main():
    parser = argparse.ArgumentParser(description='Annotate relevant sentences using HTML markup')

    # Relative paths to PE_PATH
    parser.add_argument('comparison_labels_path', type=str, help='Path to file with comparison labels')
    parser.add_argument('metadata_path', type=str, help='Path to file with metadata')
    parser.add_argument('reports_dir', type=str, help='Path to folder with radiology reports')
    parser.add_argument('output_path', type=str, help='Path to HTML file where annotations should be written')
    args = parser.parse_args()

    labels_path = os.path.join(os.environ['PE_PATH'], args.comparison_labels_path) 
    reports_dir = os.path.join(os.environ['PE_PATH'], args.reports_dir)
    output_file = os.path.join(os.environ['PE_PATH'], args.output_path)
    metadata_file = os.path.join(os.environ['PE_PATH'], args.metadata_path)
    
    labels = pd.read_csv(labels_path, index_col=0, dtype={"study": "str", "subject": "str"})
    metadata = get_metadata(metadata_file)

    annotate(labels, metadata, reports_dir, output_file, true_labels=False)

if __name__ == "__main__":
    main()


