import os
import re
import csv
import numpy as np
import pandas as pd 
from shutil import copyfile
import sys

# current_path = os.path.dirname(os.path.abspath(__file__))
# negex_path = os.path.join(current_path, 'negex/negex.python/')
negex_path = os.path.join(os.environ['PE_PATH'], 'negex/negex.python/')
sys.path.insert(0, negex_path)
import negex

class WordMatch(object):
    """Word matching in a sentence with negation detection
    """

    def __init__(self, sentence, words=[], case_insensitive=True):
        self.sentence = sentence
        self.words = words
        self.case_insensitive = case_insensitive
        self.words_mentioned = {}
        self.words_mentioned_positive = {}
        self.words_mentioned_negative = {}

    def mention(self):
        """Determine which words in the given list are mentioned 
        in the given sentence.

        Returns:
            A dictionary with words and whether or not they're mentioned.  
        """
        if self.case_insensitive:
            lower_sentence = self.sentence.lower()
            for word in self.words:
                if word.lower() in lower_sentence:
                    self.words_mentioned[word] = True
                else:
                    self.words_mentioned[word] = False
        else:
            for word in self.words:
                if word in self.sentence:
                    self.words_mentioned[word] = True
                else:
                    self.words_mentioned[word] = False

        return self.words_mentioned

    def mention_positive(self):
        """Determine if the mention words are affirmed (positive)

        Returns:
            A dictionary with words and whether or not they're mentioned
            and affirmed
        """
        self.mention()
        rfile = open(negex_path+'negex_triggers.txt', 'r')
        irules = negex.sortRules(rfile.readlines())
        for key in self.words_mentioned:
            if not self.words_mentioned[key]:
                self.words_mentioned_positive[key] = False
            else:
                tagger = negex.negTagger(sentence = self.sentence, 
                                         phrases = [key], rules = irules,
                                         negP = False)
                if tagger.getNegationFlag() == 'affirmed':
                    self.words_mentioned_positive[key] = True
                else:
                    self.words_mentioned_positive[key] = False

        return self.words_mentioned_positive

    def mention_negative(self):
        """Determine if the mention words are negated (negative)

        Returns:
            A dictionary with words and whether or not they're mentioned
            and negated
        """
        self.mention()
        rfile = open(negex_path+'negex_triggers.txt', 'r')
        irules = negex.sortRules(rfile.readlines())
        for key in self.words_mentioned:
            if not self.words_mentioned[key]:
                self.words_mentioned_negative[key] = False
            else:
                tagger = negex.negTagger(sentence = self.sentence, 
                                         phrases = [key], rules = irules,
                                         negP = False)
                if tagger.getNegationFlag() == 'negated':
                    self.words_mentioned_negative[key] = True
                else:
                    self.words_mentioned_negative[key] = False

        return self.words_mentioned_negative 


def main():

    report_dir = '/home/shu98/radiology/files'

    HF_reports = []
    reports_to_label = ''
    csv_dir = '/home/shu98/pe-data/codebase'
    HF_label_path = os.path.join(csv_dir, 'results/03192020/comparisons-document-all.csv')
    documents = pd.read_csv(HF_label_path, dtype={"study": "str", "subject": "str"})

    level0_words = ['no pulmonary edema', 'no vascular congestion',\
                    'no fluid overload', 'no acute cardiopulmonary process']
    level0_words_n = ['pulmonary edema', 'vascular congestion',\
                      'fluid overload', 'acute cardiopulmonary process']
    level1_words = ['cephalization', 'mild pulmonary vascular congestion',\
                    'mild hilar engorgement', 'mild vascular plethora']
    level2_words = ['interstitial opacities', 'kerley',\
                    'interstitial edema', 'interstitial thickening'\
                    'interstitial pulmonary edema', 'interstitial marking'\
                    'interstitial abnormality', 'interstitial abnormalities'\
                    'interstitial process']
    level3_words = ['alveolar infiltrates', 'severe pulmonary edema',\
                    'perihilar infiltrates', 'hilar infiltrates',\
                    'parenchymal opacities', 'alveolar opacities',\
                    'ill defined opacities', 'ill-defined opacities'\
                    'patchy opacities']

    severity = pd.DataFrame(float('nan'), index=np.arange(documents.shape[0]), columns=['severity']) 
    for index, row in documents.iterrows():
        report_path = os.path.join(report_dir, "p{}".format(row['subject']), "s{}.txt".format(row['study']))
        file = open(report_path, 'r')
        report = file.read()
        sentences = re.split('\.|\:', report)
        label = float('nan')
        for sentence in sentences:
            word_match = WordMatch(sentence, level0_words)
            level0_mentioned = word_match.mention()
            for k in level0_mentioned:
                if level0_mentioned[k]:
                    label = 0
                    keyword = k
            word_match = WordMatch(sentence, level0_words_n)
            level0_mentioned = word_match.mention_negative()
            for k in level0_mentioned:
                if level0_mentioned[k]:
                    label = 0
                    keyword = k             
            word_match = WordMatch(sentence, level1_words)
            level1_mentioned = word_match.mention_positive()
            for k in level1_mentioned:
                if level1_mentioned[k]:
                    label = 1
                    keyword = k
            word_match = WordMatch(sentence, level2_words)
            level2_mentioned = word_match.mention_positive()
            for k in level2_mentioned:
                if level2_mentioned[k]:
                    label = 2
                    keyword = k
            word_match = WordMatch(sentence, level3_words)
            level3_mentioned = word_match.mention_positive()
            for k in level3_mentioned:
                if level3_mentioned[k]:
                    label = 3
                    keyword = k 

        severity['severity'][index] = label

    documents = pd.concat([documents['subject'], documents['study'], documents['comparisons'], severity], axis=1)
    document.to_csv(os.path.join(csv_dir, 'results/03192020/automatic-document-labels.csv'))


if __name__ == '__main__': 
    main()

    