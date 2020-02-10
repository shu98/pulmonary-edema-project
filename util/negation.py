import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import re

# Set up negex path
negex_path = os.path.join(os.environ['PE_PATH'], 'negex/negex.python/')
sys.path.insert(0, negex_path)
import negex

def is_positive(sentence, keywords, mode='last'):
    """
    sentence    sentence to check
    keywords    list of keywords to look for in sentence 
    mode        aggregation method when more than one keyword is present
                'first'     return flag for first keyword found
                'last'      return flag for last keyword found 
                'majority'  return T/F depending on relative number of keywords that are positive/negative
                'sum'       return T if at least one keyword is positive

    Returns a flag indicating the presence/absence of keywords in a sentence
        - True  1+ keywords present in sentence and assertion is positive
        - False 1+ keywords present in sentence and there is negation 
        - None  no keywords present 
    """
    flag = None
    num_pos, num_neg = 0, 0
    for key in keywords:
        # Open negex rules 
        rfile = open(negex_path+'negex_triggers.txt', 'r')
        irules = negex.sortRules(rfile.readlines())
        st = sentence.lower()

        # Search for keyword in sentence 
        match_obj = re.search(key, st)

        # If keyword found, check whether it is negated 
        if match_obj is not None:
            tagger = negex.negTagger(sentence = st, phrases = [st[match_obj.start():match_obj.end()]], rules = irules, negP = False)
            if tagger.getNegationFlag() == 'affirmed': 
                flag = True
                num_pos += 1 
            if tagger.getNegationFlag() == 'negated': 
                flag = False
                num_neg += 1

            # If mode is 'first', return immediately after first keyword is detected 
            if mode == 'first': return flag

            # If mode is 'sum', return immediately after first positive keyword is detected 
            if mode == 'sum' and flag is True: return flag 

    # If mode is 'majority', return T if more positive keywords than negative, else return F
    if mode == 'majority':
        if num_pos >= num_neg: return True 
        else: return False 

    return flag