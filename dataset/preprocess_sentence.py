import re

def substitute_modifiers(sentence):
    new_sentence = sentence.lower() 
    modifiers = {'mild(-| )to(-| )moderate': 'mild-to-moderate', 'moderate(-| )to(-| )severe': 'moderate-to-severe'}
    for mod, sub in modifiers.items():
        new_sentence = re.sub(mod, sub, new_sentence)

    return new_sentence

def remove_conjadv(sentence):
    new_sentence = sentence.lower()
    conjadv = {'however(,|) ', 'although(,|) ', 'nevertheless(,|)', 'though(,|) '} 
    for word in conjadv:
        new_sentence = re.sub(word, '', new_sentence)  

    return new_sentence