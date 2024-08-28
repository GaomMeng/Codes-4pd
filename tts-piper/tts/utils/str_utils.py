# utils/str_utils.py
import os
import re

def replace_punctuation_with_space(text, punctuation_to_ignore):
    for p in punctuation_to_ignore:
        text = text.replace(p, " ")
    
    return text
