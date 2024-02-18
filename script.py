import os
import pandas as pd
from nltk import word_tokenize, sent_tokenize
from googleapiclient.discovery import build
from langdetect import detect, lang_detect_exception

# import numpy as np
# import rege
# from transformers import AutoModelForSequenceClassification
# from transformers import AutoTokenizer
# from scipy.special import softmax


API_KEY = os.getenv('GOOGLE_API_KEY')


def is_valid_sentence(text):
    pattern = regex.compile(r'^\p{So}*\s*\P{So}\s*[\p{L}\d\s.,!?"\'():;-]*\s*\P{So}\s*$')
    if pattern.match(text):
        return True
    else:
        return False


def lang_detect(comment):
    if is_valid_sentence(comment):
        try:
            lang = detect(comment)
            return lang
        except lang_detect_exception.LangDetectException:
            return None


def comment_len(comments):
    for item in comments['Comment'].tolist():
        comments.loc[comments['Comment'] == item, 'WordCount'] = int(len(word_tokenize(item)))
        comments.loc[comments['Comment'] == item, 'SentCount'] = int(len(sent_tokenize(item)))
    return comments


