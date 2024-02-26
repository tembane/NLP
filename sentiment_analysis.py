import os
import pandas as pd
import numpy as np
from nltk import word_tokenize, sent_tokenize
from googleapiclient.discovery import build
from langdetect import detect, lang_detect_exception
import regex
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from scipy.special import softmax


def sentiment_analysis(comments, model_name):
    eng_comments = comments.loc[comments.Language == 'en', 'Comment'].tolist()
    vectorizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    emotions = ['negative', 'neutral', 'positive']
    for comment in eng_comments:
        vectorized_comments = vectorizer(comment, return_tensors='pt')
        if len(vectorized_comments[0]) > 514:
            continue
        output = model(**vectorized_comments)
        scorse = output[0][0].detach().numpy()
        predict_value = softmax(scorse).tolist()
        index = predict_value.index(max(predict_value))
        comments.loc[
            comments['Comment'] == comment,
            'Predict'
        ] = np.round(float(max(predict_value)), 2)
        comments.loc[
            comments['Comment'] == comment,
            'Emotional'
        ] = emotions[index]
    return comments