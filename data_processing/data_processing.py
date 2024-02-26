import os
import pandas as pd
import regex
from nltk import word_tokenize, sent_tokenize
from googleapiclient.discovery import build
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException


def is_valid_sentence(text: str) -> bool:
    pattern = regex.compile(r'^\p{So}*\s*\P{So}\s*[\p{L}\d\s.,!?"\'():;-]*\s*\P{So}\s*$')
    if pattern.match(text):
        return True
    else:
        return False


def lang_detect(text: str):
    if is_valid_sentence(text):
        try:
            lang = detect(text)
            return lang
        except LangDetectException:
            return None


def detect_language(df: pd.DataFrame):
    languages = df['Comment'].apply(lang_detect)
    df['Language'] = languages
    return df


def word_count(text):
    return len(word_tokenize(text))


def sentence_count(text):
    return len(sent_tokenize(text))


def comment_len(df: pd.DataFrame) -> pd.DataFrame:
    df['WordCount'] = df['Comment'].apply(word_count)
    df['SentCount'] = df['Comment'].apply(sentence_count)
    return df
