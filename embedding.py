import numpy as np
import pandas as pd
from nltk import sent_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA

sbert_model = SentenceTransformer('sentence-transformers/models/MiniLM-SBERT')
labse_model = SentenceTransformer('sentence-transformers/models/LA-BERT')


def get_embedding(comments):
    perf = comments.ChannelName.unique().tolist()
    for index, item in enumerate(perf):
        print(f'{index + 1}. {item}')
    if len(perf) > 1:
        numb = int(input('Enter channel title number:'))
        channel_name = perf[numb-1]
    else:
        channel_name = perf[0]
    emotional_colors = ['positive', 'negative', 'neutral']
    numb = int(input('Choose sentiment:\n1. positive\n2. negative\n3. neutral\n4. all'))
    if numb != 4:
        sentiment = emotional_colors[numb - 1]
        sample = comments.loc[
            (comments.Language == 'en') & (comments.ChannelName == channel_name) & (comments.Emotional == sentiment),
            'Comment'
        ].tolist()
    else:
        sample = comments.loc[(comments.Language == 'en') & (comments.ChannelName == channel_name), 'Comment'].tolist()
    text = []
    for i in sample:
        text.extend(sent_tokenize(i))
    choose = int(input('Choose encoder type:\n1. SBERT\n2. LABSE'))
    if choose == 1:
        embeddings = sbert_model.encode(sentences=text, convert_to_tensor=True)
    elif choose == 2:
        embeddings = labse_model.encode(sentences=text)
    pca = PCA(n_components=2)
    embeddings = pca.fit_transform(embeddings)
    return embeddings
