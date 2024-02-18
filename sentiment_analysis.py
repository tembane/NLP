


def sentiment_analysis(comments):
    eng_comments = comments.loc[comments.Language == 'en', 'Comment'].tolist()

    model_name = f"cardiffnlp/twitter-roberta-base-sentiment-latest"
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