import seaborn as sns
import matplotlib.pyplot as plt


def lang_visual(comments, comment_count):
    plt.title(f'Гистограмма кол-ва комментариев на различных языках для {comment_count} последних коментариев')
    sns.countplot(x='Language', data=comments)


def visual_word_count(comments, comment_count):
    plt.title(
        f'Гистограмма кол-ва слов в предложении для {comment_count} последних коментариев'
    )
    sns.histplot(comments['WordCount'])


def visual_sent_count(comments, comment_count):
    plt.title(
        f'Гистограмма кол-ва предложении в комментарии для {comment_count} последних коментариев'
    )
    sns.histplot(comments['SentCount'], bins=20)


def visual_sentiment(comments, comment_count):
    plt.title(
        f'Гистограмма эмоциональной окраски комментариев для {comment_count} последних коментариев'
    )
    sns.histplot(comments['Emotional'])


def visual_comment_count_by_channel(comments):
    plt.title(
        f'Гистограмма кол-ва комментариев для каждого канала'
    )
    sns.histplot(comments['ChannelName'])


def visual(comments, comment_count):
    plt.title(f"Соотношение эмоциональной окраски комментариев для {comment_count} последних комментариев")
    sns.countplot(
        data=comments,
        x='ChannelName',
        hue='Emotional'
    )


def print_emotional_comments(comments):
    emotion = ['positive', 'neutral', 'negative']
    for item in emotion:
        com = comments.loc[
            comments.Emotional == item,
            ['ChannelName', 'Comment', 'Predict', 'Emotional']
        ].reset_index(drop=True)
        top5 = com.nlargest(5, 'Predict')['Predict'].unique()
        top5_rows = com.loc[com['Predict'].isin(top5)].reset_index(drop=True)
        print(f'\nTop 5 the most: {item} comments:')
        for count, comment in enumerate(top5_rows['Comment'].tolist()):
            print(f"{count + 1}: from channel: {top5_rows['ChannelName'][count]}")
            print(comment)