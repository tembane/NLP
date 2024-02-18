import os
import pandas as pd
from googleapiclient.discovery import build
import regex

API_KEY: str = os.getenv('GOOGLE_API_KEY')
service = build('youtube', 'v3', developerKey=API_KEY)


def is_valid_sentence(text: str) -> bool:
    """Checks if the sentence is valid based on the given pattern."""
    pattern = regex.compile(r'^\p{So}*\s*\P{So}\s*[\p{L}\d\s.,!?"\'():;-]*\s*\P{So}\s*$')
    return bool(pattern.match(text))


def snippet_to_dict(snippet: dict, performer: str) -> dict:
    """Converts a comment snippet to a dictionary."""
    return {
        'ChannelName': performer,
        'VideoId': snippet['videoId'],
        'AuthorId': snippet['authorDisplayName'],
        'Comment': snippet['textOriginal'],
    }


def get_channel_info(channel_name: str) -> tuple | None:
    """
    Gets the channel info (channel id and channel title) by YouTube channel nickname
    Args:
        channel_name (str): The channel nickname. Must stick to the following pattern: "@your_channel_name"
    Returns:
        tuple: Channel_id and Channel_title
    """

    search_response = service.search().list(
        q=channel_name,
        part='snippet',
        type='channel',
        maxResults=1
    ).execute()
    if search_response['items']:
        channel_id = search_response['items'][0]['snippet']['channelId']
        channel_title = search_response['items'][0]['snippet']['title']
        return channel_id, channel_title
    else:
        return None


def get_channels() -> dict:
    channels = {}
    numbers_of_channels = int(input("Enter number of channels: "))
    for i in range(numbers_of_channels):
        nickname: str = input("Enter channel nickname: ex'@nickname'")
        channel_id, channel_title = get_channel_info(nickname)
        if channel_id is not None:
            channels[channel_id] = channel_title
    return channels


def get_comments(max_result: int = 20, number_of_pages: int = 10) -> pd.DataFrame:
    """
    Gets comments from YouTube channels and returns them as a pandas dataframe.

    Args:
        max_result (int): Maximum number of comments per page.
        number_of_pages (int): Number of pages to scrape from YouTube.

    Returns:
        DataFrame: A pandas DataFrame containing the following columns:
            1. ChannelName: Titles of YouTube channels associated with the videos.
            2. VideoID: Unique identifiers of the YouTube videos.
            3. AuthorID: Unique identifiers of the authors who posted comments.
            4. Comment: Comments posted by users on the respective videos.
    """

    comments: list[dict] = []
    channels = get_channels()
    for ids, performer in channels.items():
        args = {
            'allThreadsRelatedToChannelId': ids,
            'part': 'id, snippet, replies',
            'maxResults': max_result
        }
        for page in range(number_of_pages):
            comment_threads = service.commentThreads().list(**args).execute()
            for item in comment_threads['items']:
                top_level_comment = item['snippet']['topLevelComment']
                comment_snippet = top_level_comment['snippet']
                comments.append(snippet_to_dict(comment_snippet, performer))
                if 'replies' in item:
                    reply = item['replies']
                    for rep in reply['comments']:
                        comments.append(snippet_to_dict(rep['snippet'], performer))
            args['pageToken'] = comment_threads.get('nextPageToken')
            if not args['pageToken']:
                break
    return pd.DataFrame(comments)
