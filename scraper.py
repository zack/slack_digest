#! /usr/local/bin/python

# imports
import json
import numpy as np
import os
import pdb
import pprint
import re
from slackclient import SlackClient

# module settings
pp = pprint.PrettyPrinter(indent=2)
np.set_printoptions(threshold='nan')

# Takes the name of a file in the pwd
# Returns a slack client using credentials from file argument
def create_slack_client(file_name):
    with open(file_name, 'r') as creds:
        slack_creds = json.load(creds)
    slack_token = slack_creds['test_token']
    return SlackClient(slack_token)

# Takes a name of a channel
# Returns the result of an api call for the history of that channel
def get_history_for_channel(channel_name):
    if channel_is_cached(channel_name):
        return get_channel_cache(channel_name)
    else:
        channels = slack_client.api_call('channels.list')
        channel = [c for c in channels['channels'] if c['name'] == channel_name][0]
        channel_id = channel['id']
        channel_content = slack_client.api_call('channels.history', channel=channel_id)
        cache_channel(channel_name, channel_content)
        return channel_content

# Takes the messages from the history of a slack channel
# Returns a numpy array of numpy arrays of: [word, user, timestamp]
# Removes all non-letters
def build_vocabulary(channel_messages):
    vocabulary = []
    for message in channel_messages:
        for word in message['text'].split(' '):
            sanitized_word = sanitize_word(word)
            word_array = np.array([sanitized_word, message['user'], message['ts']])
            vocabulary.append(word_array)
    return np.array(vocabulary)

# Takes a numpy array of numpy arrays of: [word, user, timestamp]
# Returns a unique, alphabetically ordered, numpy array of all words used
def build_word_vector(vocabulary):
    words = (map(lambda array: array[0], vocabulary))
    unique_words = np.unique(words)
    return unique_words

# Takes a string
# Returns a downcased string with no non-standard-letters
def sanitize_word(word):
    regex = re.compile('[^a-zA-Z]')
    stripped_word = regex.sub('', word)
    downcased_word = stripped_word.lower()
    return downcased_word

# Takes a channel name
# Returns a boolean indicating whether a channel has been cached locally
def channel_is_cached(channel_name):
    return os.path.isfile(get_cache_file_name(channel_name))

# Takes a channel name
# Returns the cache of that channel from a local file
def get_channel_cache(channel_name):
    with open(get_cache_file_name(channel_name), 'r') as content:
        return json.load(content)

# Takes a channel name and channel content
# Writes that content to a local file
def cache_channel(channel_name, content):
    f = open(get_cache_file_name(channel_name), 'w')
    json.dump(content, f)
    f.close()

# Takes a file name
# Returns a string that should be used for the name of that channel's local cache file
def get_cache_file_name(channel_name):
    return ('./' + channel_name + '_cache.txt')

slack_client = create_slack_client('slack_creds.txt')
channel_history = get_history_for_channel('politics')
channel_vocabulary = build_vocabulary(channel_history['messages'])
word_vector = build_word_vector(channel_vocabulary)
print(word_vector)
