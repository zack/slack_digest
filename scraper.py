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
#np.set_printoptions(threshold='nan')

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
# Returns a set of all users in the channel
def get_users_in_channel(channel_messages):
    users = set()
    for message in channel_messages:
        users.add(message['user'])
    return users

# Takes the messages from the history of a slack channel
# Returns a numpy array of numpy arrays of: [word, user, timestamp]
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
def build_word_set(vocabulary):
    words = (map(lambda array: array[0], vocabulary))
    unique_words = np.unique(words)
    return unique_words

# Takes:
    # numpy array of numpy arrays of: [word, user, timestamp]
    # sorted set of words
    # optional user ID
# Returns an array of integers representing word usage count
# Array is sorted identically to `build_word`
# If user ID is specified, the data is for that user. Otherwise, channel.
def build_word_vector(vocabulary, word_set, user_id = None):
    vector = np.zeros(len(word_set), int)
    for word in vocabulary:
        if(user_id is None or user_id == word[1]):
            index = np.where(word_set == word[0])
            vector[index] += 1
    return vector

# Takes:
    # numpy array of numpy arrays of: [word, user, timestamp]
    # sorted set of words
    # set of all users represented in channel data
# Returns a dictionary with keys users and values user word vectors
def build_user_word_vectors(vocabulary, word_set, users):
    user_map = {user: [] for (user) in users}
    for user in users:
        user_map[user] = build_word_vector(vocabulary, word_set, user)
    return user_map

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
channel_users = get_users_in_channel(channel_history['messages'])
channel_word_set = build_word_set(channel_vocabulary)
channel_word_vector = build_word_vector(channel_vocabulary, channel_word_set)
user_word_vectors = build_user_word_vectors(channel_vocabulary, channel_word_set, channel_users)
pp.pprint(channel_users)
pp.pprint(channel_word_vector)
pp.pprint(user_word_vectors)
