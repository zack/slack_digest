#! /usr/local/bin/python

from scraper import ChannelScraper
import numpy as np
import pdb
import pprint
import re

# module settings
pp = pprint.PrettyPrinter(indent=2)
np.set_printoptions(threshold='nan')

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

channel_history = ChannelScraper.get_history_for_channel('politics')
channel_vocabulary = build_vocabulary(channel_history['messages'])
channel_users = get_users_in_channel(channel_history['messages'])
channel_word_set = build_word_set(channel_vocabulary)
channel_word_vector = build_word_vector(channel_vocabulary, channel_word_set)
user_word_vectors = build_user_word_vectors(channel_vocabulary, channel_word_set, channel_users)
pp.pprint(channel_users)
pp.pprint(channel_word_vector)
pp.pprint(user_word_vectors)
