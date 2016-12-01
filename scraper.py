#! /usr/local/bin/python

# imports
import pdb
import json
import numpy as np
import os
import pprint
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
    channels = slack_client.api_call('channels.list')
    channel = [c for c in channels['channels'] if c['name'] == channel_name][0]
    channel_id = channel['id']
    return slack_client.api_call('channels.history', channel=channel_id)

# Takes the messages from the history of a slack channel
# Returns a numpy array of numpy arrays of: [word, user, timestamp]
def build_vocabulary(channel_messages):
    vocabulary = []
    for message in channel_messages:
        for word in message['text'].split(' '):
            word_array = np.array([word, message['user'], message['ts']])
            vocabulary.append(word_array)
    return np.array(vocabulary)

# Takes a numpy array of numpy arrays of: [word, user, timestamp]
# Returns a unique, alphabetically ordered, numpy array of all words used
def build_word_vector(vocabulary):
    words = (map(lambda array: array[0], vocabulary))
    unique_words = np.unique(words)
    pdb.set_trace()
    return unique_words

slack_client = create_slack_client('slack_creds.txt')
channel_history = get_history_for_channel('politics')
channel_vocabulary = build_vocabulary(channel_history['messages'])
word_vector = build_word_vector(channel_vocabulary)
print(channel_vocabulary)
