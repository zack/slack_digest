import os
import json
import pprint
from slackclient import SlackClient

pp = pprint.PrettyPrinter(indent=2)

with open('slack_creds.txt', 'r') as creds:
    slack_creds = json.load(creds)

slack_token = slack_creds['test_token']
sc = SlackClient(slack_token)

channels = sc.api_call('channels.list')
politics_channel = [c for c in channels['channels'] if c['name'] == 'politics'][0]
politics_channel_id = politics_channel['id']

channel_history = sc.api_call('channels.history', channel=politics_channel_id)

pp.pprint(channel_history)
