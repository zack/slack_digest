from cacher import ChannelCacher
from slackclient import SlackClient
import json
import os

class SlackScraper:

    def __init__(self):
        print "Creating slack client"
        with open('creds.json', 'r') as creds:
            slack_creds = json.load(creds)
        slack_token = slack_creds['slack']['test_token']
        self.client = SlackClient(slack_token)


    def get_user_name_map(self):
        user_data = self.client.api_call('users.list')['members']
        user_map = map(lambda u:{u['id']:u['name']}, user_data)
        return user_map

    # Takes a name of a channel
    # Returns the result of an api call for the history of that channel
    def get_history_for_channel(self, channel_name):
        print "Retrieving slack channel history for channel: " + channel_name
        if ChannelCacher.channel_is_cached(channel_name):
            print "Found and using channel cache"
            return ChannelCacher.get_channel_cache(channel_name)
        else:
            print "Did not find channel cache"
            print"Querying slack for history for %s" % channel_name
            channels = self.client.api_call('channels.list')
            channel = [c for c in channels['channels'] if c['name'] == channel_name][0]
            channel_id = channel['id']
            channel_content = self.client.api_call('channels.history',
                    channel=channel_id, count=1000)
            ChannelCacher.cache_channel(channel_name, channel_content)
            return channel_content
