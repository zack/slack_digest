from cacher import ChannelCacher
from slackclient import SlackClient
import json
import os

class ChannelScraper:

    # Takes the name of a file in the pwd
    # Returns a slack client using credentials from file argument
    @staticmethod
    def create_slack_client(file_name):
        print "Creating slack client"
        with open(file_name, 'r') as creds:
            slack_creds = json.load(creds)
        slack_token = slack_creds['slack']['test_token']
        return SlackClient(slack_token)

    # Takes a name of a channel
    # Returns the result of an api call for the history of that channel
    @staticmethod
    def get_history_for_channel(channel_name):
        print "Retrieving slack channel history for channel: " + channel_name
        slack_client = ChannelScraper.create_slack_client('creds.json')
        if ChannelCacher.channel_is_cached(channel_name):
            print "Found and using channel cache"
            return ChannelCacher.get_channel_cache(channel_name)
        else:
            print "Did not find channel cache"
            print"Querying slack for history for %s" % channel_name
            channels = slack_client.api_call('channels.list')
            channel = [c for c in channels['channels'] if c['name'] == channel_name][0]
            channel_id = channel['id']
            channel_content = slack_client.api_call('channels.history',
                    channel=channel_id, count=1000)
            ChannelCacher.cache_channel(channel_name, channel_content)
            return channel_content
