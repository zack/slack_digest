import json
import os

class ChannelCacher:

    # Takes a channel name
    # Returns a boolean indicating whether a channel has been cached locally
    @staticmethod
    def channel_is_cached(channel_name):
        return os.path.isfile(ChannelCacher.get_cache_file_name(channel_name))

    # Takes a channel name
    # Returns the cache of that channel from a local file
    @staticmethod
    def get_channel_cache(channel_name):
        with open(ChannelCacher.get_cache_file_name(channel_name), 'r') as content:
            return json.load(content)

    # Takes a channel name and channel content
    # Writes that content to a local file
    @staticmethod
    def cache_channel(channel_name, content):
        f = open(ChannelCacher.get_cache_file_name(channel_name), 'w')
        json.dump(content, f)
        f.close()

    # Takes a file name
    # Returns a string that should be used for the name of that channel's local cache file
    @staticmethod
    def get_cache_file_name(channel_name):
        return ('./' + channel_name + '_cache.txt')
