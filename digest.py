import numpy as np
import pdb
import pprint
import re
import scipy.spatial.distance

from scraper import ChannelScraper
from receptiviti import ReceptivitiAPI
from word_frequency import WordFrequency

# module settings
pp = pprint.PrettyPrinter(indent=2)
np.set_printoptions(threshold='nan')

# Takes the messages from the history of a slack channel
# Returns a set of all users in the channel
def get_users_in_channel(channel_messages):
    print "Compiling users in channel"
    user_set = set()
    for message in channel_messages:
        user_set.add(message['user'])
    users = list(user_set)
    print "Found %d users" % len(users)
    return users

# Takes the messages from the history of a slack channel
# Returns a numpy array of numpy arrays of: [word, user, timestamp]
def build_vocabulary(channel_messages):
    print "Building channel vocabulary from %d messages" % len(channel_messages)
    vocabulary = []
    for message in channel_messages:
        for word in message['text'].split(' '):
            sanitized_word = sanitize_word(word)
            if sanitized_word:
                word_array = np.array([sanitized_word, message['user'], message['ts']])
                vocabulary.append(word_array)
    print "Found %d words in channel history" % len(vocabulary)
    return np.array(vocabulary)

# Takes a numpy array of numpy arrays of: [word, user, timestamp]
# Returns a unique, alphabetically ordered, numpy array of all words used
def build_word_list(vocabulary):
    print "Building word list from vocabulary of length %d" % len(vocabulary)
    words = (map(lambda array: array[0], vocabulary))
    unique_words = np.unique(words)
    print "Found %d unique words" % len(unique_words)
    return unique_words

# Takes:
    # numpy array of numpy arrays of: [word, user, timestamp]
    # sorted set of words
    # optional user ID
# Returns an array of integers representing word usage count
# Array is sorted identically to `build_word`
# If user ID is specified, the data is for that user. Otherwise, channel.
def build_word_vector(vocabulary, word_set, user_id = None):
    if user_id is None:
        print "Building word vector for channel"
    else:
        print "Building word vector for user " + str(user_id)
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
    print "Building word vectors for all users"
    user_map = {user: [] for (user) in users}
    for user in users:
        user_map[user] = build_word_vector(vocabulary, word_set, user)
    return user_map

# Takes:
    # Channel word list
    # Map of users to user word vectors
# Returns a map of users to a string of all words used by each user (not unique)
def build_user_word_strings(channel_word_list, user_vectors):
    print "Building word strings for all users"
    user_map = {}
    for user, vector in user_vectors.iteritems():
        user_map[user] = build_user_word_string(channel_word_list, vector, user)
    return user_map

# Takes:
    # Channel word list
    # User word vector
# Returns a string of all words used by the user. This can be sent to
# Receptiviti for analysis
def build_user_word_string(channel_word_vector, user_word_vector, user):
    print "Building word string for user " + str(user)
    user_string = ""
    it = np.nditer(user_word_vector, flags=['f_index'])
    while not it.finished:
        for i in range(it[0]):
            user_string += " " + channel_word_vector[it.index]
        it.iternext()
    print "Word list has length of %d" % len(user_string.split())
    return user_string.strip()

# Takes a string
# Returns a downcased string with no non-standard-letters
def sanitize_word(word):
    regex = re.compile('[^a-zA-Z]')
    stripped_word = regex.sub('', word)
    downcased_word = stripped_word.lower()
    return downcased_word

# Takes two vectors
# Returns the calculated cosine distance between the two vectors
def cosine_similarity(vector_a, vector_b):
    return scipy.spatial.distance.cosine(vector_a, vector_b)

# Takes a word list and a similarly indexed frequency vector
# Returns an array of tuples each with:
    # The word
    # The number of times the word was used in the conversation
    # The weight of the word's use in the conversation based on:
        # How commonly it was used divided by how commonly it's used in English
def get_words_with_frequencies(word_list, frequency_vector):
    print "Calculating word frequencies for %i words..." % len(word_list)
    total_word_count = float(reduce(lambda x,y:x+y, frequency_vector, 0))
    words_with_channel_counts = zip(word_list, frequency_vector)
    words_with_ratios = {}
    for word, channel_count in words_with_channel_counts:
        english_word_frequency = freq.get_percent(word)
        if english_word_frequency > 0:
            channel_word_frequency = channel_count/total_word_count
            ratio = channel_word_frequency / english_word_frequency
            weight = ratio * channel_count
            words_with_ratios[word] = (ratio, channel_count, weight)
    return sorted(words_with_ratios.items(), key=lambda x:x[1][2], reverse=True)

# Takes a word list, a same-indexed frequency vector, and a topic count
# Returns a number of strings equal to the count, indicating topics for the conversation
def get_channel_topics(word_list, frequency_vector, count):
    words_with_frequencies = get_words_with_frequencies(word_list, frequency_vector)[:count]
    return [x[0] for x in words_with_frequencies]

freq = WordFrequency()

channel_history = ChannelScraper.get_history_for_channel('nihilistic_hell')
channel_vocabulary = build_vocabulary(channel_history['messages'])
channel_users = get_users_in_channel(channel_history['messages'])
channel_word_list = build_word_list(channel_vocabulary)
channel_word_vector = build_word_vector(channel_vocabulary, channel_word_list)
user_word_vectors = build_user_word_vectors(channel_vocabulary, channel_word_list, channel_users)
user_word_strings = build_user_word_strings(channel_word_list, user_word_vectors)

receptiviti = ReceptivitiAPI()

user_receptiviti_data = []
for user in channel_users:
    user_receptiviti_data.append(receptiviti.post_contents(user_word_strings[user]))

for user_data in user_receptiviti_data:
    print(user_data['contents'][0]['emotional_analysis']['emotional_tone'])

topics = get_channel_topics(channel_word_list, channel_word_vector, 5)
pp.pprint(topics)
