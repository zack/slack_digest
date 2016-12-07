import enchant
import numpy as np
import pdb
import pprint
import re
import scipy.spatial.distance

from scraper import SlackScraper
from receptiviti import ReceptivitiAPI
from word_frequency import WordFrequency
from sklearn.cluster import MeanShift, estimate_bandwidth
from stop_words import get_stop_words

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
            if word and is_valid_word(word):
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
    print "Building user word string"
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

# Takes a string
# Returns true if the word is:
    # Not a stopword
    # A valid English word
def is_valid_word(word):
    valid_english = enchant.check(word)
    not_stopword = word not in stopwords
    not_sanitized_stopword = sanitize_word(word) not in (map (lambda x:sanitize_word(x), stopwords))
    return valid_english and not_stopword and not_sanitized_stopword

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
        if channel_count > 2:
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
    print "Getting channel topics"
    words_with_frequencies = get_words_with_frequencies(word_list, frequency_vector)[:count]
    return [x[0] for x in words_with_frequencies]

# Takes a channel history and a count n
# Returns the n most important messages from the channel history
def get_important_messages(channel_messages, count):
    reacted_messages = get_messages_with_reactions(channel_messages)
    reacted_messages_with_reaction_counts = map(lambda x:(x['text'],get_reaction_count_on_message(x)), reacted_messages)

def get_messages_with_mentions(channel_messages):
    messages = []
    regex = re.compile(r'[A-Z0-9]{9}')
    for message in channel_messages:
        if regex.search(message['text']):
            messages.append(message)
    return messages

# Takes a channel history
# Returns all messages with reactions
def get_messages_with_reactions(channel_messagse):
    return filter(lambda x: 'reactions' in x.keys(), channel_messages)

# Takes a channel messages
# Returns the total number of reactions to the message
def get_reaction_count_on_message(message):
    count = 0
    for reaction in message['reactions']:
        count += int(reaction['count'])
    return count

# Takes a list of strings representing unix timestamps
# Returns an array of arrays, in which each inner array is a cluster of close timestamps
# Code copied with minor alterations from jabaldonedo on stackoverflow at http://stackoverflow.com/a/18364570
def cluster_messages_by_timestamps(message_times):
    arr = np.array(zip(message_times,np.zeros(len(message_times))), dtype=np.float)
    bandwidth = estimate_bandwidth(arr, quantile=0.1)
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(arr)
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_

    cluster_count = len(np.unique(labels))

    clusters = []
    for k in range(cluster_count):
        my_members = labels == k
        clusters.append(arr[my_members,0])
    return clusters

# Takes raw channel messages
# Returns a clean array of timestamps
def get_channel_message_times(channel_messages):
    return map(lambda x:float(x['ts']), channel_messages)

# Takes clustered arrays of timestamps and channel message history
# Returns an array of arrays of clustered (user, message) using their timestamps
def get_message_with_clusters(clusters, messages):
    dct = dict((float(d['ts']), dict(d, index=index)) for (index, d) in enumerate(messages))
    clustered_messages = []
    for cluster in clusters:
        message_cluster = []
        for timestamp in cluster:
            message = dct[timestamp]
            message_cluster.append((message['user'],message['text']))
        clustered_messages.append(message_cluster)
    return clustered_messages

# Takes clustered (user,message) tuples
# Returns the same clusters, but collapses all tuples from the same users in
# each cluster into a single tuple with the user and a string of all words from
# that user in that cluster
def get_user_cluster_strings(message_clusters):
    new_clusters = []
    for cluster in message_clusters:
        user_messages = {}
        for message in cluster:
            if message[0] in user_messages.keys():
                user_messages[message[0]] += message[1]
            else:
                user_messages[message[0]] = message[1]
            user_messages[message[0]] += " "
        new_clusters.append(user_messages)
    return new_clusters

# Takes clustered sets of {user:text_string}
# Returns clustered sets of {user:receptiviti_data}
def get_user_receptiviti_data_from_clusters(clusters):
    new_clusters = []
    for cluster in clusters:
        new_cluster = {}
        users = cluster.keys()
        for user in users:
            user_data = receptiviti.post_contents(cluster[user])
            new_cluster[user] = user_data['contents'][0]['emotional_analysis']['emotional_tone']
        new_clusters.append(new_cluster)
    return new_clusters

# Takes clustered sets of {user:receptiviti_data}
# Returns a hash of users, where each value is a hash of all other users and
# the key's (user's) average level of agreement. 1 is full agreement. 0 is
# full disagreement.
def build_user_sentiment_associations(clusters): # Sorry this function is SUPER GROSS
    users = {}
    for cluster in clusters:
        users_in_cluster = cluster.keys()
        for user_in_cluster in users_in_cluster:
            if user_in_cluster not in users.keys():
                users[user_in_cluster] = {}

            u = users[user_in_cluster]
            for other_user_in_cluster in users_in_cluster:
                if user_in_cluster != other_user_in_cluster:
                    if other_user_in_cluster not in u.keys():
                        u[other_user_in_cluster] = {'score': 0, 'count': 0}

                    o = u[other_user_in_cluster]
                    # Diff of our user and other user inside this cluster
                    raw_diff = abs(cluster[user_in_cluster]['score'] - cluster[other_user_in_cluster]['score'])
                    norm_diff = 1-(raw_diff/100)
                    new_score = (o['score'] * o['count'] + abs(norm_diff)) / (o['count'] + 1)
                    o['score'] = new_score
                    o['count'] += 1
    return users

def build_cluster_topics(clusters):
    result = []
    for cluster in clusters:
        cluster_vocabulary = build_vocabulary_from_cluster(cluster) # not unique
        cluster_word_list = np.unique(cluster_vocabulary) # unique
        cluster_word_vector = build_word_vector_from_cluster(cluster_vocabulary, cluster_word_list)
        res = get_words_with_frequencies(cluster_word_list, cluster_word_vector)
        if res:
            result.append(res[:2])
    pp.pprint(result)
    return result

def build_vocabulary_from_cluster(cluster):
    vocabulary = []
    for message in cluster:
        for word in message[1].split(' '):
            if word and is_valid_word(word):
                sanitized_word = sanitize_word(word)
                if sanitized_word:
                    vocabulary.append(sanitized_word)
    return np.array(vocabulary)

def build_word_vector_from_cluster(vocabulary, word_list):
    vector = np.zeros(len(word_list), int)
    for word in vocabulary:
        index = np.where(word_list == word)
        vector[index] += 1
    return vector

def build_user_vocabulary_associations(user_word_vectors):
    user_associations = {}
    users = user_word_vectors.keys()
    for user1 in users:
        user_associations[user1] = {}
        for user2 in users:
            if user2 != user1:
                similarity = cosine_similarity(user_word_vectors[user1], user_word_vectors[user2])
                user_associations[user1][user2] = similarity
    return user_associations

freq = WordFrequency()
receptiviti = ReceptivitiAPI()
slack = SlackScraper()
enchant = enchant.Dict('en_US')
stopwords = get_stop_words('english')
user_map = slack.get_user_name_map()

channel_history = slack.get_history_for_channel('politics')
channel_messages = channel_history['messages']
#  important_messages = get_important_messages(channel_messages, 10)
#  channel_vocabulary = build_vocabulary(channel_messages)
#  channel_users = get_users_in_channel(channel_messages)
#  channel_word_list = build_word_list(channel_vocabulary)
#  channel_word_vector = build_word_vector(channel_vocabulary, channel_word_list)
#  user_word_vectors = build_user_word_vectors(channel_vocabulary, channel_word_list, channel_users)
#  user_word_vocab_associations = build_user_vocabulary_associations(user_word_vectors)
#  user_word_strings = build_user_word_strings(channel_word_list, user_word_vectors)
#  mention_messages = get_messages_with_mentions(channel_messages)
#  pp.pprint(map(lambda x:x['text'], mention_messages))


#  user_receptiviti_data = []
#  for user in channel_users:
    #  if user_word_strings[user]:
        #  user_receptiviti_data.append(receptiviti.post_contents(user_word_strings[user]))

#  for user_data in user_receptiviti_data:
    #  print(user_data['contents'][0]['emotional_analysis']['emotional_tone'])

#  pp.pprint(get_words_with_frequencies(channel_word_list, channel_word_vector))

time_clusters = cluster_messages_by_timestamps((get_channel_message_times(channel_messages)))
message_clusters = get_message_with_clusters(time_clusters, channel_messages)
clustered_topics = build_cluster_topics(message_clusters)
#  user_message_clusters = get_user_cluster_strings(message_clusters)
#  user_clusters = get_user_receptiviti_data_from_clusters(user_message_clusters)
#  user_sentiment_association = build_user_sentiment_associations(user_clusters)
#  pp.pprint(user_sentiment_association)
