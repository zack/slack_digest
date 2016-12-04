class WordFrequency:
    def __init__(self):
        with open('word_freq.txt', 'r') as f:
            raw_words = f.read()
        self.words = raw_words.split('\n')

    def get_rank(self, word):
        try:
            return self.words.index(word)
        except ValueError:
            return -1
