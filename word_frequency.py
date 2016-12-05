import pdb

class WordFrequency:
    def __init__(self):
        with open('word_freq.txt', 'r') as f:
            raw_words = f.read()
        lines = raw_words.split('\n')[:-1]
        lines_split = map(lambda line:tuple(line.split(' ')), lines)
        counts = map(lambda line:float(line[1]), lines_split)
        total = reduce(lambda memo,line: memo+line, counts, 0)

        self.words = map(lambda line:line.split(' ')[0], lines)
        self.percents = map(lambda line: line/total, counts)

    def get_percent(self, word):
        try:
            index = self.words.index(word)
            return self.percents[index]
        except ValueError:
            return -1
