import numpy as np
from nltk.tokenize import word_tokenize


class Dataset(object):
    def __init__(self, text_file, context_size, vocab_min_count):
        self.text_file = text_file
        self.context_size = context_size
        self.vocab_min_count = vocab_min_count

        self.vocab = []
        self.comat = None
        self.coocs = None

        # Load and process the data
        self.load()

    def load(self):
        f = open(self.text_file, 'r')
        text = f.read().lower()
        f.close()

        # Tokenize the text to create a word list
        word_list = word_tokenize(text)
        w_list_size = len(word_list)

        # Get the vocabulary
        words, counts = np.unique(word_list, return_counts=True)
        self.vocab = []
        for w, count in zip(words, counts):
            if count >= self.vocab_min_count:
                self.vocab.append(w)

        self.vocab_size = len(self.vocab)

        word_to_idx = {w: i for i, w in enumerate(self.vocab)}

        # Construct a co-occurance matrix
        self.comat = np.zeros((self.vocab_size, self.vocab_size))
        for i in range(w_list_size):
            w = word_list[i]
            if w not in self.vocab:
                continue

            idx = word_to_idx[w]

            for j in range(1, self.context_size + 1):

                # Words in the left context
                if i - j > 0:
                    left_idx = word_to_idx.get(word_list[i - j], None)
                    if left_idx is not None:
                        self.comat[idx, left_idx] += 1.0 / j

                # Words in the right context
                if i + j < w_list_size:
                    right_idx = word_to_idx.get(word_list[i + j], None)
                    if right_idx is not None:
                        self.comat[idx, right_idx] += 1.0 / j

        # Non-zero co-occurances
        self.coocs = np.transpose(np.nonzero(self.comat))

    def __getitem__(self, index):
        # Get the left and right word indexes using the self.coocs numpy array
        i, j = self.coocs[index]

        # Return left_idx, right_idx and the co-occurance count
        return int(i), int(j), float(self.comat[i, j])

    def __len__(self):
        return len(self.coocs)
