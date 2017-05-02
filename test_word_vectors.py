from __future__ import absolute_import, division, print_function

from argparse import ArgumentParser
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def print_nearest_words(args):
    word = args.word.lower().strip()

    # Load the word vectors
    embeddings_index = {}
    f = open(args.vectors)
    for line in f:
        values = line.split(' ')
        w = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[w] = coefs
    f.close()

    w_v = np.zeros_like(embeddings_index[w])
    for w in word.split():
        if w not in embeddings_index.keys():
            continue

        w_v += embeddings_index[w]

    # Get the similarity scores
    score_dict = {}
    for w in embeddings_index.keys():
        if word == w:
            continue

        score = cosine_similarity(w_v.reshape(1, -1), embeddings_index[w].reshape(1, -1))[0][0]
        score_dict[w] = score

    closest = Counter(score_dict).most_common(args.num_words)

    close_words = []
    for word, score in closest:
        if args.verbose:
            print(score, word)
        else:
            close_words.append(word)

    if not args.verbose:
        print(', '.join(close_words))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--vectors', default='vectors.txt', help='Word vector file')
    parser.add_argument('--vocab', default='vocab.txt', help='Vocab file')
    parser.add_argument('--word', default='dollar', help='Input word')
    parser.add_argument('--verbose', type=bool, default=False, help='Print score')
    parser.add_argument('--num_words', type=int, default=5, help='Number of closest words to print')
    args = parser.parse_args()

    print_nearest_words(args)
