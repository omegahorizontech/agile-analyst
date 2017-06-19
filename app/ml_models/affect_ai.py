import sys
from collections import Counter
import pandas as pd
import copy

# New direction: store vocab words in a single dictionary. Keep track of corpora. At the end, when all corpora have been tallied, generate a dictionary where each key is a corpus and each value is a number between 0 and the number of corpora. During scoring, the number value for each word is put into a counter, then converted into a dictionary for final reporting.

class affect_AI:
    def __init__(self):
        # This is where we set up whatever objects we need for the hash table and dictionaries.
        """
        Params
        ------
         vocab_size: int, the total size of the vocabulary to be stored.

         secondary_dict_size: int, the maximum number of words to be stored in the second-order dictionaries.

        """
        # self.vocab_size = vocab_size
        # self.secondary_dict_size = secondary_dict_size
        # self.primary_size = vocab_size // secondary_dict_size
        # The keys in our primary dictionary should correspond to ranges within our corpora, so those will need to be set in 'train'
        self.vocab = {}
        self.corpora = {}
        self.weights = {}

    def train(self, vocab, weights):
        # This is where we actually 'learn' the vocabulary and its r-emotion scores.
        # Dilemma: Do we use 400 columnns (or num columns = num corpora) or do we just have words with lists of the corpora and tiers they're in? The first is larger, but we can use column labels to facilitate some training/scoring functions more easily. The second way saves some memory, but we'd have to build a list of all the unique corpora we've encountered as we take in vocabulary.
        """
        Uses nested dictionaries to keep lookup times to a minimum. Python dictionaries are implemented with hash tables, and their overhead stays relatively low up to hundreds or thousands of members, so we try to keep each dictionary close to this number of members. Future development directions might include self-adjusting or empirically self-determined dictionary sizes and ratios, where the sizes would all be chosen to maximize lookup time while keeping to minimum complexity.

        Inputs:
            --'vocab', DataFrame. 'vocab' is a pandas DataFrame object. Contains a row for each word in the original corpus. First column is the word. Subsequent columns are filled as needed to specify r-emotion corpus and tier each word belongs to.
            --'weights', python dictionary. 'weights' is a dictionary containing scaling coefficients for each tier of each corpus. These coefficients are meant to scale any word found in a tier based on its frequency. Tiers with more words will have smaller coefficients, and tiers with fewer words will have larger coefficients.

        Outputs: None. Stores as an internal object (an attribute on 'self') an ordered dictionary of ordered dictionaries containing our words as keys in the second order dictionaries and the corpora and tiers it's part of as values in the second order dictionaries.
        """
        # if len(vocab) != self.vocab_size:
        #     raise ValueError("corpus length does not match initialized vocab size")
        word_col = vocab.axes[1][0]
        vocab.sort_values(by=word_col)
        print '---vocab:',vocab
        # vocabulary = vocab[col]

        corp_num = 0
        for row in range(vocab.shape[0]):
            self.vocab[vocab.iloc[row][0]] = self.vocab[vocab.iloc[row][1]]
        for value in Counter(self.vocab.values()):
            self.corpora[value] = corp_num
            corp_num += 1
        self.weights = weights
        self.symbolify()


    def score(self, sample):
        # This is where we take a sample and return the 400 r-emotion scores.
        """
        Inputs: Str. A string composed of any number of words, sentences, or paragraphs.

        Outputs: Dictionary. A dictionary of r-emotion symbols and 1200 floats, each float corresponding to an r-emotion value score.
        """
        # For each word in the sample, we check if it's where it should be in our hash table. If it's there, we add its contribution to the total r-emotion scores for the sample.
        scores = Counter()
        r_scores = {}
        sample = self.wordify(sample)
        for word in sample:
            # primary_index = self.find_index(word)
            # print 'primary_index:', primary_index
            # secondary_dict = self.dict[primary_index]
            # print 'this is secondary_dict:',secondary_dict
            # print 'this is word in secondary_dict:',word
            if word in self.vocab:
                scores.update([self.vocab[word]])

        for symbol in scores:
            print 'this is symbol:',symbol,'this is scores:',scores
            # We need to multiply the score for each symbol by its weight for the corpus.
            r_scores[symbol] = scores[symbol] * self.weights[symbol]

        return r_scores


    def symbolify(self):
        # This method should only be called at the end of trianing. It reduces the corpora for each word in the affect_ai's dictionary to a symbol. These symbols are generated using the 'reduce_chars' method. Each symbol is the minimum number of characters required to differentiate it from another symbol, followed by a number for each corresponding tier within the corpus.

        for word in self.vocab:
            corp = self.vocab[word]
            if type(corp) == list():
                new_corpora = []
                for corpus in corp:
                    new_corpora.append(self.corpora[corpus])
                self.vocab[word] = new_corpora
            else:
                self.vocab[word] = self.corpora[corp]
        for corpus in self.weights.keys():
            new_weights[self.symbols[corpus]] = self.weights[corpus]
        self.weights = new_weights

    # def reduce_chars(self, verbose):
    #     # This method takes a list of strings and returns a dictionary. The returned dictionary's keys are each of the original words and its values are a reduced version of the word. The reduction is based on keeping the minimum number of characters required to differentiate it from its preceding neighbor. ["apple", "apply", "adequate"] would therefore be returned as ["a", "ap", "ad"]. If the word contains a hyphen or space followed by a number, like ["apple-1", "apple 2" "apply", "adequate"] the word is returned in reduced form followed by a hyphen and its number, like so: ["a-1", "a-2", "ap", "ad"].
    #     # print verbose
    #     reduced = {}
    #     reduced[verbose[0]] = verbose[0][0] + '-' + verbose[0].split(" ")[1]
    #     # Iterate over the words
    #     for word in range(1,len(verbose)):
    #         # Use the minimum number of letters to differentiate it from a previous neighbor.
    #         index = 0
    #         cur_symbol = verbose[word][index]
    #         current_numeral = verbose[word].split(" ")[1]
    #         corpus_and_tier = reduced[verbose[word-1]].split(" ")
    #         prev_symbol = corpus_and_tier[0]
    #         while cur_symbol == prev_symbol:
    #             index += 1
    #             cur_symbol += verbose[word][index]
    #         # Reassociate any number it had.
    #         # print cur_symbol, current_numeral
    #         cur_symbol += "-" + current_numeral
    #         # Store the new symbol in a dictionary with the word it replaces.
    #         reduced[verbose[word]] = cur_symbol
    #     return reduced

    def wordify(self, sentence):
        """ wordify method takes a sentence or paragraph, as a single string, and returns a list of words with letters and numbers only.

        Inputs
            sentence: string. The string to be transformed into a list of words.
        Outputs
            words: list. A list of alphanumeric words, found separated within 'sentence' by spaces.

        """
        words = sentence.split(" ")
        words = [str().join(filter(str.isalpha, word)) for word in words]
        words = [word for word in words if word]
        return words

    # def find_index(self, query):
    #     keys = self.primary_keys
    #     if query in keys:
    #         print 'dict keys:', keys
    #         print 'dict primary keys:', self.dict.keys()
    #         return query
    #     keys.append(query)
    #     keys.sort()
    #     location = keys.index(query)
    #     index_word = keys[location-1]
    #     # print index_word
    #     return index_word
