# TODO: Implement methods for the affective AI based on hash tables. We need to initialize the storage and mgmt mechanisms, train it, and score samples with it.
import sys
from collections import Counter
import copy

class affect_AI:
    def __init__(vocab_size, secondary_dict_size):
        # This is where we set up whatever objects we need for the hash table and dictionaries.
        """
        Params
        ------
         vocab_size: int, the total size of the vocabulary to be stored.

         secondary_dict_size: int, the maximum number of words to be stored in the second-order dictionaries.

        """
        self.vocab_size = vocab_size
        self.secondary_dict_size = secondary_dict_size
        self.primary_size = vocab_size // secondary_dict_size
        # The keys in our primary dictionary should correspond to ranges within our corpora, so those will need to be set in 'train'
        self.dict = {}
        pass
    def train(corpora):
        # This is where we actually 'learn' the vocabulary and its r-emotion scores.
        # Dilemma: Do we use 400 columnns (or num columns = num corpora) or do we just have words with lists of the corpora and tiers they're in? The first is larger, but we can use column labels to facilitate some training/scoring functions more easily. The second way saves some memory, but we'd have to build a list of all the unique corpora we've encountered as we take in vocabulary.
        """
        Uses nested dictionaries to keep lookup times to a minimum. Python dictionaries are implemented with hash tables, and their overhead stays relatively low up to hundreds or thousands of members, so we try to keep each dictionary close to this number of members. Future development directions might include self-adjusting or empirically self-determined dictionary sizes and ratios, where the sizes would all be chosen to maximize lookup time while keeping to minimum complexity.

        Inputs: DataFrame. 'corpora' is a pandas DataFrame object. Contains a row for each word in the original corpus. First column is the word. Subsequent columns are filled as needed to specify r-emotion corpus and tier each word belongs to.

        Outputs: None. Stores as an internal object (an attribute on 'self') an ordered dictionary of ordered dictionaries containing our words as keys in the second order dictionaries and the corpora and tiers it's part of as values in the second order dictionaries.
        """
        # We need to articulate each corpus into a fixed number of dictionaries, which in turn will be stored in dictionaries. Dictionaries in python use hash tables for lookup and storage, so this will be our hash table.

        if corpora.length != self.vocab_size:
            raise ValueError: print "corpus length does not match initialized vocab size"


        # For each future secondary dictionary within our corpora, we need to find a range that will serve as a key in our primary dictionary. This will tell us which secondary dictionary to retrieve.

        # Each key in the primary dictionary will represent the range of words present in the secondary dictionary. If a word has a lower alphabetical value than a key, it must belong to the prior key. Thus, we will need to specify sequences to use as keys based on the size of our total corpus and secondary dictionaries. Additionally, we will need to consider the unique distribution of words and the letters they begin with in our corpus.

        # We should start by alphabetizing the words in our corpus

        # We then find every nth word, where n = secondary dict size. These will serve as the cutoff words for our keys for the primary dictionary.
        keys = []
        for primary in xrange(0, self.vocab_size, self.secondary_dict_size):
            # We use an xrange because it's a generator, not a static list.
            keys.append[corpora[primary]['word']]
        # We want to preserve a full list of the keys that's readily accessible
        keys.sort()
        self.primary_keys = copy.copy(keys)
        # We use the first m letters of each word such that we have the minimum number required to distinguish one key from its neighbor. eg, 'making' has the key neighbor 'masking', so assuming we're constrained into using 'mak' for the first one by its earlier neighbor, we only need to use 'mas' for the second one.
        # TODO: write 'reduce_chars' helper function to reduce chars to least number required to distinguish each member of a list.
        self.corpora = Counter()

        # Now that we have keys for the primary dictionary, we can create each of the secondary dictionaries.
        for primary in xrange(0, self.primary_size):
            # We need two 'for' loops, one for the primary key we're dealing with, and one for each of the secondary keys we'll be dealing with.
            current_key = keys[primary]
            self.dict[current_key] = {}
            for secondary in xrange(0, self.secondary_dict_size):
                # We need to get the right index from the corpora, processing each word as part of a block of secondary dictionary words for each primary key.
                current_word = corpora[self.secondary_dict_size * primary + secondary]
                # Each key in our secondary dictionaries will be a word, beginning with the word which partly served as a key in the primary dictionary.
                # The secondary key will be the word from the corpus, and the value there will be a list of symbols corresponding to the corpus names and tiers.
                corpora = []
                corpora = current_word['corpora'])
                # We track all of the corpora and tiers we've encountered
                self.corpora.update(corpora)
                # In each secondary dictionary, each key (word in our corpus) will have the corpora its found in and its tier stored as a list of symbols (eg, 'Ag-1', 'Cl-2', etc.). This will make scoring a simple matter of looking up a word in our dictionaries, tracking the count of each symbol, and then calculating the score for each affect category at the end by applying our scoring coefficients to the symbol counter.
                self.dict[current_key][current_word['word']] = corpora

        # TODO: Write 'symbolify' method to reduce corpora names and tiers to symbols.
        self.symbolify()

        pass
    def score(sample):
        # This is where we take a sample and return the 400 r-emotion scores.
        """
        Inputs: Str. A string composed of any number of words, sentences, or paragraphs.

        Outputs: List. A list of 400 floats, each float corresponding to an r-emotion value score.
        """
        # For each word in the sample, we check if it's where it should be in our hash table. If it's there, we add its contribution to the total r-emotion scores for the sample.
        scores = Counter()
        # TODO: Implement 'wordify' method
        sample = wordify(sample)
        for word in sample:
            # TODO: implement 'find_index' method for returning which primary index a word should be found in
            primary_index = find_index(word)
            secondary_dict = self.dict[self.primary_keys[primary_index]]
            if word in secondary_dict:
                scores.update(secondary_dict[word])

        pass

    def symbolify():
        # This method should only be called at the end of trianing. It reduces the corpora for each word in the affect_ai's dictionary to a symbol. These symbols are generated using the 'reduce_chars' method. Each symbol is the minimum number of characters required to differentiate it from another symbol, followed by a number for each corresponding tier within the corpus. 
