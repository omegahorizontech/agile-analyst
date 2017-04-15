# TODO: Implement methods for the affective AI based on hash tables. We need to initialize the storage and mgmt mechanisms, train it, and score samples with it.

class affect_AI:
    def __init__():
        # This is where we set up whatever objects we need for the hash table and dictioanries.
        pass
    def train(corpora):
        # This is where we actually 'learn' the vocabulary and its r-emotion scores.
        """
        Inputs: corpora, a pandas DataFrame object. Contains a row for each word in the original corpus. First column is the word. Subsequent columns are filled as needed to specify r-emotion corpus and tier each word belongs to.

        Outputs: None. Stores as an internal object (an attribute on 'self') an ordered dictionary of ordered dictionaries containing our words as keys in the second order dictionaries and the corpora and tiers it's part of as values in the second order dictionaries. 
        """
        # We need to articulate each corpus into a fixed number of dictionaries, which in turn will be stored in dictionaries. Dictionaries in python use hash tables for lookup and storage, so this will be our hash table
        pass
    def score(sample):
        # This is where we take a sample and return the 400 r-emotion scores.

        # For each word in the sample, we check if it's where it should be in our hash table. If it's there, we add its contribution to the total r-emotion scores for the sample.
        pass
