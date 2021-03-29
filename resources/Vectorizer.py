import pandas as pd
import numpy as np
import string
from config import config
from collections import Counter
from resources.Vocabularies import Vocabulary, SequenceVocabulary

class Vectorizer(object):
    """ The Vectorizer which coordinates the Vocabularies and puts them to use"""
    def __init__(self, text_vocab, y_values_vocab):
        self.text_vocab = text_vocab
        self.y_values_vocab = y_values_vocab

    @classmethod
    def from_dataframe(cls, data_df, cutoff=config.unk_words_cutoff):
        """Instantiate the vectorizer from the dataset dataframe

        Args:
            data_df (pandas.DataFrame): the target dataset
            cutoff (int): frequency threshold for including in Vocabulary
        Returns:
            an instance of the Vectorizer
        """
        y_values_vocab = Vocabulary()
        for y_value in sorted(set(data_df[config.y_values_column])):
            y_values_vocab.add_token(y_value)

        word_counts = Counter()
        for text in data_df[config.text_column]:
            for token in text.split(" "):
                if token not in string.punctuation:
                    word_counts[token] += 1

        # we want to prepare the text_vocab such that it can vectorize text ith sequence tokens for sequence inputs
        text_vocab = SequenceVocabulary()
        for word, word_count in word_counts.items():
            if word_count >= cutoff:
                text_vocab.add_token(word)

        return cls(text_vocab, y_values_vocab)

    def vectorize(self, text, vector_length=-1):
        """
        Args:
            text (str): the string of words separated by a space
            vector_length (int): an argument for forcing the length of index vector
        Returns:
            the vectorized text (numpy.array)
        """
        # here we build the sequence input using the sequence tokens
        # add the BEGIN-OF-SEQUENCE token index first
        indices = [self.text_vocab.begin_seq_index]
        # map all other words to their indices and append (UNK if not available in text_vocab)
        indices.extend(self.text_vocab.lookup_token(token)
                       for token in text.split(" "))
        # add END-OF-SEQUENCE token to the end of sentence
        indices.append(self.text_vocab.end_seq_index)

        # check if vector length is fixed, if not, use the length of indices
        # vector_length will usually be defined by _max_seq_length (i.e. the length of longest text + 2 for BEGIN-OF-SEQUENCE and END-OF-SEQUENCE tokens), to ensure that all vectors are the same size
        if vector_length < 0:
            vector_length = len(indices)

        # now build output_vector based on the fixed vector_length
        output_vector = np.zeros(vector_length, dtype=np.int64)
        output_vector[:len(indices)] = indices
        output_vector[len(indices):] = self.text_vocab.mask_index

        # example output_vector: [BEGIN-OF-SEQUENCE-INDEX, 13,89,UNK-INDEX,5,31,67,32,113, UNK-INDEX, 456, END-OF-SEQUENCE-INDEX, MASK-INDEX, MASK-INDEX, MASK-INDEX]
        # fixed length of 15, but the actual text is only 10 words long
        return output_vector