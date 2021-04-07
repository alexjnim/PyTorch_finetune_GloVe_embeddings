import numpy as np
import torch

def load_glove_from_file(glove_filepath):
    """
    Load the GloVe embeddings

    Args:
        glove_filepath (str): path to the glove embeddings file
    Returns:
        word_to_index (dict), embeddings (numpy.ndarary)
    """

    word_to_index = {}
    embeddings = []
    with open(glove_filepath, "r") as fp:
        for index, line in enumerate(fp):
            line = line.split(" ") # each line: word num1 num2 ...
            word_to_index[line[0]] = index # word = line[0]
            embedding_i = np.array([float(val) for val in line[1:]])
            embeddings.append(embedding_i)
    return word_to_index, np.stack(embeddings)

def make_embedding_matrix(glove_filepath, words):
    """
    Create embedding matrix for a specific set of words.

    Args:
        glove_filepath (str): file path to the glove embeddigns
        words (list): list of words in the dataset
    """
    word_to_idx, glove_embeddings = load_glove_from_file(glove_filepath)
    embedding_size = glove_embeddings.shape[1]

    final_embeddings = np.zeros((len(words), embedding_size))
    in_glove = []
    not_in_glove = []
    for i, word in enumerate(words):
        if word in word_to_idx:
            in_glove.append(word)
            final_embeddings[i, :] = glove_embeddings[word_to_idx[word]]
        else:
            # if the word in our vocabulary is not contained in the glove embeddings we initialise a new entry using the xavier_uniform method
            not_in_glove.append(word)
            embedding_i = torch.ones(1, embedding_size)
            torch.nn.init.xavier_uniform_(embedding_i)
            final_embeddings[i, :] = embedding_i
    print('number of word embeddings found in GloVe: {}/{}'.format(len(in_glove), len(words)))
    print('number of word embeddings not found in GloVe: {}/{}'.format(len(not_in_glove), len(words)))
    # print(not_in_glove)
    return final_embeddings