from resources.Vectorizer import Vectorizer
from resources.Vocabularies import SequenceVocabulary, Vocabulary
from utils.HelperFunctions import *
from resources.Dataset import Dataset, generate_batches
from resources.LoadGloVeEmbeddings import make_embedding_matrix
from utils.CleanData import clean_data
from model.EmbCNN_Model import Classifier
from train_val_test_loop import train_val_test_model
from config import config
import pandas as pd
import numpy as np
import os.path
import torch
import torch.nn as nn
import torch.optim as optim

# Check CUDA
if not torch.cuda.is_available():
    config.cuda = False
device = torch.device("cuda" if config.cuda else "cpu")
print("Using CUDA: {}".format(config.cuda))
# Set seed for reproducibility
set_seed_everywhere(config.seed, config.cuda)

''' need to develop loading and saving capabilities'''
if os.path.isfile(config.data_path):
    print("Cleaned data exists")
else:
    print ("Cleaned data does not exist, computing this now")
    clean_data()

# NEED TO UPDATE THIS PART TO INVOLVE LOADING AND SAVING OF DATASET + VECTORIZER
if config.reload_from_files:
    # training from a checkpoint
    dataset = Dataset.load_dataset_and_load_vectorizer(config.data_path, config.vectorizer_file)
else:
    # create dataset and vectorizer
    dataset = Dataset(config.data_path)
    vectorizer = dataset._vectorizer
    # dataset.save_vectorizer(config.vectorizer_file)

# load the GloVe embeddings
words = vectorizer.text_vocab._token_to_idx.keys()
if config.use_glove:
    print("Using pre-trained embeddings")
    embeddings = make_embedding_matrix(glove_filepath=config.glove_filepath, words=words)
    config.embedding_size = embeddings.shape[1]
else:
    print("Not using pre-trained embeddings")
    embeddings = None

print('input vector length: {}'.format(dataset._max_seq_length))

classifier = Classifier(embedding_size=config.embedding_size,
                            num_embeddings=len(vectorizer.text_vocab._idx_to_token),
                            kernel_width = config.kernel_width,
                            num_channels=config.num_channels,
                            hidden_dim=config.hidden_dim,
                            num_classes=len(vectorizer.y_values_vocab._idx_to_token),
                            dropout_p=config.dropout_p,
                            pretrained_embeddings=embeddings,
                            padding_idx=0)
classifier = classifier.to(device)

# a manual rescaling weight given to each class
dataset.class_weights = dataset.class_weights.to(device)
# use class weights in case of imbalanced data
loss_func = nn.CrossEntropyLoss(dataset.class_weights)
# set optimizer
optimizer = optim.Adam(classifier.parameters(), lr=config.learning_rate)

train_state, classifier = train_val_test_model(classifier, dataset, device, optimizer, loss_func)
print(train_state)
# now that training is over, save the embeddings
save_embeddings(classifier, train_state, words)



