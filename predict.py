import torch
import json
import re
import pandas as pd
import utils.TextNormalizer as tn
from utils.Contractions import CONTRACTION_MAP
from model.EmbCNN_Model import Classifier
from resources.Vectorizer import Vectorizer
from resources.Dataset import Dataset
from resources.LoadGloVeEmbeddings import make_embedding_matrix
from config import config


def predict_category(text, classifier, vectorizer, max_length):
    """Predict the category for a new text

    Args:
        text (str): a raw text string
        classifier (Classifier): an instance of the trained classifier
        vectorizer (Vectorizer): the corresponding vectorizer
        max_length (int): the max sequence length
            Note: CNNs are sensitive to the input data tensor size.
                  This ensures to keep it the same size as the training data
    """
    processed_text = tn.normalize_corpus(corpus=pd.Series(text), html_stripping=True,
                                    contraction_expansion=True, accented_char_removal=True,
                                    text_lower_case=True, text_lemmatization=True,
                                    text_stemming=False, special_char_removal=True,
                                    remove_digits=False, stopword_removal=False)
    vectorized_title = torch.tensor(vectorizer.vectorize(processed_text[0], vector_length=max_length))
    result = classifier(vectorized_title.unsqueeze(0), apply_softmax=True)
    probability_values, indices = result.max(dim=1)
    predicted_y_value = vectorizer.y_values_vocab.lookup_index(indices.item())

    return {'category': predicted_y_value,
            'probability': probability_values.item()}


if __name__ == "__main__":
    if not torch.cuda.is_available():
        cuda = False
    device = torch.device("cuda" if cuda else "cpu")

    # should add functionality that allows to load a saved vectorizer instead
    dataset = Dataset(config.data_path)
    vectorizer = dataset._vectorizer

    with open('results/training_results.json') as json_file:
        train_state = json.load(json_file)

    model = Classifier(embedding_size=config.embedding_size,
                            num_embeddings=len(vectorizer.text_vocab._idx_to_token),
                            kernel_width = config.kernel_width,
                            num_channels=config.num_channels,
                            hidden_dim=config.hidden_dim,
                            num_classes=len(vectorizer.y_values_vocab._idx_to_token),
                            dropout_p=config.dropout_p,
                            # pretrained_embeddings=embeddings,
                            padding_idx=0)
    # this line will load the embeddings from the saved model
    model.load_state_dict(torch.load(config.save_model_directory+config.model_filename+str(train_state['best_epoch_index'])))
    model = model.to(device)

    text = """there be 2 mass shootings in texas last week but only 1 on tv she leave her husband he kill their child just another day in america"""

    print(text)

    results = predict_category(text, model, vectorizer, dataset._max_seq_length + 1)
    print(results)
