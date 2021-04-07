import pandas as pd
import utils.TextNormalizer as tn
from utils.Contractions import CONTRACTION_MAP
from config import config

def clean_data():
    data_df = pd.read_json(r'data/News_Category_Dataset_v2.json', lines = True)

    data_df['text'] = data_df['headline'] + ". " + data_df['short_description']

    norm_corpus = tn.normalize_corpus(corpus=data_df['text'], html_stripping=True,
                                    contraction_expansion=True, accented_char_removal=True,
                                    text_lower_case=True, text_lemmatization=True,
                                    text_stemming=False, special_char_removal=True,
                                    remove_digits=False, stopword_removal=False)
    data_df['clean_text'] = norm_corpus

    clean_data_df = data_df[['category','clean_text']]
    clean_data_df = clean_data_df.dropna()
    clean_data_df.to_csv('data/clean_News_Category_Dataset_v2.csv', index=False)