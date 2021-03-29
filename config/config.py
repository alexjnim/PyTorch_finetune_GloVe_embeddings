
original_data_path = 'data/News_Category_Dataset_v2.csv'
data_path='data/clean_News_Category_Dataset_v2.csv'
text_column = 'clean_text'
y_values_column = 'category'
unk_words_cutoff = 25

save_model_directory = 'model/saved_models/'
model_filename = 'embedding_model_'
save_embeddings_directory = 'model/save_embeddings/'
embeddings_filename = 'embeddings_'
vectorizer_directory = 'model/saved_vectorizers/'
vectorizer_filename = 'vectorizer.json'

reload_from_files = False

# Model hyper parameters
glove_filepath='data/glove/glove.6B.50d.txt'
use_glove=True
# embedding_size=50 will change in the code if use_glove = True, being replaced by the size of loaded GloVe embeddings - glove.6B.50d is 50 anyway
embedding_size=50
hidden_dim=100
num_channels=100
kernel_width = 3

# Training hyper parameter
seed=1337
# after 2 steps of the validation loss not improvin, the training will stop
num_epochs=1
early_stopping_best_val = 1e8
learning_rate=0.001
dropout_p=0.1
batch_size=300
early_stopping_criteria=5

# Runtime option
cuda=True
# catch_keyboard_interrupt=True,
# reload_from_files=False,
# expand_filepaths_to_save_dir=True
