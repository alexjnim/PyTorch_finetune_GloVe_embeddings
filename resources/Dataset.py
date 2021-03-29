import torch
from torch.utils.data import Dataset, DataLoader
from resources.Vectorizer import Vectorizer
from sklearn.model_selection import train_test_split
from config import config
import pandas as pd

class Dataset(Dataset):
    def __init__(self, csv_file_path):
        """
        Args:
            data_df (pandas.DataFrame): the dataset
            vectorizer (Vectorizer): vectorizer instatiated from dataset
        """
        self.data_df = pd.read_csv(csv_file_path)
        self.data_df = self.data_df.dropna()

        # +1 if only using begin_seq, +2 if using both begin and end seq tokens
        length = self.data_df[config.text_column].apply(lambda x: len(x.split(" ")))
        self._max_seq_length = max(length)+2

        self.train_df, self.test_df = train_test_split(self.data_df, test_size=0.1, random_state=42, shuffle=True)
        self.train_df, self.val_df = train_test_split(self.train_df, test_size=0.2, random_state=42, shuffle=True)

        self._lookup_dict = {'train': (self.train_df, len(self.train_df)),
                             'val': (self.val_df, len(self.val_df)),
                             'test': (self.test_df, len(self.test_df))}

        self.set_split('train')

        # build vectorizer
        self._vectorizer = Vectorizer.from_dataframe(self.train_df)

        # Class weights
        class_counts = self.data_df[config.y_values_column].value_counts().to_dict()
        def sort_key(item):
            return self._vectorizer.y_values_vocab.lookup_token(item[0])
        sorted_counts = sorted(class_counts.items(), key=sort_key)
        frequencies = [count for _, count in sorted_counts]
        self.class_weights = 1.0 / torch.tensor(frequencies, dtype=torch.float32)

    def save_vectorizer(self, vectorizer_filepath):
        """saves the vectorizer to disk using json

        Args:
            vectorizer_filepath (str): the location to save the vectorizer
        """
        with open(vectorizer_filepath, "w") as fp:
            json.dump(self._vectorizer.to_serializable(), fp)

    def set_split(self, split="train"):
        """ selects the splits in the dataset using a column in the dataframe
        Args:
            split (str): one of "train", "val", or "test"
        """
        self._target_split = split
        self._target_df, self._target_size = self._lookup_dict[split]

    # must have __len__ for Dataset class in pytorch
    def __len__(self):
        return self._target_size

    # must have __getitem__ for Dataset class in pytorch
    def __getitem__(self, index):
        """the primary entry point method for PyTorch datasets

        Args:
            index (int): the index to the data point
        Returns:
            a dictionary holding the data point's features (x_data) and label (y_target)
        """
        row = self._target_df.iloc[index]

        # here we vectorize the input with sequence tokens, where all input_vectors will be a fixed size given by self._max_seq_length
        text_vector = \
            self._vectorizer.vectorize(row[config.text_column], self._max_seq_length)

        y_index = \
            self._vectorizer.y_values_vocab.lookup_token(row.category)

        return {'x_data': text_vector,
                'y_target': y_index}

    # def get_num_batches(self, batch_size):
    #     """Given a batch size, return the number of batches in the dataset

    #     Args:
    #         batch_size (int)
    #     Returns:
    #         number of batches in the dataset
    #     """
    #     return len(self) // batch_size

def generate_batches(dataset, batch_size, shuffle=True,
                     drop_last=True, device="cpu"):
    """
    A generator function which wraps the PyTorch DataLoader. It will
      ensure each tensor is on the write device location.
    """
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size,
                            shuffle=shuffle, drop_last=drop_last)

    for data_dict in dataloader:
        out_data_dict = {}
        for name, tensor in data_dict.items():
            out_data_dict[name] = data_dict[name].to(device)
        yield out_data_dict