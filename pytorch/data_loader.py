import string
import torch
import numpy as np
import pandas as pd
import nltk

from nltk.corpus import stopwords
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from tqdm import tqdm

# for progress bar during pandas ops
tqdm.pandas()

stop_words = set(stopwords.words('english') + list(string.punctuation))


# -------------- Helper Functions --------------

def tokenize(text):
    '''
    :param text: a doc with multiple sentences, type: str
    return a word list, type: list
    https://textminingonline.com/dive-into-nltk-part-ii-sentence-tokenize-and-word-tokenize
    e.g.
    Input: 'It is a nice day. I am happy.'
    Output: ['it', 'is', 'a', 'nice', 'day', 'i', 'am', 'happy']
    '''
    tokens = []
    for word in nltk.word_tokenize(text):
        word = word.lower()
        if word not in stop_words and not word.isnumeric():
            tokens.append(word)
    return tokens


def get_sequence(data, seq_length, vocab_dict):
    '''
    :param data: a list of words, type: list
    :param seq_length: the length of sequences,, type: int
    :param vocab_dict: a dict from words to indices, type: dict
    return a dense sequence matrix whose elements are indices of words,
    '''
    data_matrix = np.zeros((len(data), seq_length), dtype=int)
    for i, doc in enumerate(data):
        for j, word in enumerate(doc):
            # YOUR CODE HERE
            if j == seq_length:
                break
            word_idx = vocab_dict.get(word, 1)  # 1 means the unknown word
            data_matrix[i, j] = word_idx
    return data_matrix


# TODO: frequency based Vocab + pruning + GloVe
def build_vocab(sentence_list, vocab=None):
    """
    :param sentence_list: an iterable object with multiple words in each sub-list, type: iterable object
    :param vocab: a dictionary from words to indices, type: dict
    :return: a dictionary from words to indices and indices to words
    """
    if vocab is None:
        vocab = set()
        for sentence in sentence_list:
            for word in sentence:
                vocab.add(word)

    word2idx = dict()
    word2idx['<pad>'] = 0  # 0 means the padding signal
    word2idx['<unk>'] = 1  # 1 means the unknown word
    vocab_size = 2
    for v in vocab:
        word2idx[v] = vocab_size
        vocab_size += 1

    return vocab, word2idx


# ----------------- End of Helper Functions-----------------

class SentimentDataset(Dataset):
    """
    Defines a dataset composed of sentiment text and labels

    Attributes:
        vocab (dict{str: int}: A vocabulary dictionary from word to indices for this dataset
        sample_weights(ndarray, shape(len(labels),)): An array with each sample_weight[i] as the weight of the ith sample
        data (list[int, [int]]): The data in the set
    """

    def __init__(self, path, fix_length=None, vocab=None):
        df = pd.read_csv(path)

        # pre-process
        df['words'] = df["text"].progress_apply(tokenize)

        # take lengths of words, with fixed max length
        df['lengths'] = df['words'].progress_apply(lambda x: fix_length if len(x) > fix_length else len(x))

        # filter out rows with lengths of 0
        df = df.loc[df['lengths'] >= 1]

        # build vocab
        self.vocab, word2idx = build_vocab(df['words'], vocab)

        # change class indices to 0 - 4
        labels = df["stars"].progress_apply(int) - 1

        # pad to fix length & numericalize
        seqs = get_sequence(df['words'], fix_length, word2idx)

        # compute sample weights from inverse class frequencies
        class_sample_count = np.unique(labels, return_counts=True)[1]
        weight = 1. / class_sample_count
        self.samples_weight = torch.from_numpy(weight[labels])

        self.data = list(zip(labels, seqs, df["lengths"]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]


def get_loader(fix_length, batch_size):
    train_dataset = SentimentDataset("data/train.csv", fix_length=fix_length)

    vocab = train_dataset.vocab

    valid_dataset = SentimentDataset("data/valid.csv", fix_length=fix_length, vocab=vocab)

    # test_dataset = SentimentDataset("data/test.csv", fix_length=300, vocab=vocab)

    sampler = WeightedRandomSampler(train_dataset.samples_weight, len(train_dataset.samples_weight))

    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=batch_size,
                                  # sampler=sampler,
                                  shuffle=True,
                                  num_workers=4)

    valid_dataloader = DataLoader(dataset=valid_dataset,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=4)

    return train_dataloader, valid_dataloader, vocab
