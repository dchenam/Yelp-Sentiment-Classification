import string
import re
import torch
import nltk
import numpy as np
import pandas as pd

from collections import Counter
from tqdm import tqdm
from torchtext import vocab
from torch.utils.data import Dataset, DataLoader

# for progress bar during pandas ops
tqdm.pandas()

stop_words = set(nltk.corpus.stopwords.words('english') + list(string.punctuation))


# -------------- Helper Functions --------------

def preprocess(text):
    "Add spaces around / and #"
    text = re.sub(r'([/#\n])', r' \1 ', text)
    "Remove extra spaces"
    text = re.sub(' {2,}', ' ', text)
    "Removes any repeated characters > 2 to 2"
    text = re.sub(r'(.)\1+', r'\1\1', text)
    "Remove any numbers and words mixed within them"
    text = re.sub(r'\w*\d\w*', '', text).strip()
    "Remove 's -"
    return text.replace("'s", "").replace("-", "")


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
    for word in nltk.casual_tokenize(text, preserve_case=False):
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


def build_vocab(sentence_list, threshold, vocab=None):
    """
    :param sentence_list: an iterable object with multiple words in each sub-list, type: iterable object
    :param threshold: minimum number of a word's count to be included into the vocabulary object type: int
    :param vocab: a Vocab object, type: object
    :return: a dictionary from words to indices and indices to words
    """
    counter = Counter()
    for sentence in sentence_list:
        counter.update(sentence)

    # sort by most common
    word_count = sorted(counter.items(), key=lambda x: x[1], reverse=True)

    # exclude words that are below a frequency
    words = [word for word, count in word_count if count > threshold]

    if vocab is None:
        vocab = Vocab()
        vocab.add_word('<pad>')  # 0 means the padding signal
        vocab.add_word('<unk>')  # 1 means the unknown word

    # add the words to the vocab
    for word in words:
        vocab.add_word(word)

    return vocab


class Vocab(object):
    def __init__(self):
        self.word2idx = dict()
        self.vocab_size = 0

    def __len__(self):
        return len(self.word2idx)

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.vocab_size
            self.vocab_size += 1

    def get_embedding(self, name, embedding_dim):
        if name == 'glove':
            pretrained_type = vocab.GloVe(name='42B', dim=embedding_dim)
        elif name == 'fasttext':
            if embedding_dim != 300:
                raise ValueError("Got embedding dim {}, expected size 300".format(embedding_dim))
            pretrained_type = vocab.FastText('en')

        embedding_len = len(self)
        weights = np.zeros((embedding_len, embedding_dim))
        words_found = 0

        for word, index in self.word2idx.items():
            try:
                # torchtext.vocab.__getitem__ defaults key error to a zero vector
                weights[index] = pretrained_type.vectors[pretrained_type.stoi[word]]
                words_found += 1
            except KeyError:
                if index == 0:
                    continue
                weights[index] = np.random.normal(scale=0.6, size=(embedding_dim))

        print(embedding_len - words_found, "words missing from pretrained")
        return torch.from_numpy(weights).float()


# ----------------- End of Helper Functions-----------------

class SentimentDataset(Dataset):
    """
    Defines a dataset composed of sentiment text and labels

    Attributes:
        df (Dataframe): Dataframe of the CSV from teh path
        vocab (dict{str: int}: A vocabulary dictionary from word to indices for this dataset
        sample_weights(ndarray, shape(len(labels),)): An array with each sample_weight[i] as the weight of the ith sample
        data (list[int, [int]]): The data in the set
    """

    def __init__(self, path, fix_length, threshold, vocab=None):
        df = pd.read_csv(path)

        self.df = df

        # preprocess
        df["text"] = df["text"].progress_apply(preprocess)

        # tokenize
        df['words'] = df["text"].progress_apply(tokenize)

        # take lengths of words, with fixed max length
        df['lengths'] = df['words'].apply(lambda x: fix_length if len(x) > fix_length else len(x))

        # filter out rows with lengths of 0
        df = df.loc[df['lengths'] >= 1]

        # build vocab
        self.vocab = build_vocab(df['words'], threshold, vocab)

        # change class indices to 0 - 4
        labels = df["stars"].apply(int) - 1

        # pad to fix length & numericalize
        seqs = get_sequence(df['words'], fix_length, self.vocab.word2idx)

        # compute sample weights from inverse class frequencies
        class_sample_count = np.unique(labels, return_counts=True)[1]
        weight = 1. / class_sample_count
        self.samples_weight = torch.from_numpy(weight[labels])

        self.data = list(zip(labels, seqs, df["lengths"]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]


def get_loader(fix_length, vocab_threshold, batch_size):
    train_dataset = SentimentDataset("data/train.csv", fix_length, vocab_threshold)

    vocab = train_dataset.vocab

    valid_dataset = SentimentDataset("data/valid.csv", fix_length, vocab_threshold, vocab)

    test_dataset = SentimentDataset("data/test.csv", fix_length, vocab_threshold, vocab)

    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=4)

    valid_dataloader = DataLoader(dataset=valid_dataset,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=4)
    test_dataloader = DataLoader(dataset=test_dataset,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=4)

    return train_dataloader, valid_dataloader, test_dataloader, vocab
