import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence


class SimpleLSTMModel(nn.Module):
    def __init__(self, embedding_size, hidden_size, dropout_rate, vocab):
        super(SimpleLSTMModel, self).__init__()

        # word2idx size = len(vocab) + 2
        input_size = len(vocab) + 2
        output_size = 5

        self.embedding = nn.Embedding(input_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, input, lengths):
        embeddings = self.embedding(input)
        embeddings = self.dropout(embeddings)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True)
        # lstm_out: tensor containing all the output hidden state, for each timestep. shape: (length, batch, hidden_size)
        # hidden_state: tensor containing the hidden state for last timestep. shape: (1, batch, hidden_size)
        # cell state: tensor containing the cell state for last timestep. shape: (1, batch, hidden_size)
        lstm_out, (hidden_state, cell_state) = self.lstm(packed)
        out = self.linear(hidden_state.squeeze(0))
        return out


class ImprovedLSTMModel(nn.Module):
    def __init__(self, embedding_size, hidden_size, dropout_rate, vocab):
        super(ImprovedLSTMModel, self).__init__()

        input_size = len(vocab) + 2
        output_size = 5

        self.embedding = nn.Embedding(input_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers=2, dropout=dropout_rate, bidirectional=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(2 * hidden_size, output_size)

    def forward(self, input, lengths):
        embeddings = self.embedding(input)
        embeddings = self.dropout(embeddings)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True)

        lstm_out, (hidden_state, cell_state) = self.lstm(packed)

        # concat final forward and backwards and then apply dropout
        hidden_state = self.dropout(torch.cat((hidden_state[-1, :, :], hidden_state[-2, :, :]), dim=1))
        out = self.linear(hidden_state)
        return out
