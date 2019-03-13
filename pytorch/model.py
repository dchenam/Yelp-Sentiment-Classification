import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class SimpleLSTMModel(nn.Module):
    """
    Baseline: Single LSTM only
    """

    def __init__(self, embedding_size, hidden_size, dropout_rate, vocab):
        super(SimpleLSTMModel, self).__init__()

        input_size = len(vocab)
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
    """
    Increase LSTM layer and Bidirectional=True
    """

    def __init__(self, embedding_size, hidden_size, dropout_rate, vocab):
        super(ImprovedLSTMModel, self).__init__()

        input_size = len(vocab)
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


class GloveModel(nn.Module):
    """
    Pretrained Embedding + Pooling of LSTM Hidden States
    """

    def __init__(self, embedding_size, hidden_size, dropout_rate, glove):
        super(GloveModel, self).__init__()

        output_size = 5

        self.embedding = nn.Embedding.from_pretrained(glove, freeze=False)
        self.lstm = nn.LSTM(embedding_size, hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(3 * hidden_size, output_size)

    def forward(self, input, lengths):
        embeddings = self.embedding(input)
        embeddings = self.dropout(embeddings)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True)

        lstm_out, (hidden_state, cell_state) = self.lstm(packed)
        lstm_out, lengths = pad_packed_sequence(lstm_out)

        # pool the lengths
        avg_pool = self.dropout(F.adaptive_avg_pool1d(lstm_out.permute((1, 2, 0)), 1).squeeze())
        max_pool = self.dropout(F.adaptive_max_pool1d(lstm_out.permute((1, 2, 0)), 1).squeeze())

        # concat forward and pooled states
        concat = torch.cat((hidden_state[-1, :, :], max_pool, avg_pool), dim=1)
        concat = self.dropout(concat)
        out = self.linear(concat)
        return out
