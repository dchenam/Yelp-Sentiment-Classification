import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from regularization import WeightDropout


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
        self.attention = WordSentenceAttention(2 * hidden_size)
        self.linear = nn.Linear(2 * hidden_size, output_size)

    def forward(self, input, lengths):
        embeddings = self.embedding(input)
        embeddings = self.dropout(embeddings)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True)

        lstm_out, (hidden_state, cell_state) = self.lstm(packed)

        lstm_out, lengths = pad_packed_sequence(lstm_out, batch_first=True,
                                                total_length=300)  # gru_out: (batch, length, 2 * hidden_size)

        sentence = self.attention(lstm_out)
        out = self.linear(sentence)
        # concat final forward and backwards and then apply dropout
        # hidden_state = self.dropout(torch.cat((hidden_state[-1, :, :], hidden_state[-2, :, :]), dim=1))
        # out = self.linear(hidden_state)
        return out


class GloveModel(nn.Module):
    """
    Pretrained Embedding + Pooling of LSTM Hidden States
    """

    def __init__(self, embedding_size, hidden_size, dropout_rate, glove):
        super(GloveModel, self).__init__()

        output_size = 5

        self.embedding = nn.Embedding.from_pretrained(glove)
        self.lstm = nn.LSTM(embedding_size, hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(3 * hidden_size, output_size)

    def forward(self, input, lengths):
        embeddings = self.embedding(input)
        embeddings = self.dropout(embeddings)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True)

        lstm_out, (hidden_state, cell_state) = self.lstm(packed, num_layers=2)
        lstm_out, lengths = pad_packed_sequence(lstm_out)

        # pool the lengths
        avg_pool = F.adaptive_avg_pool1d(lstm_out.permute((1, 2, 0)), 1).squeeze()
        max_pool = F.adaptive_max_pool1d(lstm_out.permute((1, 2, 0)), 1).squeeze()

        # concat forward and pooled states
        concat = torch.cat((hidden_state[-1, :, :], max_pool, avg_pool), dim=1)
        out = self.linear(concat)
        return out


class AWDModel(nn.Module):
    def __init__(self, embedding_size, hidden_size, dropout_rate, fasttext):
        super(AWDModel, self).__init__()

        output_size = 5

        self.embedding = nn.Embedding.from_pretrained(fasttext)
        self.lstm = WeightDropout(nn.LSTM(embedding_size, hidden_size, num_layers=3),
                                  name_w=('weight_hh_l0', 'weight_hh_l1', 'weight_hh_l2'))
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(3 * hidden_size, output_size)

    def forward(self, input, lengths):
        embeddings = self.embedding(input)
        embeddings = self.dropout(embeddings)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True)

        lstm_out, (hidden_state, cell_state) = self.lstm(packed)
        lstm_out, lengths = pad_packed_sequence(lstm_out)

        # pool the lengths
        avg_pool = F.adaptive_avg_pool1d(lstm_out.permute((1, 2, 0)), 1).squeeze()
        max_pool = F.adaptive_max_pool1d(lstm_out.permute((1, 2, 0)), 1).squeeze()

        # concat forward and pooled states
        concat = torch.cat((hidden_state[-1, :, :], max_pool, avg_pool), dim=1)
        out = self.linear(concat)
        return out


class HierarchicalAttentionModel(nn.Module):
    def __init__(self, embedding_size, hidden_size, dropout_rate, fasttext):
        super(HierarchicalAttentionModel, self).__init__()

        output_size = 5

        self.embedding = nn.Embedding.from_pretrained(fasttext)
        self.lstm = WeightDropout(nn.LSTM(embedding_size, hidden_size, num_layers=2, bidirectional=True),
                                  name_w=('weight_hh_l0', 'weight_hh_l0_reverse',
                                          'weight_hh_l1', 'weight_hh_l1_reverse'))

        self.attention = WordSentenceAttention(2 * hidden_size)
        self.linear = nn.Linear(2 * hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input, lengths):
        embeddings = self.embedding(input)
        embeddings = self.dropout(embeddings)

        packed = pack_padded_sequence(embeddings, lengths, batch_first=True)
        lstm_out, _ = self.lstm(packed)
        lstm_out, lengths = pad_packed_sequence(lstm_out, batch_first=True,
                                                total_length=300)  # gru_out: (batch, length, 2 * hidden_size)

        sentence = self.attention(lstm_out)
        out = self.linear(sentence)
        return out


class WordSentenceAttention(nn.Module):
    def __init__(self, hidden_size):
        super(WordSentenceAttention, self).__init__()
        self.context_weight = nn.Parameter(torch.Tensor(hidden_size).uniform_(-0.1, 0.1))
        self.context_projection = nn.Linear(hidden_size, hidden_size)
        self.softmax = nn.Softmax()

    def forward(self, context):
        s1, s2, s3 = context.size()  # batch, length, hidden size

        # create context projection
        context_projection = torch.tanh(self.context_projection(context))  # (batch_size, length, hidden_size)

        # calculate similarity value with context weight
        attn_score = self.softmax(context_projection.matmul(self.context_weight))

        # reshape attention score and context to perform a element-wise multipication
        attn_score = attn_score.view(s1 * s2).expand(s3, s1 * s2).reshape([s1 * s3, s2])
        context = context.permute(2, 0, 1).reshape([s1 * s3, s2])
        sentence = (context * attn_score).sum(1).view(s3, s1).transpose(0, 1)
        return sentence
