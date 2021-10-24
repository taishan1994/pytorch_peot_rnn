# coding:utf8
import torch
import torch.nn as nn
import torch.nn.functional as F


class PoetryModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(PoetryModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim,
                            self.hidden_dim,
                            num_layers=1,
                            batch_first=True,
                            bidirectional=True)
        self.linear = nn.Linear(self.hidden_dim * 2, vocab_size)

    def forward(self, input, hidden=None):
        batch_size, seq_len = input.size()
        if hidden is None:
            #  h_0 = 0.01*torch.Tensor(2, batch_size, self.hidden_dim).normal_().cuda()
            #  c_0 = 0.01*torch.Tensor(2, batch_size, self.hidden_dim).normal_().cuda()
            h_0 = input.data.new(2, batch_size, self.hidden_dim).fill_(0).float()
            c_0 = input.data.new(2, batch_size, self.hidden_dim).fill_(0).float()
        else:
            h_0, c_0 = hidden
        # size: (batch_size,seq_len,embeding_dim)
        embeds = self.embeddings(input)
        # output size: (batch_size,seq_len,hidden_dim)
        output, hidden = self.lstm(embeds, (h_0, c_0))
        # size: (batch_size*seq_len,vocab_size)
        output = self.linear(output.contiguous().view(seq_len * batch_size, -1))
        return output, hidden


class PoetryModel2(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(PoetryModel2, self).__init__()
        self.hidden_dim = hidden_dim
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, self.hidden_dim, num_layers=2, batch_first=True)
        self.linear1 = nn.Linear(self.hidden_dim, vocab_size)

    def forward(self, input, hidden=None):
        batch_size, seq_len = input.size()
        if hidden is None:
            #  h_0 = 0.01*torch.Tensor(2, batch_size, self.hidden_dim).normal_().cuda()
            #  c_0 = 0.01*torch.Tensor(2, batch_size, self.hidden_dim).normal_().cuda()
            h_0 = input.data.new(2, batch_size, self.hidden_dim).fill_(0).float()
            c_0 = input.data.new(2, batch_size, self.hidden_dim).fill_(0).float()
        else:
            h_0, c_0 = hidden
        # size: (seq_len,batch_size,embeding_dim)
        embeds = self.embeddings(input)
        # output size: (seq_len,batch_size,hidden_dim)
        output, hidden = self.lstm(embeds, (h_0, c_0))

        # size: (seq_len*batch_size,vocab_size)
        output = self.linear1(output.contiguous().view(seq_len * batch_size, -1))
        return output, hidden


if __name__ == '__main__':
    input = torch.tensor([[2, 13, 15, 20], [2, 13, 15, 20]]).long()
    model = PoetryModel2(10000, 300, 256)
    ouput, hidden = model(input)
