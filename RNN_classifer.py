import torch
import torch.nn as nn

class RNN_Classifier(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers = 1, batch_first = True, use_gpu = True):
        super(RNN_Classifier, self).__init__()
        self.hidden_size = hidden_size 
        self.input_size = input_size
        self.output_size = output_size
        self.num_layers = num_layers 
        self.rnn = nn.RNN(self.input_size, self.hidden_size, num_layers, batch_first = batch_first)
        self.fc = nn.Linear(self.hidden_size, self.output_size)
        self.softmax = torch.nn.Softmax(dim=1)
        self.use_gpu = True
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        if self.use_gpu:
            h0 = h0.cuda()
        # c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        # print(type(x))
        # print(type(h0))
        out, hidden = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        # out = self.softmax(out)
        return out
