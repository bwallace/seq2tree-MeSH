import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

USE_CUDA = False

class Encoder(nn.Module):
    '''
    Encoder model; yields H, a sequence of hidden
    representations over the input.
    '''
    def __init__(self, input_dim, hidden_dim, n_layers=1): 
        super(Encoder, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(self.input_dim, self.hidden_dim)
        self.gru = nn.GRU(self.hidden_dim, self.hidden_dim, self.n_layers)


    def forward(self, word_inputs, initial_hidden):
        seq_len = len(word_inputs)
        embedded = self.embedding(word_inputs).view(seq_len, 1, -1)
        output, hidden = self.gru(embedded, initial_hidden)
        return output, hidden

    def init_hidden(self):
        hidden = Variable(torch.zeros(self.n_layers, 1, self.hidden_dim))
        if USE_CUDA: 
            hidden = hidden.cuda()
        return hidden


class Attn(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        
        self.method = method
        self.hidden_size = hidden_size
        
        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)

        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(1, hidden_size))

    def forward(self, hidden, encoder_outputs):
        seq_len = len(encoder_outputs)

        # create variable to store attention energies
        attn_energies = Variable(torch.zeros(seq_len)) # B x 1 x S
        if USE_CUDA: 
            attn_energies = attn_energies.cuda()

        # calculate energies for each encoder output
        for i in range(seq_len):
            attn_energies[i] = self.score(hidden, encoder_outputs[i])

        # normalize energies to weights in range 0 to 1, resize to 1 x 1 x seq_len
        return F.softmax(attn_energies).unsqueeze(0).unsqueeze(0)
    

    def score(self, hidden, encoder_output):
            if self.method == 'dot':
                energy =torch.dot(hidden.view(-1), encoder_output.view(-1))
            elif self.method == 'general':
                energy = self.attn(encoder_output)
                energy = torch.dot(hidden.view(-1), energy.view(-1))
            elif self.method == 'concat':
                energy = self.attn(torch.cat((hidden, encoder_output), 1))
                energy = torch.dot(self.v.view(-1), energy.view(-1))
            return energy

class AttnDecoderRNN(nn.Module):
    def __init__(self, attn_model, hidden_size, output_size, n_layers=1, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()
        
        # keep parameters for reference
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        
        # define layers
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size * 2, hidden_size, n_layers, dropout=dropout_p)
        self.out = nn.Linear(hidden_size * 2, output_size)
        
        # set attention model
        if attn_model != 'none':
            self.attn = Attn(attn_model, hidden_size)
    
    def forward(self, word_input, last_context, last_hidden, encoder_outputs):
        # word_input here is the last thing we predicted
        word_embedded = self.embedding(word_input).view(1, 1, -1) 
        
        # feed last hidden and current context vector to RNN
        rnn_input = torch.cat((word_embedded, last_context.unsqueeze(0)), 2)

        # dims are wrong here (???). 
        # last_hidden is (1, 1, 1000) but apparently should be (2, 1, 500)
        #import pdb; pdb.set_trace()
        rnn_output, hidden = self.gru(rnn_input, last_hidden)

        # calculate attention from current RNN state and all encoder outputs; 
        # apply to encoder outputs
        attn_weights = self.attn(rnn_output.squeeze(0), encoder_outputs)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1)) 
        
        # final output layer (next word prediction) using the RNN hidden state and context vector
        rnn_output = rnn_output.squeeze(0) # 1 x B x N -> B x N
        context = context.squeeze(1)       # B x 1 x N -> B x N
        output = F.log_softmax(self.out(torch.cat((rnn_output, context), 1)))
        
        return output, context, hidden, attn_weights

class GraphDecoder(nn.Module):
    pass 